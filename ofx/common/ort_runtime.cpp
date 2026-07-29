#include "ort_runtime.h"

#include <chrono>
#include <sstream>
#include <vector>

namespace iw3
{

namespace
{
std::string narrow(const std::wstring& text)
{
    if (text.empty())
    {
        return {};
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), int(text.size()), nullptr, 0, nullptr, nullptr);
    std::string out(size_t(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), int(text.size()), out.data(), size, nullptr, nullptr);
    return out;
}
}  // namespace

OrtRuntime::~OrtRuntime()
{
    if (_api)
    {
        for (Model& model : _models)
        {
            if (model.binding) _api->ReleaseIoBinding(model.binding);
            if (model.session) _api->ReleaseSession(model.session);
        }
        if (_memoryInfo) _api->ReleaseMemoryInfo(_memoryInfo);
        if (_cudaMemoryInfo) _api->ReleaseMemoryInfo(_cudaMemoryInfo);
        if (_env) _api->ReleaseEnv(_env);
    }
    // The library is deliberately not freed: Resolve keeps plugins loaded for
    // the session, and unloading a runtime that has a live CUDA context while
    // the host is still using the GPU is a good way to crash on exit.
    //
    // In practice this destructor never runs at all -- see the note on
    // sharedRuntime() in plugin/iw3stereo.cpp.
}

std::vector<std::string> OrtRuntime::takeNewReport()
{
    std::vector<std::string> lines;
    if (_reportCursor < _report.size())
    {
        lines.assign(_report.begin() + ptrdiff_t(_reportCursor), _report.end());
        _reportCursor = _report.size();
    }
    return lines;
}

bool OrtRuntime::failed(OrtStatus* status, const char* what)
{
    if (!status)
    {
        return false;
    }
    note(std::string(what) + ": " + _api->GetErrorMessage(status));
    _api->ReleaseStatus(status);
    return true;
}

bool OrtRuntime::open(const std::wstring& directory, const std::vector<std::wstring>& modelPaths,
                      bool preferCuda, const std::vector<bool>& conserveMemory)
{
    const std::wstring dllPath = directory + L"\\onnxruntime.dll";
    note("loading " + narrow(dllPath));

    // The provider DLLs sit beside onnxruntime.dll; this is what makes them
    // resolve from our folder instead of Resolve's.
    _library = LoadLibraryExW(dllPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!_library)
    {
        note("LoadLibraryExW failed, GetLastError=" + std::to_string(GetLastError()));
        return false;
    }

    wchar_t loadedFrom[MAX_PATH] = {};
    GetModuleFileNameW(_library, loadedFrom, MAX_PATH);
    note("actually loaded: " + narrow(loadedFrom));

    auto getApiBase = reinterpret_cast<const OrtApiBase* (ORT_API_CALL*)()>(
        GetProcAddress(_library, "OrtGetApiBase"));
    if (!getApiBase)
    {
        note("no OrtGetApiBase export");
        return false;
    }

    const OrtApiBase* base = getApiBase();
    _version = base->GetVersionString();
    note("runtime version: " + _version);

    _api = base->GetApi(ORT_API_VERSION);
    if (!_api)
    {
        note("GetApi(" + std::to_string(ORT_API_VERSION) + ") returned null -- runtime too old");
        return false;
    }

    if (failed(_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "iw3", &_env), "CreateEnv"))
    {
        return false;
    }

    char** providers = nullptr;
    int providerCount = 0;
    if (!failed(_api->GetAvailableProviders(&providers, &providerCount), "GetAvailableProviders"))
    {
        std::ostringstream line;
        line << "available providers:";
        for (int i = 0; i < providerCount; ++i)
        {
            line << " " << providers[i];
        }
        note(line.str());
        _api->ReleaseAvailableProviders(providers, providerCount);
    }

    // Session options are rebuilt per model: appending a provider consumes the
    // options object's state, and reusing one across sessions is asking for
    // trouble.
    bool cudaAttached = false;
    for (size_t index = 0; index < modelPaths.size(); ++index)
    {
        OrtSessionOptions* options = nullptr;
        if (failed(_api->CreateSessionOptions(&options), "CreateSessionOptions"))
        {
            return false;
        }

        const bool conserve = index < conserveMemory.size() && conserveMemory[index];
        if (conserve)
        {
            // Memory pattern planning pre-allocates from the shapes of the
            // first run, and for a graph with dynamic shapes and very large
            // intermediates it reserves far more than the run needs and never
            // gives it back. Measured on the twelve-frame inpaint graph at HD:
            // 15.71 GiB and 16.4 s a window with it on, 10.08 GiB and 0.32 s
            // with it off. On the per-frame graph, 2.26 GiB against 1.19 GiB
            // for four hundred microseconds an eye.
            //
            // Left on for the warp graphs, which are small and run at one shape
            // per session, where it is a help rather than a hindrance.
            failed(_api->DisableMemPattern(options), "DisableMemPattern");
        }

        bool wantCuda = false;
        if (preferCuda)
        {
            const auto started = std::chrono::steady_clock::now();
            OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
            if (!failed(_api->CreateCUDAProviderOptions(&cudaOptions), "CreateCUDAProviderOptions"))
            {
                // TF32 is on by default and costs three orders of magnitude of
                // accuracy on these models for well under a millisecond.
                //
                // cudnn_conv_algo_search defaults to EXHAUSTIVE, which
                // benchmarks every convolution algorithm the first time it
                // meets a shape. That was most of the 450 ms the first frame
                // used to cost.
                const char* keys[] = {"device_id", "use_tf32", "cudnn_conv_algo_search"};
                const char* values[] = {"0", "0", "HEURISTIC"};
                failed(_api->UpdateCUDAProviderOptions(cudaOptions, keys, values, 3),
                       "UpdateCUDAProviderOptions");
                if (!failed(_api->SessionOptionsAppendExecutionProvider_CUDA_V2(options, cudaOptions),
                            "SessionOptionsAppendExecutionProvider_CUDA_V2"))
                {
                    wantCuda = true;
                }
                _api->ReleaseCUDAProviderOptions(cudaOptions);
            }
            if (index == 0)
            {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                note(std::string("CUDA provider attach: ") + (wantCuda ? "OK" : "FAILED") +
                     " in " + std::to_string(ms) + " ms");
            }
        }

        const std::wstring& path = modelPaths[index];
        note("opening " + narrow(path));
        Model model;
        const auto sessionStart = std::chrono::steady_clock::now();
        OrtStatus* status = _api->CreateSession(_env, path.c_str(), options, &model.session);
        if (status)
        {
            failed(status, "CreateSession");
            _api->ReleaseSessionOptions(options);
            if (index == 0)
            {
                return false;
            }
            // A missing optional model is not fatal; the plugin falls back to
            // the first one and says so.
            note("continuing without " + narrow(path));
            continue;
        }
        const double sessionMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sessionStart).count();
        note("CreateSession OK in " + std::to_string(sessionMs) + " ms");
        _api->ReleaseSessionOptions(options);

        if (index == 0)
        {
            cudaAttached = wantCuda;
            _provider = cudaAttached ? "CUDA" : "CPU";
            note("provider in use: " + _provider);

            if (failed(_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &_memoryInfo),
                       "CreateCpuMemoryInfo"))
            {
                return false;
            }
            if (cudaAttached)
            {
                // Device-resident tensors. If any of this fails the sessions
                // still work through run(); they just pay for the copies.
                if (failed(_api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault,
                                                  &_cudaMemoryInfo), "CreateMemoryInfo(Cuda)") ||
                    failed(_api->GetAllocatorWithDefaultOptions(&_allocator),
                           "GetAllocatorWithDefaultOptions"))
                {
                    if (_cudaMemoryInfo)
                    {
                        _api->ReleaseMemoryInfo(_cudaMemoryInfo);
                        _cudaMemoryInfo = nullptr;
                    }
                }
            }
        }

        _models.push_back(model);
        if (!readInputNames(_models.back()) || !readOutputNames(_models.back()))
        {
            note("could not read input/output names for " + narrow(path));
        }
        else
        {
            note(narrow(path) + ": " + std::to_string(_models.back().outputNames.size()) +
                 " output(s)");
        }
        if (_cudaMemoryInfo && !bindOutputs(_models.back()))
        {
            note("device binding unavailable for " + narrow(path) + "; using host copies");
        }
    }

    if (_models.empty())
    {
        return false;
    }
    if (_cudaMemoryInfo)
    {
        note("device binding ready: inputs and outputs stay on the GPU");
    }

    for (size_t index = 0; index < _models.size(); ++index)
    {
        warmUp(index);
    }
    return true;
}

bool OrtRuntime::readInputNames(Model& model)
{
    size_t count = 0;
    if (failed(_api->SessionGetInputCount(model.session, &count), "SessionGetInputCount"))
    {
        return false;
    }
    OrtAllocator* allocator = nullptr;
    if (failed(_api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions"))
    {
        return false;
    }
    for (size_t i = 0; i < count; ++i)
    {
        char* name = nullptr;
        if (failed(_api->SessionGetInputName(model.session, i, allocator, &name),
                   "SessionGetInputName"))
        {
            return false;
        }
        model.inputNames.push_back(name);
        allocator->Free(allocator, name);
    }

    // The element type of input 0 stands for the graph: every input of these
    // graphs has the same precision.
    OrtTypeInfo* info = nullptr;
    if (count > 0 && !failed(_api->SessionGetInputTypeInfo(model.session, 0, &info),
                             "SessionGetInputTypeInfo"))
    {
        const OrtTensorTypeAndShapeInfo* tensor = nullptr;
        if (!failed(_api->CastTypeInfoToTensorInfo(info, &tensor), "CastTypeInfoToTensorInfo") &&
            tensor)
        {
            failed(_api->GetTensorElementType(tensor, &model.inputType), "GetTensorElementType");
        }
        _api->ReleaseTypeInfo(info);
    }
    return true;
}

bool OrtRuntime::inputIsHalf(size_t model) const
{
    return model < _models.size() &&
           _models[model].inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

bool OrtRuntime::readOutputNames(Model& model)
{
    size_t count = 0;
    if (failed(_api->SessionGetOutputCount(model.session, &count), "SessionGetOutputCount"))
    {
        return false;
    }
    OrtAllocator* allocator = nullptr;
    if (failed(_api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions"))
    {
        return false;
    }
    for (size_t i = 0; i < count; ++i)
    {
        char* name = nullptr;
        if (failed(_api->SessionGetOutputName(model.session, i, allocator, &name),
                   "SessionGetOutputName"))
        {
            return false;
        }
        model.outputNames.push_back(name);
        allocator->Free(allocator, name);
    }
    return true;
}

size_t OrtRuntime::outputCount(size_t model) const
{
    return model < _models.size() ? _models[model].outputNames.size() : 0;
}

bool OrtRuntime::bindOutputs(Model& model)
{
    if (model.outputNames.empty())
    {
        return false;
    }
    if (failed(_api->CreateIoBinding(model.session, &model.binding), "CreateIoBinding"))
    {
        model.binding = nullptr;
        return false;
    }
    if (!rebindOutputs(model))
    {
        _api->ReleaseIoBinding(model.binding);
        model.binding = nullptr;
        return false;
    }
    return true;
}

bool OrtRuntime::rebindOutputs(Model& model)
{
    // Outputs land in device memory rather than being copied back. Bound by
    // the names the graph actually declares, so the warp's two and the
    // inpaint's one both work without this knowing which is which.
    //
    // Cleared and rebound before *every* run, which is not belt and braces. A
    // binding keeps the buffer ORT allocated for it on the previous run and
    // offers it back as a pre-allocated output; ORT then refuses any run whose
    // computed output shape differs, with "the output OrtValue provided for
    // output 'y' has shape {1,3,128,128} but the computed output shape for this
    // run is {1,3,1036,1920}". Binding to the device again drops that buffer and
    // lets ORT size the next one itself.
    //
    // This bit the inpaint graph immediately, because its warm-up is the only
    // one that goes through the binding, at 128x128. The warp graphs were one
    // timeline resolution change away from the same failure and had simply
    // never been asked to do it.
    _api->ClearBoundOutputs(model.binding);
    for (const std::string& name : model.outputNames)
    {
        if (failed(_api->BindOutputToDevice(model.binding, name.c_str(), _cudaMemoryInfo),
                   "BindOutputToDevice"))
        {
            return false;
        }
    }
    return true;
}

void OrtRuntime::warmUp(size_t index)
{
    if (index >= _models.size() || !_models[index].session)
    {
        return;
    }

    // The first Run of a session pays for loading cuDNN and cuBLAS kernel
    // modules and picking algorithms. Spending it here, off the render thread,
    // is the whole point -- open() is called from a background thread while the
    // user is still wiring the node up.
    //
    // The shape does not have to match the real frame. Most of the cost is
    // per-session rather than per-shape, and what remains is per-shape only
    // through cuDNN's algorithm cache, which HEURISTIC keeps cheap.
    constexpr int kSize = 128;
    constexpr size_t kElements = size_t(kSize) * kSize * 3;
    const int64_t shape[4] = {1, 3, kSize, kSize};

    const auto started = std::chrono::steady_clock::now();
    bool ok = false;

    if (_models[index].outputNames.size() == 1)
    {
        // The inpaint graph. Its inputs go in as host tensors and ORT copies
        // them, which is fine here -- paying for a copy off the render thread
        // is the entire purpose of a warm-up.
        if (!_models[index].binding)
        {
            note("warm-up skipped for [" + std::to_string(index) +
                 "]: no device binding (the inpaint graph is GPU-only)");
            return;
        }
        // Sized in bytes rather than floats, because the graph may be half.
        const size_t width = inputIsHalf(index) ? sizeof(uint16_t) : sizeof(float);
        std::vector<char> eye(kElements * width, 0);
        std::vector<char> mask(size_t(kSize) * kSize * width, 0);
        const float* filled = nullptr;
        ok = runInpaintWith(index, _memoryInfo,
                            reinterpret_cast<const float*>(eye.data()),
                            reinterpret_cast<const float*>(mask.data()), shape, &filled);
    }
    else
    {
        std::vector<float> image(kElements, 0.5f);
        std::vector<float> x(kElements, 0.5f);
        std::vector<float> left(kElements);
        std::vector<float> right(kElements);
        ok = run(index, image.data(), shape, x.data(), shape,
                 1.0f / float(kSize / 2 - 1),
                 left.data(), right.data(), kElements);
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    note("warm-up run [" + std::to_string(index) + "]: " + std::to_string(ms) + " ms" +
         (ok ? "" : " (FAILED)"));
}

bool OrtRuntime::run(size_t index,
                     const float* image, const int64_t* imageShape,
                     const float* x, const int64_t* xShape,
                     float deltaScale,
                     float* left, float* right, size_t imageCount)
{
    if (index >= _models.size() || !_models[index].session)
    {
        return false;
    }

    const size_t imageElements = size_t(imageShape[0] * imageShape[1] * imageShape[2] * imageShape[3]);
    const size_t xElements = size_t(xShape[0] * xShape[1] * xShape[2] * xShape[3]);

    OrtValue* inputs[3] = {nullptr, nullptr, nullptr};
    OrtValue* outputs[2] = {nullptr, nullptr};
    bool ok = false;

    if (!failed(_api->CreateTensorWithDataAsOrtValue(
            _memoryInfo, const_cast<float*>(image), imageElements * sizeof(float),
            imageShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]), "image tensor") &&
        !failed(_api->CreateTensorWithDataAsOrtValue(
            _memoryInfo, const_cast<float*>(x), xElements * sizeof(float),
            xShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[1]), "x tensor") &&
        // delta_scale is a scalar: rank 0, not a 1-element vector.
        !failed(_api->CreateTensorWithDataAsOrtValue(
            _memoryInfo, &deltaScale, sizeof(float),
            nullptr, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[2]), "delta_scale tensor"))
    {
        const char* inputNames[] = {"image", "x", "delta_scale"};
        const char* outputNames[] = {"left", "right"};

        const auto started = std::chrono::steady_clock::now();
        OrtStatus* status = _api->Run(_models[index].session, nullptr,
                                      inputNames, inputs, 3, outputNames, 2, outputs);
        _lastRunMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        if (!failed(status, "Run"))
        {
            float* leftData = nullptr;
            float* rightData = nullptr;
            if (!failed(_api->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&leftData)), "left data") &&
                !failed(_api->GetTensorMutableData(outputs[1], reinterpret_cast<void**>(&rightData)), "right data"))
            {
                memcpy(left, leftData, imageCount * sizeof(float));
                memcpy(right, rightData, imageCount * sizeof(float));
                ok = true;
            }
        }
    }

    for (OrtValue* value : inputs)
    {
        if (value) _api->ReleaseValue(value);
    }
    for (OrtValue* value : outputs)
    {
        if (value) _api->ReleaseValue(value);
    }
    return ok;
}

void OrtRuntime::releaseBoundOutputs(Model& model)
{
    if (!model.boundOutputs)
    {
        return;
    }
    for (size_t i = 0; i < model.boundOutputCount; ++i)
    {
        if (model.boundOutputs[i]) _api->ReleaseValue(model.boundOutputs[i]);
    }
    _api->AllocatorFree(_allocator, model.boundOutputs);
    model.boundOutputs = nullptr;
    model.boundOutputCount = 0;
}

bool OrtRuntime::runInpaintDevice(size_t index,
                                  const float* eye, const float* mask, const int64_t* shape,
                                  const float** filled)
{
    if (!deviceCapable())
    {
        return false;
    }
    return runInpaintWith(index, _cudaMemoryInfo, eye, mask, shape, filled);
}

bool OrtRuntime::runInpaintWith(size_t index, OrtMemoryInfo* memoryInfo,
                                const float* eye, const float* mask, const int64_t* shape,
                                const float** filled)
{
    if (index >= _models.size() || !_models[index].binding || !memoryInfo)
    {
        return false;
    }
    Model& model = _models[index];

    // The previous frame's outputs are only valid until here.
    releaseBoundOutputs(model);
    if (!rebindOutputs(model))
    {
        return false;
    }

    const size_t eyeElements = size_t(shape[0] * shape[1] * shape[2] * shape[3]);
    // Same geometry, one channel: the mask is per pixel, not per plane.
    const int64_t maskShape[4] = {shape[0], 1, shape[2], shape[3]};
    const size_t maskElements = size_t(shape[0] * shape[2] * shape[3]);

    // The graph's own element type, not an assumption. Both inpaint graphs are
    // half; the caller has already cast its buffers to match, and passing the
    // wrong width here would be read as garbage rather than rejected.
    const ONNXTensorElementDataType type = model.inputType;
    const size_t elementSize =
        (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? sizeof(uint16_t) : sizeof(float);

    OrtValue* eyeValue = nullptr;
    OrtValue* maskValue = nullptr;
    bool ok = false;

    if (!failed(_api->CreateTensorWithDataAsOrtValue(
            memoryInfo, const_cast<float*>(eye), eyeElements * elementSize,
            shape, 4, type, &eyeValue), "inpaint eye tensor") &&
        !failed(_api->CreateTensorWithDataAsOrtValue(
            memoryInfo, const_cast<float*>(mask), maskElements * elementSize,
            maskShape, 4, type, &maskValue), "inpaint mask tensor"))
    {
        // By position, not by literal name: the per-frame graph calls them
        // eye/mask and the twelve-frame one eyes/masks.
        if (model.inputNames.size() == 2 &&
            !failed(_api->BindInput(model.binding, model.inputNames[0].c_str(), eyeValue),
                    "BindInput(eye)") &&
            !failed(_api->BindInput(model.binding, model.inputNames[1].c_str(), maskValue),
                    "BindInput(mask)"))
        {
            const auto started = std::chrono::steady_clock::now();
            OrtStatus* status = _api->RunWithBinding(model.session, nullptr, model.binding);
            _lastRunMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();

            if (!failed(status, "RunWithBinding(inpaint)") &&
                !failed(_api->GetBoundOutputValues(model.binding, _allocator,
                                                   &model.boundOutputs, &model.boundOutputCount),
                        "GetBoundOutputValues") &&
                model.boundOutputCount == 1)
            {
                float* data = nullptr;
                if (!failed(_api->GetTensorMutableData(model.boundOutputs[0],
                                                       reinterpret_cast<void**>(&data)),
                            "inpaint output data"))
                {
                    *filled = data;
                    ok = true;
                }
            }
        }
    }

    if (eyeValue) _api->ReleaseValue(eyeValue);
    if (maskValue) _api->ReleaseValue(maskValue);
    return ok;
}

bool OrtRuntime::runMlbwDevice(size_t index,
                               const float* x, const int64_t* xShape,
                               const float** delta, const float** layerWeight,
                               const float** maskLogits)
{
    if (!deviceCapable() || index >= _models.size() || !_models[index].binding)
    {
        return false;
    }
    Model& model = _models[index];

    // The previous eye's outputs are only valid until here. This graph runs
    // twice a frame, so that is not a per-frame subtlety but a per-eye one: the
    // left eye's heads have to be consumed before the right eye's call.
    releaseBoundOutputs(model);
    if (!rebindOutputs(model))
    {
        return false;
    }

    const size_t elements = size_t(xShape[0] * xShape[1] * xShape[2] * xShape[3]);
    // fp32, unlike the inpaint graphs, and measured rather than assumed: these
    // deltas become grid coordinates, and at 1920 wide fp16's resolution near
    // the grid's edges is coarser than the spacing between adjacent columns.
    const ONNXTensorElementDataType type = model.inputType;
    const size_t elementSize =
        (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? sizeof(uint16_t) : sizeof(float);

    OrtValue* xValue = nullptr;
    bool ok = false;

    if (!failed(_api->CreateTensorWithDataAsOrtValue(
            _cudaMemoryInfo, const_cast<float*>(x), elements * elementSize,
            xShape, 4, type, &xValue), "mlbw x tensor"))
    {
        // By position, as everywhere else here: the graph declares one input and
        // binding by literal name would break on a re-export that renamed it.
        if (model.inputNames.size() == 1 &&
            !failed(_api->BindInput(model.binding, model.inputNames[0].c_str(), xValue),
                    "BindInput(x)"))
        {
            const auto started = std::chrono::steady_clock::now();
            OrtStatus* status = _api->RunWithBinding(model.session, nullptr, model.binding);
            _lastRunMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();

            if (!failed(status, "RunWithBinding(mlbw)") &&
                !failed(_api->GetBoundOutputValues(model.binding, _allocator,
                                                   &model.boundOutputs, &model.boundOutputCount),
                        "GetBoundOutputValues") &&
                model.boundOutputCount == 3)
            {
                const float** targets[3] = {delta, layerWeight, maskLogits};
                ok = true;
                for (int i = 0; i < 3 && ok; ++i)
                {
                    float* data = nullptr;
                    if (failed(_api->GetTensorMutableData(model.boundOutputs[i],
                                                          reinterpret_cast<void**>(&data)),
                               "mlbw output data"))
                    {
                        ok = false;
                        break;
                    }
                    *targets[i] = data;
                }
            }
        }
    }

    if (xValue) _api->ReleaseValue(xValue);
    return ok;
}

bool OrtRuntime::runDevice(size_t index,
                           const float* image, const int64_t* imageShape,
                           const float* x, const int64_t* xShape,
                           float deltaScale,
                           const float** left, const float** right)
{
    if (!deviceCapable() || index >= _models.size() || !_models[index].binding)
    {
        return false;
    }
    Model& model = _models[index];

    // The previous frame's outputs are only valid until here.
    releaseBoundOutputs(model);
    if (!rebindOutputs(model))
    {
        return false;
    }

    const size_t imageElements = size_t(imageShape[0] * imageShape[1] * imageShape[2] * imageShape[3]);
    const size_t xElements = size_t(xShape[0] * xShape[1] * xShape[2] * xShape[3]);

    OrtValue* imageValue = nullptr;
    OrtValue* xValue = nullptr;
    OrtValue* scaleValue = nullptr;
    bool ok = false;

    if (!failed(_api->CreateTensorWithDataAsOrtValue(
            _cudaMemoryInfo, const_cast<float*>(image), imageElements * sizeof(float),
            imageShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &imageValue), "device image tensor") &&
        !failed(_api->CreateTensorWithDataAsOrtValue(
            _cudaMemoryInfo, const_cast<float*>(x), xElements * sizeof(float),
            xShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xValue), "device x tensor") &&
        // delta_scale stays on the host: four bytes, and ORT wants it where the
        // shape arithmetic happens anyway.
        !failed(_api->CreateTensorWithDataAsOrtValue(
            _memoryInfo, &deltaScale, sizeof(float),
            nullptr, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &scaleValue), "delta_scale tensor"))
    {
        if (!failed(_api->BindInput(model.binding, "image", imageValue), "BindInput(image)") &&
            !failed(_api->BindInput(model.binding, "x", xValue), "BindInput(x)") &&
            !failed(_api->BindInput(model.binding, "delta_scale", scaleValue), "BindInput(delta_scale)"))
        {
            const auto started = std::chrono::steady_clock::now();
            OrtStatus* status = _api->RunWithBinding(model.session, nullptr, model.binding);
            _lastRunMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();

            if (!failed(status, "RunWithBinding") &&
                !failed(_api->GetBoundOutputValues(model.binding, _allocator,
                                                   &model.boundOutputs, &model.boundOutputCount),
                        "GetBoundOutputValues") &&
                model.boundOutputCount == 2)
            {
                float* leftData = nullptr;
                float* rightData = nullptr;
                if (!failed(_api->GetTensorMutableData(model.boundOutputs[0],
                                                       reinterpret_cast<void**>(&leftData)),
                            "left device data") &&
                    !failed(_api->GetTensorMutableData(model.boundOutputs[1],
                                                       reinterpret_cast<void**>(&rightData)),
                            "right device data"))
                {
                    *left = leftData;
                    *right = rightData;
                    ok = true;
                }
            }
        }
    }

    // Binding an input takes its own reference, so dropping ours here is right.
    for (OrtValue* value : {imageValue, xValue, scaleValue})
    {
        if (value) _api->ReleaseValue(value);
    }
    _api->ClearBoundInputs(model.binding);
    return ok;
}

}  // namespace iw3
