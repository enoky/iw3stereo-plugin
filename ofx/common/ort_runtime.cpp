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
                      bool preferCuda)
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

bool OrtRuntime::bindOutputs(Model& model)
{
    if (failed(_api->CreateIoBinding(model.session, &model.binding), "CreateIoBinding"))
    {
        model.binding = nullptr;
        return false;
    }
    // Outputs land in device memory rather than being copied back.
    if (failed(_api->BindOutputToDevice(model.binding, "left", _cudaMemoryInfo), "BindOutput(left)") ||
        failed(_api->BindOutputToDevice(model.binding, "right", _cudaMemoryInfo), "BindOutput(right)"))
    {
        _api->ReleaseIoBinding(model.binding);
        model.binding = nullptr;
        return false;
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

    std::vector<float> image(kElements, 0.5f);
    std::vector<float> x(kElements, 0.5f);
    std::vector<float> left(kElements);
    std::vector<float> right(kElements);

    const auto started = std::chrono::steady_clock::now();
    const bool ok = run(index, image.data(), shape, x.data(), shape,
                        1.0f / float(kSize / 2 - 1),
                        left.data(), right.data(), kElements);
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
