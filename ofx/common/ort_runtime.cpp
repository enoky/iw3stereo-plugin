#include "ort_runtime.h"

#include <chrono>
#include <sstream>

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
        if (_memoryInfo) _api->ReleaseMemoryInfo(_memoryInfo);
        if (_session) _api->ReleaseSession(_session);
        if (_env) _api->ReleaseEnv(_env);
    }
    // The library is deliberately not freed: Resolve keeps plugins loaded for
    // the session, and unloading a runtime that has a live CUDA context while
    // the host is still using the GPU is a good way to crash on exit.
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

bool OrtRuntime::open(const std::wstring& directory, const std::wstring& modelPath, bool preferCuda)
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

    OrtSessionOptions* options = nullptr;
    if (failed(_api->CreateSessionOptions(&options), "CreateSessionOptions"))
    {
        return false;
    }

    bool cudaAttached = false;
    if (preferCuda)
    {
        // The whole point of this probe: does a CUDA execution provider come up
        // inside a process that already has Resolve's own CUDA context?
        const auto started = std::chrono::steady_clock::now();
        OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
        if (!failed(_api->CreateCUDAProviderOptions(&cudaOptions), "CreateCUDAProviderOptions"))
        {
            // TF32 is on by default and costs three orders of magnitude of
            // accuracy on this model for well under a millisecond.
            const char* keys[] = {"device_id", "use_tf32"};
            const char* values[] = {"0", "0"};
            failed(_api->UpdateCUDAProviderOptions(cudaOptions, keys, values, 2),
                   "UpdateCUDAProviderOptions");
            if (!failed(_api->SessionOptionsAppendExecutionProvider_CUDA_V2(options, cudaOptions),
                        "SessionOptionsAppendExecutionProvider_CUDA_V2"))
            {
                cudaAttached = true;
            }
            _api->ReleaseCUDAProviderOptions(cudaOptions);
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        note(std::string("CUDA provider attach: ") + (cudaAttached ? "OK" : "FAILED") +
             " in " + std::to_string(ms) + " ms");
    }

    note("opening " + narrow(modelPath));
    const auto sessionStart = std::chrono::steady_clock::now();
    OrtStatus* status = _api->CreateSession(_env, modelPath.c_str(), options, &_session);
    if (status)
    {
        failed(status, "CreateSession");
        if (cudaAttached)
        {
            // Retry on CPU so the rest of the probe still reports something,
            // and so the log distinguishes "CUDA is broken here" from "the
            // model or the path is wrong".
            note("retrying without CUDA");
            _api->ReleaseSessionOptions(options);
            options = nullptr;
            if (failed(_api->CreateSessionOptions(&options), "CreateSessionOptions"))
            {
                return false;
            }
            if (failed(_api->CreateSession(_env, modelPath.c_str(), options, &_session), "CreateSession(CPU)"))
            {
                _api->ReleaseSessionOptions(options);
                return false;
            }
            cudaAttached = false;
        }
        else
        {
            _api->ReleaseSessionOptions(options);
            return false;
        }
    }
    const double sessionMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - sessionStart).count();
    note("CreateSession OK in " + std::to_string(sessionMs) + " ms");

    _api->ReleaseSessionOptions(options);
    _provider = cudaAttached ? "CUDA" : "CPU";
    note("provider in use: " + _provider);

    if (failed(_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &_memoryInfo),
               "CreateCpuMemoryInfo"))
    {
        return false;
    }
    return true;
}

bool OrtRuntime::run(const float* image, const int64_t* imageShape,
                     const float* x, const int64_t* xShape,
                     float deltaScale,
                     float* left, float* right, size_t imageCount)
{
    if (!_session)
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
        OrtStatus* status = _api->Run(_session, nullptr, inputNames, inputs, 3, outputNames, 2, outputs);
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

}  // namespace iw3
