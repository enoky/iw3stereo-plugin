// Dynamically-loaded ONNX Runtime, deliberately never linked against.
//
// Resolve ships its own onnxruntime.dll -- version 1.13 from October 2022,
// CPU-only, exporting nothing but OrtGetApiBase and
// OrtSessionOptionsAppendExecutionProvider_CPU -- and it lives in the
// application directory, which Windows searches before almost anywhere else
// when resolving a DLL by name. A plugin with onnxruntime.dll in its import
// table would therefore bind to *Resolve's* copy: no CUDA provider, and an API
// surface fifteen versions out of date.
//
// So there is no import library here and no link-time dependency. The runtime
// is loaded from an absolute path inside our own bundle and reached through
// OrtGetApiBase, which is the entry point the C API is designed to be used
// through. LOAD_WITH_ALTERED_SEARCH_PATH makes its sibling provider DLLs
// resolve out of the same folder rather than out of Resolve's.

#pragma once

#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "onnxruntime_c_api.h"

namespace iw3
{

class OrtRuntime
{
public:
    OrtRuntime() = default;
    ~OrtRuntime();

    OrtRuntime(const OrtRuntime&) = delete;
    OrtRuntime& operator=(const OrtRuntime&) = delete;

    // Loads onnxruntime.dll from `directory`, creates an environment, and opens
    // `modelPath` with the CUDA provider (TF32 off). Falls back to CPU if CUDA
    // will not come up, and says so in `report`. Returns false only if even
    // that fails.
    bool open(const std::wstring& directory, const std::wstring& modelPath, bool preferCuda);

    // Runs the graph. Shapes are NCHW. `left` and `right` must have room for
    // imageCount floats each.
    bool run(const float* image, const int64_t* imageShape,
             const float* x, const int64_t* xShape,
             float deltaScale,
             float* left, float* right, size_t imageCount);

    bool ready() const { return _session != nullptr; }
    const std::string& provider() const { return _provider; }
    const std::string& version() const { return _version; }
    const std::vector<std::string>& report() const { return _report; }
    double lastRunMilliseconds() const { return _lastRunMs; }

private:
    void note(const std::string& line) { _report.push_back(line); }
    bool failed(OrtStatus* status, const char* what);

    HMODULE _library = nullptr;
    const OrtApi* _api = nullptr;
    OrtEnv* _env = nullptr;
    OrtSession* _session = nullptr;
    OrtMemoryInfo* _memoryInfo = nullptr;
    std::string _provider = "none";
    std::string _version;
    std::vector<std::string> _report;
    double _lastRunMs = 0.0;
};

}  // namespace iw3
