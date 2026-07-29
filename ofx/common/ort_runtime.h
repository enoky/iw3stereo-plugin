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

    // Loads onnxruntime.dll from `directory`, creates one environment, and
    // opens each of `modelPaths` as its own session with the CUDA provider
    // (TF32 off). Falls back to CPU if CUDA will not come up, and says so in
    // `report`. Returns false only if even that fails.
    //
    // Several models share one environment and one loaded library on purpose:
    // ORT expects a single OrtEnv per process, and the sessions are small next
    // to the provider's own start-up cost.
    // `conserveMemory` marks, per model, the graphs whose sessions should be
    // built to use as little VRAM as they can. It is not a tuning knob: with it
    // off, the twelve-frame inpaint graph takes 15.7 GiB and sixteen seconds a
    // window at HD, and with it on, 9.0 GiB and a quarter of a second. Pass an
    // empty vector to leave every session on ORT's defaults.
    bool open(const std::wstring& directory, const std::vector<std::wstring>& modelPaths,
              bool preferCuda, const std::vector<bool>& conserveMemory = {});

    // Runs one of the graphs, by index into the modelPaths given to open().
    // Shapes are NCHW; `left` and `right` need room for imageCount floats each.
    bool run(size_t model,
             const float* image, const int64_t* imageShape,
             const float* x, const int64_t* xShape,
             float deltaScale,
             float* left, float* right, size_t imageCount);

    // The same run with every buffer already on the GPU. `left` and `right`
    // come back as device pointers owned by this object, valid until the next
    // call. Only usable when deviceCapable().
    //
    // This is what keeps a frame off the PCIe bus: through the host run()
    // above, 1080p costs 25 MB up and 50 MB back, every frame.
    bool runDevice(size_t model,
                   const float* image, const int64_t* imageShape,
                   const float* x, const int64_t* xShape,
                   float deltaScale,
                   const float** left, const float** right);

    // The inpaint graph, which has a different signature: one eye and its hole
    // mask in, the filled eye out, all NCHW and all on the device. `shape` is
    // the eye's; the mask is the same but with one channel.
    //
    // `filled` comes back as a device pointer owned by this object, valid until
    // the next call on this model.
    bool runInpaintDevice(size_t model,
                          const float* eye, const float* mask, const int64_t* shape,
                          const float** filled);

    // mask_mlbw_l2, which has a third signature again: the model's input tensor
    // in, and three things out -- a sampling delta per layer, a softmax weight
    // per layer, and the hole mask as *logits*.
    //
    // The warp is not in this graph and that is deliberate; see
    // ofx/plugin/mlbw_gpu.h. So unlike the row_flow graphs this returns the
    // network's raw heads rather than a pair of eyes, and the geometry happens
    // in MlbwGpu afterwards.
    //
    // All three come back as device pointers owned by this object, valid until
    // the next call on this model -- which matters here more than elsewhere,
    // because this graph runs twice per frame, once per eye.
    bool runMlbwDevice(size_t model,
                       const float* x, const int64_t* xShape,
                       const float** delta, const float** layerWeight,
                       const float** maskLogits);

    // How many outputs a graph has, which is what distinguishes them: the warp
    // returns left and right, the inpaint one filled eye.
    size_t outputCount(size_t model) const;

    // True when the graph wants half-precision input, which the inpaint graphs
    // do. Everything the plugin computes is fp32, so this decides whether a
    // cast has to happen either side of the call.
    bool inputIsHalf(size_t model) const;

    bool deviceCapable() const { return ready() && _provider == "CUDA" && _cudaMemoryInfo; }

    bool ready() const { return !_models.empty() && _models[0].session != nullptr; }
    size_t modelCount() const { return _models.size(); }
    const std::string& provider() const { return _provider; }
    const std::string& version() const { return _version; }
    const std::vector<std::string>& report() const { return _report; }

    // Report lines added since the last call to this.
    //
    // Without it a render-time failure says nothing: open() dumps the report
    // once at bring-up and nothing ever reads it again, so every ORT error
    // message after that went into the vector and stayed there. That is how a
    // "GPU inpaint failed" with no cause attached happens.
    std::vector<std::string> takeNewReport();
    double lastRunMilliseconds() const { return _lastRunMs; }

private:
    struct Model
    {
        OrtSession* session = nullptr;
        OrtIoBinding* binding = nullptr;
        OrtValue** boundOutputs = nullptr;
        size_t boundOutputCount = 0;
        // Read from the graph rather than assumed, so binding does not have to
        // know which kind of graph it is looking at. The two inpaint graphs
        // name their inputs differently -- eye/mask for one frame, eyes/masks
        // for a twelve-frame window -- and binding by position rather than by
        // literal name is what lets one code path drive both.
        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;
        // Read from the graph too. The inpaint graphs are exported in half
        // precision -- in fp32 the twelve-frame window does not fit in 17 GiB
        // -- and the caller has to know which kind of buffer to hand over.
        ONNXTensorElementDataType inputType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };

    void note(const std::string& line) { _report.push_back(line); }
    bool failed(OrtStatus* status, const char* what);
    void releaseBoundOutputs(Model& model);
    bool bindOutputs(Model& model);
    bool rebindOutputs(Model& model);
    bool readOutputNames(Model& model);
    bool readInputNames(Model& model);

    // The inpaint run, with the memory the inputs live in left open: the render
    // path passes device memory, the warm-up passes host memory and lets ORT
    // copy, because the copy is the point of doing it early.
    bool runInpaintWith(size_t index, OrtMemoryInfo* memoryInfo,
                        const float* eye, const float* mask, const int64_t* shape,
                        const float** filled);

    // One throwaway inference per model, so the first real frame does not pay
    // for loading cuDNN's kernels and choosing algorithms.
    void warmUp(size_t model);

    HMODULE _library = nullptr;
    const OrtApi* _api = nullptr;
    OrtEnv* _env = nullptr;
    std::vector<Model> _models;
    OrtMemoryInfo* _memoryInfo = nullptr;
    OrtMemoryInfo* _cudaMemoryInfo = nullptr;
    OrtAllocator* _allocator = nullptr;
    std::string _provider = "none";
    std::string _version;
    std::vector<std::string> _report;
    size_t _reportCursor = 0;
    double _lastRunMs = 0.0;
};

}  // namespace iw3
