// iw3 Stereo -- the plugin.
//
// A Fusion node with two image inputs, Source and Depth, producing one image:
// anaglyph, either eye, or half side-by-side. It is iw3's warping half, driven
// by depth that arrives as a clip rather than being estimated.
//
// Interface is specified in docs/phase3-interface.md, which was agreed before
// this was written. Findings behind the non-obvious decisions are in
// docs/phase0-findings.md and docs/phase2-onnx.md.

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "ort_runtime.h"
#include "probe_log.h"
#include "stereo_gpu.h"
#include "stereo_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define kPluginName "iw3 Stereo"
#define kPluginGrouping "iw3"
#define kPluginIdentifier "com.nunif.iw3.stereo"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kDepthClipName "Depth"

namespace
{

std::wstring bundleDirectory()
{
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&bundleDirectory), &self);
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(self, path, MAX_PATH);
    const std::wstring full(path);
    const size_t slash = full.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : full.substr(0, slash);
}

int componentCount(OFX::PixelComponentEnum components)
{
    switch (components)
    {
        case OFX::ePixelComponentRGBA: return 4;
        case OFX::ePixelComponentRGB: return 3;
        case OFX::ePixelComponentAlpha: return 1;
        default: return 0;
    }
}

// Row addressing done once per row rather than per pixel. The probe used
// getPixelAddress() per pixel and that cost more than the inference did.
inline float* rowPointer(OFX::Image* image, int y)
{
    const OfxRectI& bounds = image->getBounds();
    if (y < bounds.y1 || y >= bounds.y2)
    {
        return nullptr;
    }
    char* base = static_cast<char*>(image->getPixelData());
    return reinterpret_cast<float*>(base + size_t(y - bounds.y1) * size_t(image->getRowBytes()));
}

}  // namespace

// Defined below, next to the runtime singleton it starts.
static void startRuntimeBringUp();

////////////////////////////////////////////////////////////////////////////////

class Iw3StereoEffect : public OFX::ImageEffect
{
public:
    explicit Iw3StereoEffect(OfxImageEffectHandle handle)
        : OFX::ImageEffect(handle)
    {
        _dstClip = fetchClip(kOfxImageEffectOutputClipName);
        _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
        try
        {
            _depthClip = fetchClip(kDepthClipName);
        }
        catch (...)
        {
            _depthClip = nullptr;  // filter context has no second input
        }

        _divergence = fetchDoubleParam("divergence");
        _convergence = fetchDoubleParam("convergence");
        _preserveScreenBorder = fetchBooleanParam("preserveScreenBorder");
        _depthInverted = fetchBooleanParam("depthInverted");
        _foregroundScale = fetchDoubleParam("foregroundScale");
        _mapperType = fetchChoiceParam("mapperType");
        _stereoWidth = fetchIntParam("stereoWidth");
        _depthRange = fetchChoiceParam("depthRange");
        _output = fetchChoiceParam("output");
        _modelChoice = fetchChoiceParam("model");

        startRuntimeBringUp();
    }

    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments&, OFX::Clip*&, double&) override { return false; }

    // Full SBS is twice as wide as its source, so the effect has to ask for a
    // bigger output than it was given. Resolve reports supportsMultiResolution
    // = 0, which in OFX terms means it need not honour this at all; whether it
    // does is measured rather than assumed, and render() copes either way.
    virtual bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args,
                                       OfxRectD& rod) override;

private:
    iw3::Settings readSettings(double time) const;

    // What to render and at what size. These differ only for Full SBS, where
    // the warp runs at the source's resolution and the output is twice as wide.
    struct Layout
    {
        iw3::OutputMode output = iw3::OutputMode::Anaglyph;
        int width = 0;   // the size the warp runs at
        int height = 0;
    };
    Layout resolveLayout(const iw3::Settings& settings, const OfxRectI& window, OFX::Image* src);
    // Passes the source through, on whichever side of the PCIe bus the buffers
    // actually live. Every giving-up path must go through this: with CUDA
    // render enabled getPixelData() is device memory, and reading it on the CPU
    // is an access violation that takes Resolve down.
    void passSourceThrough(const OFX::RenderArguments& args, OFX::Image* dst, OFX::Image* src,
                           const OfxRectI& window, int components);
    void passthroughHost(OFX::Image* dst, OFX::Image* src, const OfxRectI& window, int components);
    void report(const std::string& message, bool warning);

    OFX::Clip* _dstClip = nullptr;
    OFX::Clip* _srcClip = nullptr;
    OFX::Clip* _depthClip = nullptr;

    OFX::DoubleParam* _divergence = nullptr;
    OFX::DoubleParam* _convergence = nullptr;
    OFX::BooleanParam* _preserveScreenBorder = nullptr;
    OFX::BooleanParam* _depthInverted = nullptr;
    OFX::DoubleParam* _foregroundScale = nullptr;
    OFX::ChoiceParam* _mapperType = nullptr;
    OFX::IntParam* _stereoWidth = nullptr;
    OFX::ChoiceParam* _depthRange = nullptr;
    OFX::ChoiceParam* _output = nullptr;
    OFX::ChoiceParam* _modelChoice = nullptr;

    // Reused across frames; resized only when the frame size changes.
    std::vector<float> _image, _depthFull, _depthSmall, _x, _left, _right, _composed;
    iw3::DepthResizer _resizer;
    iw3::GpuPipeline _gpu;
    std::string _lastMessage;
    unsigned _frameCounter = 0;

    // Returns false if the GPU path could not be taken, so render() can fall
    // through to the CPU one.
    bool renderCuda(const OFX::RenderArguments& args, const iw3::Settings& settings,
                    OFX::Image* dst, OFX::Image* src, OFX::Image* depth,
                    int components, const OfxRectI& window);
};

// One session for the process. Bring-up costs 60-240 ms for the CUDA provider
// plus 40-55 ms for the session; per instance would be unusable.
//
// Deliberately leaked, and that is load-bearing rather than laziness. As a
// plain function-local static it is destroyed during static destruction at
// process exit, which releases an ORT session owning a CUDA context while the
// driver is already shutting down. That deadlocks: Resolve's window closes but
// the process stays in Task Manager until it is killed. Never destroying it
// leaves the OS to reclaim everything at exit, which it does correctly.
static iw3::OrtRuntime& sharedRuntime()
{
    static iw3::OrtRuntime* instance = []()
    {
        auto* runtime = new iw3::OrtRuntime();
        const std::wstring directory = bundleDirectory();
        // Index order matters: it is what the Model parameter selects.
        const std::vector<std::wstring> graphs = {
            directory + L"\\stereo_warp.onnx",     // 0: row_flow_v2
            directory + L"\\stereo_warp_v3.onnx",  // 1: row_flow_v3
        };
        const bool ok = runtime->open(directory + L"\\ort", graphs, true);
        probe::logf("---- iw3 Stereo: ONNX Runtime bring-up (%s) ----", ok ? "OK" : "FAILED");
        for (const std::string& line : runtime->report())
        {
            probe::logf("    %s", line.c_str());
        }
        return runtime;
    }();
    return *instance;
}

// Bring the runtime up as soon as a node exists, rather than on the first
// render. Provider start-up, session creation and the warm-up run together
// cost a few hundred milliseconds; spending them while the user is still
// wiring the node up means the first visible frame does not.
//
// No extra synchronisation is needed: sharedRuntime()'s function-local static
// already blocks any later caller until the initializer finishes, so a render
// that arrives early simply waits.
//
// The thread is leaked rather than joined. A std::thread destructor on a
// joinable thread calls terminate(), and joining at process exit is the same
// shutdown deadlock this plugin avoids with the runtime itself.
static void startRuntimeBringUp()
{
    static std::thread* thread = new std::thread([]() { sharedRuntime(); });
    (void)thread;
}

iw3::Settings Iw3StereoEffect::readSettings(double time) const
{
    iw3::Settings settings;
    _divergence->getValueAtTime(time, settings.divergence);
    _convergence->getValueAtTime(time, settings.convergence);
    _foregroundScale->getValueAtTime(time, settings.foregroundScale);
    _preserveScreenBorder->getValueAtTime(time, settings.preserveScreenBorder);
    _depthInverted->getValueAtTime(time, settings.depthInverted);
    _stereoWidth->getValueAtTime(time, settings.stereoWidth);

    int choice = 0;
    _mapperType->getValueAtTime(time, choice);
    settings.mapperType = choice == 1 ? iw3::MapperType::Shift : iw3::MapperType::Multiply;
    _depthRange->getValueAtTime(time, choice);
    settings.undoVideoRange = (choice == 1);
    _modelChoice->getValueAtTime(time, choice);
    settings.model = size_t(choice == 1 ? 1 : 0);
    _output->getValueAtTime(time, choice);
    // The option indices and the enum are kept in the same order deliberately:
    // getRegionOfDefinition() runs before any Settings exist and compares the
    // raw choice, so it does not have to duplicate this mapping.
    settings.output = choice == 1 ? iw3::OutputMode::LeftEye
                    : choice == 2 ? iw3::OutputMode::RightEye
                    : choice == 3 ? iw3::OutputMode::HalfSbs
                    : choice == 4 ? iw3::OutputMode::DepthDebug
                    : choice == 5 ? iw3::OutputMode::FullSbs
                                  : iw3::OutputMode::Anaglyph;
    return settings;
}

Iw3StereoEffect::Layout Iw3StereoEffect::resolveLayout(const iw3::Settings& settings,
                                                       const OfxRectI& window, OFX::Image* src)
{
    Layout layout;
    layout.output = settings.output;
    layout.width = window.x2 - window.x1;
    layout.height = window.y2 - window.y1;

    if (settings.output != iw3::OutputMode::FullSbs)
    {
        return layout;
    }

    // Did the host actually grant the doubled region asked for in
    // getRegionOfDefinition()? Resolve says supportsMultiResolution = 0, so it
    // is entitled not to, and the answer decides what can be rendered.
    const int sourceWidth = src ? (src->getBounds().x2 - src->getBounds().x1) : 0;
    if (sourceWidth > 0 && layout.width >= sourceWidth * 2 - 2)
    {
        layout.width /= 2;
        return layout;
    }

    report("Resolve's OFX host does not allow an effect to change its output size "
           "(supportsMultiResolution = 0), so Full SBS cannot work here. Rendering Half SBS. "
           "The bundle's README has a Fusion node setup that does give full side by side.", true);
    layout.output = iw3::OutputMode::HalfSbs;
    return layout;
}

bool Iw3StereoEffect::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args,
                                            OfxRectD& rod)
{
    int choice = 0;
    _output->getValueAtTime(args.time, choice);
    if (choice != int(iw3::OutputMode::FullSbs) || !_srcClip || !_srcClip->isConnected())
    {
        return false;  // same size as the source, which is the default anyway
    }

    rod = _srcClip->getRegionOfDefinition(args.time);
    rod.x2 = rod.x1 + (rod.x2 - rod.x1) * 2.0;
    return true;
}

void Iw3StereoEffect::report(const std::string& message, bool warning)
{
    if (message == _lastMessage)
    {
        return;
    }
    _lastMessage = message;
    probe::logf("iw3 Stereo: %s", message.c_str());
    try
    {
        if (message.empty())
        {
            clearPersistentMessage();
        }
        else
        {
            setPersistentMessage(warning ? OFX::Message::eMessageWarning : OFX::Message::eMessageMessage,
                                 "", message);
        }
    }
    catch (...)
    {
        // Some hosts refuse messages from a render thread. The log still has it.
    }
}

void Iw3StereoEffect::passSourceThrough(const OFX::RenderArguments& args, OFX::Image* dst,
                                        OFX::Image* src, const OfxRectI& window, int components)
{
    if (!dst)
    {
        return;
    }
    if (args.isEnabledCudaRender)
    {
        const OfxRectI& dstBounds = dst->getBounds();
        const size_t dstPitch = size_t(dst->getRowBytes()) / sizeof(float);
        float* destination = static_cast<float*>(dst->getPixelData()) +
            size_t(window.y1 - dstBounds.y1) * dstPitch +
            size_t(window.x1 - dstBounds.x1) * size_t(components);

        const float* source = nullptr;
        size_t sourcePitch = 0;
        int offsetX = 0, offsetY = 0, sourceWidth = 0, sourceHeight = 0;
        if (src)
        {
            const OfxRectI& srcBounds = src->getBounds();
            source = static_cast<const float*>(src->getPixelData());
            sourcePitch = size_t(src->getRowBytes()) / sizeof(float);
            offsetX = window.x1 - srcBounds.x1;
            offsetY = window.y1 - srcBounds.y1;
            sourceWidth = srcBounds.x2 - srcBounds.x1;
            sourceHeight = srcBounds.y2 - srcBounds.y1;
        }

        iw3::devicePassthrough(source, sourcePitch, offsetX, offsetY, sourceWidth, sourceHeight,
                               destination, dstPitch,
                               window.x2 - window.x1, window.y2 - window.y1, components,
                               args.pCudaStream);
        return;
    }
    passthroughHost(dst, src, window, components);
}

void Iw3StereoEffect::passthroughHost(OFX::Image* dst, OFX::Image* src, const OfxRectI& window, int components)
{
    for (int y = window.y1; y < window.y2; ++y)
    {
        float* out = rowPointer(dst, y);
        const float* in = src ? rowPointer(src, y) : nullptr;
        if (!out)
        {
            continue;
        }
        for (int x = window.x1; x < window.x2; ++x)
        {
            float* pixel = out + size_t(x - dst->getBounds().x1) * size_t(components);
            if (in)
            {
                const float* source = in + size_t(x - src->getBounds().x1) * size_t(components);
                for (int c = 0; c < components; ++c) pixel[c] = source[c];
            }
            else
            {
                for (int c = 0; c < components; ++c) pixel[c] = 0.0f;
            }
        }
    }
}

// The GPU path. Everything stays in device memory from Resolve's source buffer
// through to Resolve's destination buffer -- at 1080p the CPU path was moving
// 25 MB up and 50 MB back across PCIe every frame, which cost more than the
// inference did.
//
// Returns false without touching the output if the GPU path is unavailable, so
// the caller can fall through to the CPU implementation.
bool Iw3StereoEffect::renderCuda(const OFX::RenderArguments& args, const iw3::Settings& settings,
                                 OFX::Image* dst, OFX::Image* src, OFX::Image* depth,
                                 int components, const OfxRectI& window)
{
    if (!args.isEnabledCudaRender || !iw3::GpuPipeline::deviceAvailable())
    {
        return false;
    }
    iw3::OrtRuntime& ort = sharedRuntime();
    if (!ort.deviceCapable())
    {
        return false;
    }
    const auto started = std::chrono::steady_clock::now();

    const Layout layout = resolveLayout(settings, window, src);
    const int width = layout.width;
    const int height = layout.height;

    const std::pair<int, int> target = iw3::depthTargetSize(width, height, settings.stereoWidth);
    const int depthWidth = target.first;
    const int depthHeight = target.second;
    if (depthWidth < 4)
    {
        return false;
    }

    if (!_gpu.prepare(width, height, depthWidth, depthHeight))
    {
        report("GPU setup failed: " + _gpu.error() + " -- falling back to the CPU.", true);
        return false;
    }

    // getRowBytes() is a byte stride; the kernels index in floats.
    const size_t sourcePitch = size_t(src->getRowBytes()) / sizeof(float);
    const size_t depthPitch = size_t(depth->getRowBytes()) / sizeof(float);
    const size_t destinationPitch = size_t(dst->getRowBytes()) / sizeof(float);

    void* stream = args.pCudaStream;
    const iw3::Mapper mapper(settings.foregroundScale, settings.mapperType);

    // Every image gets its own bounds handed to the kernel. They are not
    // guaranteed to start at the render window's origin or to cover it, and
    // assuming otherwise is an out-of-bounds device read.
    const OfxRectI& srcBounds = src->getBounds();
    const OfxRectI& depthBounds = depth->getBounds();
    const OfxRectI& dstBounds = dst->getBounds();

    _gpu.packSource(static_cast<const float*>(src->getPixelData()), sourcePitch, components,
                    window.x1 - srcBounds.x1, window.y1 - srcBounds.y1,
                    srcBounds.x2 - srcBounds.x1, srcBounds.y2 - srcBounds.y1, stream);
    _gpu.packDepth(static_cast<const float*>(depth->getPixelData()), depthPitch,
                   componentCount(depth->getPixelComponents()),
                   window.x1 - depthBounds.x1, window.y1 - depthBounds.y1,
                   depthBounds.x2 - depthBounds.x1, depthBounds.y2 - depthBounds.y1,
                   settings.undoVideoRange, settings.depthInverted, mapper.params(), stream);
    _gpu.resizeDepth(stream);
    _gpu.buildInputTensor(settings.divergence, settings.convergence,
                          settings.preserveScreenBorder, stream);

    // ORT runs on its own stream, so Resolve's work has to have landed first.
    // Sharing the stream through user_compute_stream would remove this
    // synchronise; it is not done because the session is built once and the
    // host's stream is only known per render call.
    if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
    {
        report("CUDA stream sync failed -- falling back to the CPU.", true);
        return false;
    }

    const int64_t imageShape[4] = {1, 3, height, width};
    const int64_t xShape[4] = {1, 3, depthHeight, depthWidth};
    const float* left = nullptr;
    const float* right = nullptr;
    const size_t model = std::min(settings.model, ort.modelCount() - 1);
    if (!ort.runDevice(model, _gpu.imageDevice(), imageShape, _gpu.inputTensorDevice(), xShape,
                       iw3::deltaScale(depthWidth), &left, &right))
    {
        report("GPU inference failed; see the log. Falling back to the CPU.", true);
        return false;
    }

    float* destination = static_cast<float*>(dst->getPixelData()) +
        size_t(window.y1 - dstBounds.y1) * destinationPitch +
        size_t(window.x1 - dstBounds.x1) * size_t(components);

    if (layout.output == iw3::OutputMode::FullSbs)
    {
        // Nothing to compose: the two eyes go straight into their own halves.
        _gpu.unpackFullSbs(left, right, destination, destinationPitch, components,
                           window.x2 - window.x1, stream);
    }
    else
    {
        _gpu.compose(layout.output, left, right, stream);
        _gpu.unpack(destination, destinationPitch, components, stream);
    }

    if (!_gpu.ok())
    {
        report("GPU kernel failed: " + _gpu.error(), true);
        return false;
    }

    // Logged every 60th frame so playback does not swamp the file.
    //
    // The total is honest about most of the work but not all of it: ORT's Run
    // synchronises, so everything up to and including inference is inside the
    // window, while compose and unpack are still queued on Resolve's stream
    // when this returns. Forcing a sync to measure them would slow the path
    // being measured.
    if ((_frameCounter++ % 60) == 0)
    {
        const double total = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        probe::logf("iw3 Stereo: CUDA path %dx%d, depth %dx%d, inference %.2f ms, "
                    "total %.2f ms (compose+unpack still queued), model=%zu",
                    width, height, depthWidth, depthHeight, ort.lastRunMilliseconds(), total, model);
    }
    report(std::string(), false);
    return true;
}

void Iw3StereoEffect::render(const OFX::RenderArguments& args)
{
    const iw3::Settings settings = readSettings(args.time);

    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(_srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> depth(
        _depthClip && _depthClip->isConnected() ? _depthClip->fetchImage(args.time) : nullptr);

    if (!dst || !src)
    {
        return;
    }

    const int components = componentCount(dst->getPixelComponents());
    if (components < 3 || dst->getPixelDepth() != OFX::eBitDepthFloat)
    {
        report("Needs a float RGB or RGBA image.", true);
        return;
    }

    // Clamped to what we are actually allowed to write. OFX says the render
    // window lies inside the output image, but the cost of checking is nothing
    // and the cost of being wrong is a GPU fault.
    const OfxRectI& bounds = dst->getBounds();
    OfxRectI window = args.renderWindow;
    window.x1 = std::max(window.x1, bounds.x1);
    window.y1 = std::max(window.y1, bounds.y1);
    window.x2 = std::min(window.x2, bounds.x2);
    window.y2 = std::min(window.y2, bounds.y2);

    const int width = window.x2 - window.x1;
    const int height = window.y2 - window.y1;
    if (width <= 0 || height <= 0)
    {
        return;
    }
    const size_t pixels = size_t(width) * size_t(height);

    if (!depth)
    {
        report("Depth input is not connected -- passing the source through.", false);
        passSourceThrough(args, dst.get(), src.get(), window, components);
        return;
    }

    iw3::OrtRuntime& ort = sharedRuntime();
    if (!ort.ready())
    {
        report("ONNX Runtime failed to start; see " +
               std::string("%LOCALAPPDATA%\\iw3probe\\probe.log"), true);
        passSourceThrough(args, dst.get(), src.get(), window, components);
        return;
    }

    if (renderCuda(args, settings, dst.get(), src.get(), depth.get(), components, window))
    {
        return;
    }

    if (args.isEnabledCudaRender)
    {
        // The GPU path declined and the CPU one below cannot stand in for it:
        // with CUDA render enabled these buffers are device memory, and the
        // loops that follow would read it on the CPU. That was the crash.
        // Whatever went wrong has already been reported by renderCuda().
        passSourceThrough(args, dst.get(), src.get(), window, components);
        return;
    }

    // --- unpack the two inputs into planar float ---------------------------
    const auto cpuStarted = std::chrono::steady_clock::now();

    // Full SBS is deliberately not implemented here. This path only runs when
    // the host renders without CUDA, which at ~250 ms a frame is not a mode
    // anyone delivers from, and a doubled output would mean a second copy of
    // all the sizing logic for no practical gain.
    iw3::Settings effective = settings;
    if (effective.output == iw3::OutputMode::FullSbs)
    {
        report("Full SBS needs the GPU render path; rendering Half SBS on the CPU.", true);
        effective.output = iw3::OutputMode::HalfSbs;
    }

    _image.resize(pixels * 3);
    _depthFull.resize(pixels);

    const int depthComponents = componentCount(depth->getPixelComponents());
    const iw3::Mapper mapper(effective.foregroundScale, effective.mapperType);

    for (int y = 0; y < height; ++y)
    {
        const float* sourceRow = rowPointer(src.get(), window.y1 + y);
        const float* depthRow = rowPointer(depth.get(), window.y1 + y);
        const size_t rowStart = size_t(y) * size_t(width);

        for (int x = 0; x < width; ++x)
        {
            const size_t index = rowStart + size_t(x);
            if (sourceRow)
            {
                const float* pixel = sourceRow + size_t(window.x1 + x - src->getBounds().x1) * size_t(components);
                _image[index] = pixel[0];
                _image[pixels + index] = pixel[1];
                _image[2 * pixels + index] = pixel[2];
            }
            else
            {
                _image[index] = _image[pixels + index] = _image[2 * pixels + index] = 0.0f;
            }

            float value = 0.0f;
            if (depthRow)
            {
                value = depthRow[size_t(window.x1 + x - depth->getBounds().x1) * size_t(depthComponents)];
            }
            if (settings.undoVideoRange)
            {
                // Resolve expanded 16-235 to 0-1 on the way in; put it back.
                value = (value * 219.0f + 16.0f) / 255.0f;
            }
            value = std::clamp(value, 0.0f, 1.0f);
            if (settings.depthInverted)
            {
                value = 1.0f - value;
            }
            _depthFull[index] = mapper.isIdentity() ? value : float(mapper(double(value)));
        }
    }

    // --- resize the depth --------------------------------------------------
    // Not optional: Resolve delivers depth at composition resolution, which is
    // the exact case that makes the warp stripe. See docs/phase0-findings.md.
    const std::pair<int, int> target = iw3::depthTargetSize(width, height, settings.stereoWidth);
    const int depthWidth = target.first;
    const int depthHeight = target.second;

    const float* depthForModel = _depthFull.data();
    if (depthWidth != width || depthHeight != height)
    {
        _resizer.resize(_depthFull.data(), width, height, _depthSmall, depthWidth, depthHeight);
        depthForModel = _depthSmall.data();
    }

    if (depthWidth < 4)
    {
        report("Frame is too small to warp.", true);
        passSourceThrough(args, dst.get(), src.get(), window, components);
        return;
    }

    // --- run ---------------------------------------------------------------
    // Both eyes are always synthesised, each carrying half the parallax. iw3
    // can instead keep one eye pristine and put the full divergence on the
    // other; that is not exposed here.
    iw3::buildInputTensor(depthForModel, depthWidth, depthHeight,
                          settings.divergence, settings.convergence,
                          settings.preserveScreenBorder, _x);

    _left.resize(pixels * 3);
    _right.resize(pixels * 3);

    const int64_t imageShape[4] = {1, 3, height, width};
    const int64_t xShape[4] = {1, 3, depthHeight, depthWidth};

    const size_t model = std::min(settings.model, ort.modelCount() - 1);
    if (!ort.run(model, _image.data(), imageShape, _x.data(), xShape, iw3::deltaScale(depthWidth),
                 _left.data(), _right.data(), pixels * 3))
    {
        report("Inference failed; see the log.", true);
        passSourceThrough(args, dst.get(), src.get(), window, components);
        return;
    }

    const std::string status = ort.provider() == "CUDA"
        ? std::string()
        : std::string("Running on the CPU -- roughly 20x slower than the GPU. See the log.");
    report(status, !status.empty());

    // --- compose -----------------------------------------------------------
    _composed.resize(pixels * 3);
    switch (effective.output)
    {
        case iw3::OutputMode::Anaglyph:
            iw3::duboisAnaglyph(_left.data(), _right.data(), pixels, _composed.data());
            break;

        case iw3::OutputMode::LeftEye:
            std::copy(_left.begin(), _left.end(), _composed.begin());
            break;

        case iw3::OutputMode::RightEye:
            std::copy(_right.begin(), _right.end(), _composed.begin());
            break;

        case iw3::OutputMode::HalfSbs:
        {
            // Each eye squeezed to half width, averaging pixel pairs rather
            // than dropping every other column.
            const int half = width / 2;
            for (int c = 0; c < 3; ++c)
            {
                for (int y = 0; y < height; ++y)
                {
                    const float* leftRow = _left.data() + size_t(c) * pixels + size_t(y) * size_t(width);
                    const float* rightRow = _right.data() + size_t(c) * pixels + size_t(y) * size_t(width);
                    float* out = _composed.data() + size_t(c) * pixels + size_t(y) * size_t(width);
                    for (int x = 0; x < half; ++x)
                    {
                        const int a = std::min(x * 2, width - 1);
                        const int b = std::min(x * 2 + 1, width - 1);
                        out[x] = 0.5f * (leftRow[a] + leftRow[b]);
                        out[half + x] = 0.5f * (rightRow[a] + rightRow[b]);
                    }
                    for (int x = half * 2; x < width; ++x)
                    {
                        out[x] = 0.0f;
                    }
                }
            }
            break;
        }

        case iw3::OutputMode::FullSbs:  // downgraded to HalfSbs above
        case iw3::OutputMode::DepthDebug:
        {
            // The depth as the model actually saw it, nearest-upscaled back to
            // the frame, so the effect of Stereo Width is visible.
            for (int y = 0; y < height; ++y)
            {
                const int sy = std::min(y * depthHeight / std::max(1, height), depthHeight - 1);
                for (int x = 0; x < width; ++x)
                {
                    const int sx = std::min(x * depthWidth / std::max(1, width), depthWidth - 1);
                    const float value = depthForModel[size_t(sy) * size_t(depthWidth) + size_t(sx)];
                    const size_t index = size_t(y) * size_t(width) + size_t(x);
                    _composed[index] = _composed[pixels + index] = _composed[2 * pixels + index] = value;
                }
            }
            break;
        }
    }

    if ((_frameCounter++ % 60) == 0)
    {
        probe::logf("iw3 Stereo: CPU path %dx%d, depth %dx%d, inference %.2f ms, total %.2f ms",
                    width, height, depthWidth, depthHeight, ort.lastRunMilliseconds(),
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - cpuStarted).count());
    }

    // --- pack back ---------------------------------------------------------
    for (int y = 0; y < height; ++y)
    {
        float* out = rowPointer(dst.get(), window.y1 + y);
        if (!out)
        {
            continue;
        }
        const size_t rowStart = size_t(y) * size_t(width);
        for (int x = 0; x < width; ++x)
        {
            float* pixel = out + size_t(window.x1 + x - dst->getBounds().x1) * size_t(components);
            const size_t index = rowStart + size_t(x);
            pixel[0] = _composed[index];
            pixel[1] = _composed[pixels + index];
            pixel[2] = _composed[2 * pixels + index];
            if (components > 3)
            {
                pixel[3] = 1.0f;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

mDeclarePluginFactory(Iw3StereoFactory, {}, {});

void Iw3StereoFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(
        "Stereo 3D from a colour frame and a depth pass, using iw3's row_flow_v2 warp.\n"
        "\n"
        "Connect the depth pass to the Depth input. Leave the depth clip's Data Levels "
        "alone: Resolve's range handling already matches what iw3 does.");

    // General is where the second input clip exists; filter is described only
    // so the effect appears rather than failing oddly on the Edit page.
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);

    // Ask for CUDA render. Resolve then hands render() device pointers and a
    // stream, and the whole frame stays on the GPU. Falls back automatically:
    // the host clears args.isEnabledCudaRender when it renders on the CPU.
    desc.setSupportsCudaRender(true);
    desc.setSupportsCudaStream(true);
}

void Iw3StereoFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setSupportsTiles(false);

    if (context == OFX::eContextGeneral)
    {
        OFX::ClipDescriptor* depth = desc.defineClip(kDepthClipName);
        depth->addSupportedComponent(OFX::ePixelComponentRGBA);
        depth->addSupportedComponent(OFX::ePixelComponentAlpha);
        depth->setSupportsTiles(false);
        depth->setOptional(true);
    }

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

    OFX::ChoiceParamDescriptor* model = desc.defineChoiceParam("model");
    model->setLabels("Model", "Model", "Model");
    model->setHint("row_flow_v2 is a small convolution stack and the faster of the two. "
                   "row_flow_v3 is a windowed-attention model, roughly four times the parameters.");
    model->appendOption("row_flow_v2");
    model->appendOption("row_flow_v3");
    model->setDefault(0);
    page->addChild(*model);

    // -- Stereo ------------------------------------------------------------
    OFX::GroupParamDescriptor* stereo = desc.defineGroupParam("stereoGroup");
    stereo->setLabels("Stereo", "Stereo", "Stereo");

    OFX::DoubleParamDescriptor* divergence = desc.defineDoubleParam("divergence");
    divergence->setLabels("Divergence", "Divergence", "Divergence");
    divergence->setHint("Strength of the 3D effect, as a percentage of image width. 0-2 is a reasonable range.");
    divergence->setDefault(2.0);
    divergence->setRange(0.0, 10.0);
    divergence->setDisplayRange(0.0, 5.0);
    divergence->setParent(*stereo);
    page->addChild(*divergence);

    OFX::DoubleParamDescriptor* convergence = desc.defineDoubleParam("convergence");
    convergence->setLabels("Convergence", "Convergence", "Convergence");
    convergence->setHint("The depth value that sits on the screen plane. "
                         "0 puts everything behind the screen, 1 everything in front.");
    convergence->setDefault(0.5);
    convergence->setRange(-1.0, 2.0);
    convergence->setDisplayRange(0.0, 1.0);
    convergence->setParent(*stereo);
    page->addChild(*convergence);

    OFX::BooleanParamDescriptor* border = desc.defineBooleanParam("preserveScreenBorder");
    border->setLabels("Preserve Screen Border", "Preserve Screen Border", "Preserve Screen Border");
    border->setHint("Taper parallax to zero at the left and right edges.");
    border->setDefault(false);
    border->setParent(*stereo);
    page->addChild(*border);

    // -- Depth -------------------------------------------------------------
    OFX::GroupParamDescriptor* depthGroup = desc.defineGroupParam("depthGroup");
    depthGroup->setLabels("Depth", "Depth", "Depth");

    OFX::BooleanParamDescriptor* inverted = desc.defineBooleanParam("depthInverted");
    inverted->setLabels("Depth Is Inverted", "Depth Is Inverted", "Depth Is Inverted");
    inverted->setHint("External depth tools disagree on whether white is near or far.");
    inverted->setDefault(false);
    inverted->setParent(*depthGroup);
    page->addChild(*inverted);

    OFX::DoubleParamDescriptor* foreground = desc.defineDoubleParam("foregroundScale");
    foreground->setLabels("Foreground Scale", "Foreground Scale", "Foreground Scale");
    foreground->setHint("iw3's foreground scaling. 0 is off.");
    foreground->setDefault(0.0);
    foreground->setRange(-3.0, 3.0);
    foreground->setDisplayRange(-3.0, 3.0);
    foreground->setParent(*depthGroup);
    page->addChild(*foreground);

    OFX::ChoiceParamDescriptor* mapperType = desc.defineChoiceParam("mapperType");
    mapperType->setLabels("Mapper Type", "Mapper Type", "Mapper Type");
    mapperType->setHint("Which family Foreground Scale selects from. Only matters when it is non-zero.");
    mapperType->appendOption("Multiply");
    mapperType->appendOption("Shift");
    mapperType->setDefault(0);
    mapperType->setParent(*depthGroup);
    page->addChild(*mapperType);

    OFX::IntParamDescriptor* stereoWidth = desc.defineIntParam("stereoWidth");
    stereoWidth->setLabels("Stereo Width", "Stereo Width", "Stereo Width");
    stereoWidth->setHint("Width the depth is resized to before warping. 0 picks the size a depth model "
                         "would have produced. Leaving depth at full frame resolution makes the warp stripe.");
    stereoWidth->setDefault(0);
    stereoWidth->setRange(0, 8192);
    stereoWidth->setDisplayRange(0, 3840);
    stereoWidth->setParent(*depthGroup);
    page->addChild(*stereoWidth);

    OFX::ChoiceParamDescriptor* depthRange = desc.defineChoiceParam("depthRange");
    depthRange->setLabels("Depth Range", "Depth Range", "Depth Range");
    depthRange->setHint("Resolve expands video-range clips to full range, which is what iw3 does too, "
                        "so leave this alone. Only useful for a file whose range tag is wrong.");
    depthRange->appendOption("As delivered");
    depthRange->appendOption("Undo video expansion");
    depthRange->setDefault(0);
    depthRange->setParent(*depthGroup);
    page->addChild(*depthRange);

    // -- Output ------------------------------------------------------------
    OFX::ChoiceParamDescriptor* output = desc.defineChoiceParam("output");
    output->setLabels("Output", "Output", "Output");
    output->setHint("What the output image contains.");
    output->appendOption("Anaglyph (Dubois red/cyan)");
    output->appendOption("Left eye");
    output->appendOption("Right eye");
    output->appendOption("Half SBS");
    output->appendOption("Depth (debug)");
    // Appended rather than placed beside Half SBS: the option index is what a
    // saved comp stores, so inserting in the middle would change what an
    // existing project means.
    output->appendOption("Full SBS (not supported by Resolve - see README)");
    output->setDefault(0);
    page->addChild(*output);
}

OFX::ImageEffect* Iw3StereoFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new Iw3StereoEffect(handle);
}

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static Iw3StereoFactory factory(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&factory);
}
