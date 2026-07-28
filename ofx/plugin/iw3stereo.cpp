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

#include "monobw_gpu.h"
#include "monobw_math.h"
#include "ort_runtime.h"
#include "probe_log.h"
#include "stereo_gpu.h"
#include "stereo_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
        _maskInnerDilation = fetchIntParam("maskInnerDilation");
        _maskOuterDilation = fetchIntParam("maskOuterDilation");
        _inpaintMaxWidth = fetchIntParam("inpaintMaxWidth");

        startRuntimeBringUp();
    }

    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments&, OFX::Clip*&, double&) override { return false; }

private:
    iw3::Settings readSettings(double time) const;
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
    OFX::IntParam* _maskInnerDilation = nullptr;
    OFX::IntParam* _maskOuterDilation = nullptr;
    OFX::IntParam* _inpaintMaxWidth = nullptr;

    // Reused across frames; resized only when the frame size changes.
    std::vector<float> _image, _depthFull, _depthSmall, _x, _left, _right, _composed;
    iw3::DepthResizer _resizer;
    iw3::GpuPipeline _gpu;
    iw3::MonoBwGpu _monobw;
    iw3::MonoBwVideoGpu _video;
    std::string _lastMessage;
    unsigned _frameCounter = 0;

    // Returns false if the GPU path could not be taken, so render() can fall
    // through to the CPU one.
    bool renderCuda(const OFX::RenderArguments& args, const iw3::Settings& settings,
                    OFX::Image* dst, OFX::Image* src, OFX::Image* depth,
                    int components, const OfxRectI& window);

    // Warp twelve frames and run the temporal graph over them, filling the
    // cache. Only called when the cache does not already hold this window.
    bool buildVideoWindow(const OFX::RenderArguments& args, const iw3::Settings& settings,
                          long long firstFrame, long long currentFrame,
                          const OfxRectI& window, int depthWidth,
                          int depthHeight, void* stream);

public:
    // The frames this output frame needs. Declaring the range is the documented
    // way to ask a host for other frames, and Resolve honours it -- probed in
    // ofx/probe/iw3temporal.cpp, for both clips and with CUDA render on.
    virtual void getFramesNeeded(const OFX::FramesNeededArguments& args,
                                 OFX::FramesNeededSetter& frames) override;
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
            directory + L"\\stereo_warp.onnx",        // 0: row_flow_v2
            directory + L"\\stereo_warp_v3.onnx",     // 1: row_flow_v3
            directory + L"\\light_inpaint_v1.onnx",         // 2: monobw_inpaint
            directory + L"\\light_video_inpaint_v1.onnx",   // 3: monobw_inpaint_video
        };
        // The two inpaint graphs get the memory-conservative session options;
        // the warp graphs neither need them nor benefit.
        const std::vector<bool> conserve = {false, false, true, true};
        const bool ok = runtime->open(directory + L"\\ort", graphs, true, conserve);
        probe::logf("---- iw3 Stereo: ONNX Runtime bring-up (%s) ----", ok ? "OK" : "FAILED");
        for (const std::string& line : runtime->report())
        {
            probe::logf("    %s", line.c_str());
        }
        return runtime;
    }();
    return *instance;
}

// Drain whatever ONNX Runtime has said since the last time anyone asked, into
// the log.
//
// open() dumps the report once at bring-up and nothing read it again, so every
// render-time ORT error went into a vector and stayed there -- which is how a
// "GPU inpaint failed" with no cause attached happened. Called on every failure
// path that touches the runtime.
static void logRuntimeReport(const char* context)
{
    for (const std::string& line : sharedRuntime().takeNewReport())
    {
        probe::logf("iw3 Stereo: %s: %s", context, line.c_str());
    }
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
    settings.method = choice == 1 ? iw3::Method::RowFlowV3
                    : choice == 2 ? iw3::Method::MonoBwInpaint
                    : choice == 3 ? iw3::Method::MonoBwInpaintVideo
                                  : iw3::Method::RowFlowV2;
    settings.model = size_t(choice >= 1 && choice <= 3 ? choice : 0);
    _maskInnerDilation->getValueAtTime(time, settings.maskInnerDilation);
    _maskOuterDilation->getValueAtTime(time, settings.maskOuterDilation);
    _inpaintMaxWidth->getValueAtTime(time, settings.inpaintMaxWidth);
    _output->getValueAtTime(time, choice);
    settings.output = choice == 1 ? iw3::OutputMode::LeftEye
                    : choice == 2 ? iw3::OutputMode::RightEye
                    : choice == 3 ? iw3::OutputMode::HalfSbs
                    : choice == 4 ? iw3::OutputMode::DepthDebug
                                  : iw3::OutputMode::Anaglyph;
    return settings;
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
namespace
{

// A stand-in for "have any of the settings that affect the picture changed".
// The window cache holds twelve frames' worth of work, so it has to be dropped
// when any of them moves, and comparing a fingerprint is cheaper and less
// error-prone than comparing the struct field by field.
unsigned long long settingsFingerprint(const iw3::Settings& settings, int width, int height)
{
    unsigned long long hash = 1469598103934665603ull;
    const auto mix = [&hash](unsigned long long value)
    {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    const auto mixDouble = [&mix](double value)
    {
        unsigned long long bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        mix(bits);
    };
    mixDouble(settings.divergence);
    mixDouble(settings.convergence);
    mixDouble(settings.foregroundScale);
    mix(unsigned(settings.preserveScreenBorder));
    mix(unsigned(settings.depthInverted));
    mix(unsigned(settings.undoVideoRange));
    mix(unsigned(settings.mapperType == iw3::MapperType::Shift));
    mix(unsigned(settings.stereoWidth));
    mix(unsigned(settings.maskInnerDilation));
    mix(unsigned(settings.maskOuterDilation));
    mix(unsigned(settings.inpaintMaxWidth));
    mix(unsigned(width));
    mix(unsigned(height));
    return hash;
}

}  // namespace

void Iw3StereoEffect::getFramesNeeded(const OFX::FramesNeededArguments& args,
                                      OFX::FramesNeededSetter& frames)
{
    const iw3::Settings settings = readSettings(args.time);
    if (settings.method != iw3::Method::MonoBwInpaintVideo)
    {
        return;  // the default is the current frame, which is all the rest need
    }

    const long long frame = (long long)std::llround(args.time);
    const long long first = iw3::MonoBwVideoGpu::windowFirstFrame(frame);
    OfxRangeD range;
    range.min = double(first);
    range.max = double(first + iw3::MonoBwVideoGpu::kSequence - 1);

    // Both clips: the temporal model needs twelve frames of depth as much as
    // twelve of colour, and the probe confirmed Resolve supplies both.
    if (_srcClip)
    {
        frames.setFramesNeeded(*_srcClip, range);
    }
    if (_depthClip && _depthClip->isConnected())
    {
        frames.setFramesNeeded(*_depthClip, range);
    }
}

bool Iw3StereoEffect::buildVideoWindow(const OFX::RenderArguments& args,
                                       const iw3::Settings& settings,
                                       long long firstFrame, long long currentFrame,
                                       const OfxRectI& window,
                                       int depthWidth, int depthHeight, void* stream)
{
    iw3::OrtRuntime& ort = sharedRuntime();
    const int width = window.x2 - window.x1;
    const int height = window.y2 - window.y1;
    const iw3::Mapper mapper(settings.foregroundScale, settings.mapperType);
    int workWidth = 0, workHeight = 0;
    iw3::math::inpaintWorkingSize(width, height, settings.inpaintMaxWidth,
                                  workWidth, workHeight);
    const int64_t shape[4] = {iw3::MonoBwVideoGpu::kSequence, 3, workHeight, workWidth};

    for (int pass = 0; pass < 2; ++pass)
    {
        const bool rightEye = pass == 1;

        for (int slot = 0; slot < iw3::MonoBwVideoGpu::kSequence; ++slot)
        {
            // Clip boundaries: a window centred near the start or end runs off
            // the material, and the probe showed those fetches come back null.
            // Repeating the nearest frame that exists is what iw3's own padding
            // does for a short sequence.
            //
            // The search steps *towards the frame being rendered*, which is the
            // one frame guaranteed to exist, so it clamps at both ends. Walking
            // only backwards -- which is what this did first -- cannot move at
            // all at slot 0, and slot 0 of the first window is frame -3. That
            // failed the whole window and fell back to passing the source
            // through, so the first six frames of a render came out as flat 2D
            // before the seventh snapped into stereo.
            std::unique_ptr<OFX::Image> frameSrc;
            std::unique_ptr<OFX::Image> frameDepth;
            const long long wanted = firstFrame + slot;
            const long long step = (wanted < currentFrame) ? 1 : -1;
            for (long long time = wanted;
                 !frameSrc && ((step > 0) ? (time <= currentFrame) : (time >= currentFrame));
                 time += step)
            {
                frameSrc.reset(_srcClip->fetchImage(double(time)));
                if (frameSrc && _depthClip && _depthClip->isConnected())
                {
                    frameDepth.reset(_depthClip->fetchImage(double(time)));
                    if (!frameDepth)
                    {
                        frameSrc.reset();
                    }
                }
            }
            if (!frameSrc || !frameDepth)
            {
                report("Could not fetch the frames the temporal model needs.", true);
                return false;
            }

            const OfxRectI& srcBounds = frameSrc->getBounds();
            const OfxRectI& depthBounds = frameDepth->getBounds();
            _gpu.packSource(static_cast<const float*>(frameSrc->getPixelData()),
                            size_t(frameSrc->getRowBytes()) / sizeof(float),
                            componentCount(frameSrc->getPixelComponents()),
                            window.x1 - srcBounds.x1, window.y1 - srcBounds.y1,
                            srcBounds.x2 - srcBounds.x1, srcBounds.y2 - srcBounds.y1, stream);
            _gpu.packDepth(static_cast<const float*>(frameDepth->getPixelData()),
                           size_t(frameDepth->getRowBytes()) / sizeof(float),
                           componentCount(frameDepth->getPixelComponents()),
                           window.x1 - depthBounds.x1, window.y1 - depthBounds.y1,
                           depthBounds.x2 - depthBounds.x1, depthBounds.y2 - depthBounds.y1,
                           settings.undoVideoRange, settings.depthInverted,
                           mapper.params(), stream);
            _gpu.resizeDepth(stream);

            if (!_monobw.prepareEye(_gpu.imageDevice(), _gpu.depthDevice(), rightEye,
                                    settings.divergence, settings.convergence,
                                    settings.preserveScreenBorder,
                                    settings.maskInnerDilation, settings.maskOuterDilation,
                                    depthWidth, stream))
            {
                report("monobw warp failed: " + _monobw.error(), true);
                return false;
            }
            _monobw.prepareInpaintInput(settings.inpaintMaxWidth, stream);
            _video.storeFrame(slot, _monobw.inpaintEyeHalfDevice(),
                              _monobw.processedMaskHalfDevice(), stream);
        }

        if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
        {
            report("CUDA stream sync failed -- falling back to the CPU.", true);
            return false;
        }

        const float* filled = nullptr;
        if (!ort.runInpaintDevice(3, reinterpret_cast<const float*>(_video.eyesDevice()),
                                  reinterpret_cast<const float*>(_video.masksDevice()),
                                  shape, &filled))
        {
            logRuntimeReport("video inpaint inference");
            report("GPU temporal inpaint failed; see the log.", true);
            return false;
        }
        _video.cacheOutput(rightEye, filled, stream);
    }

    return _video.ok();
}

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
    const int width = window.x2 - window.x1;
    const int height = window.y2 - window.y1;

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
    if (settings.method != iw3::Method::MonoBwInpaint &&
        settings.method != iw3::Method::MonoBwInpaintVideo)
    {
        // monobw takes the depth directly; the three-channel input tensor with
        // its divergence and convergence planes is the row_flow contract.
        _gpu.buildInputTensor(settings.divergence, settings.convergence,
                              settings.preserveScreenBorder, stream);
    }

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

    if (settings.method == iw3::Method::MonoBwInpaintVideo)
    {
        if (model != settings.model || ort.outputCount(model) != 1)
        {
            logRuntimeReport("video inpaint graph");
            report("The temporal inpaint graph did not load; see the log.", true);
            return false;
        }
        int workWidth = 0, workHeight = 0;
        iw3::math::inpaintWorkingSize(width, height, settings.inpaintMaxWidth,
                                      workWidth, workHeight);
        if (!_monobw.prepare(width, height, depthWidth, depthHeight) ||
            !_video.prepare(workWidth, workHeight))
        {
            report("GPU setup failed: " + _monobw.error() + _video.error(), true);
            return false;
        }

        // Which window this frame belongs to is derived from the frame number
        // alone, so Resolve's gaps and repeats all land on the same window and
        // the cache is hit whatever order frames arrive in.
        const long long frame = (long long)std::llround(args.time);
        const long long windowIndex = iw3::MonoBwVideoGpu::windowIndex(frame);
        const long long firstFrame = iw3::MonoBwVideoGpu::windowFirstFrame(frame);
        const unsigned long long fingerprint = settingsFingerprint(settings, width, height);

        if (!_video.holds(windowIndex, fingerprint))
        {
            _video.invalidate();
            if (!buildVideoWindow(args, settings, firstFrame, frame, window,
                                  depthWidth, depthHeight, stream))
            {
                return false;
            }
            _video.markHeld(windowIndex, fingerprint);
        }

        // The offset within the window's kept frames. The window starts kPad
        // before the first frame it emits, so this is simply the position
        // within the stride.
        const int offset = int(frame - windowIndex * iw3::MonoBwVideoGpu::kStride);
        // The cache holds the graph's output, still half and still at the
        // working resolution; this widens it, composites it back into the
        // full-resolution eye, and undoes the mirror the left eye went in with.
        //
        // The right eye is computed second on purpose: finishInpaintOutput and
        // finishEye each return a buffer the next call reuses, and mirroring
        // the left eye back copies it out of the way.
        _monobw.prepareEye(_gpu.imageDevice(), _gpu.depthDevice(), false,
                           settings.divergence, settings.convergence,
                           settings.preserveScreenBorder,
                           settings.maskInnerDilation, settings.maskOuterDilation,
                           depthWidth, stream);
        _monobw.prepareInpaintInput(settings.inpaintMaxWidth, stream);
        left = _monobw.finishEye(
            _monobw.finishInpaintOutput(_video.cachedFrame(false, offset), stream),
            false, stream);

        _monobw.prepareEye(_gpu.imageDevice(), _gpu.depthDevice(), true,
                           settings.divergence, settings.convergence,
                           settings.preserveScreenBorder,
                           settings.maskInnerDilation, settings.maskOuterDilation,
                           depthWidth, stream);
        _monobw.prepareInpaintInput(settings.inpaintMaxWidth, stream);
        right = _monobw.finishEye(
            _monobw.finishInpaintOutput(_video.cachedFrame(true, offset), stream),
            true, stream);
    }
    else if (settings.method == iw3::Method::MonoBwInpaint)
    {
        if (model != settings.model || ort.outputCount(model) != 1)
        {
            logRuntimeReport("inpaint graph");
            report("The inpaint graph did not load; see the log.", true);
            return false;
        }
        if (!_monobw.prepare(width, height, depthWidth, depthHeight))
        {
            report("GPU setup failed: " + _monobw.error() + " -- falling back to the CPU.", true);
            return false;
        }

        // The left eye runs first, and that ordering is load-bearing. The
        // graph's output lives in the session's bound output buffer and the
        // next call overwrites it, so the eye whose result has to survive a
        // second inference is the one that gets copied on the way out --
        // finishEye() mirrors the left eye into its own buffer, while the right
        // eye is already in frame orientation and is returned in place.
        for (int pass = 0; pass < 2; ++pass)
        {
            const bool rightEye = pass == 1;
            if (!_monobw.prepareEye(_gpu.imageDevice(), _gpu.depthDevice(), rightEye,
                                    settings.divergence, settings.convergence,
                                    settings.preserveScreenBorder,
                                    settings.maskInnerDilation, settings.maskOuterDilation,
                                    depthWidth, stream))
            {
                report("monobw warp failed: " + _monobw.error(), true);
                return false;
            }
            if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
            {
                report("CUDA stream sync failed -- falling back to the CPU.", true);
                return false;
            }

            // The graph is half precision, because in fp32 the twelve-frame
            // video window does not fit in 17 GiB and the per-frame one costs
            // twice as much for nothing. Everything this plugin computes is
            // fp32, so the conversion is a pass either side rather than a
            // change of type throughout.
            const bool half = ort.inputIsHalf(model);
            const float* filled = nullptr;
            _monobw.prepareInpaintInput(half ? settings.inpaintMaxWidth : 0, stream);
            if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
            {
                report("CUDA stream sync failed -- falling back to the CPU.", true);
                return false;
            }
            const int64_t inpaintShape[4] = {1, 3, _monobw.inpaintHeight(),
                                             _monobw.inpaintWidth()};
            const float* eyeIn = half
                ? reinterpret_cast<const float*>(_monobw.inpaintEyeHalfDevice())
                : _monobw.inpaintEyeDevice();
            const float* maskIn = half
                ? reinterpret_cast<const float*>(_monobw.processedMaskHalfDevice())
                : _monobw.processedMaskDevice();
            if (!ort.runInpaintDevice(model, eyeIn, maskIn, inpaintShape, &filled))
            {
                logRuntimeReport("inpaint inference");
                report("GPU inpaint failed; see the log.", true);
                return false;
            }
            filled = _monobw.finishInpaintOutput(filled, stream);
            const float* eye = _monobw.finishEye(filled, rightEye, stream);
            if (rightEye)
            {
                right = eye;
            }
            else
            {
                left = eye;
            }
        }
    }
    else if (!ort.runDevice(model, _gpu.imageDevice(), imageShape, _gpu.inputTensorDevice(), xShape,
                            iw3::deltaScale(depthWidth), &left, &right))
    {
        logRuntimeReport("warp inference");
        report("GPU inference failed; see the log. Falling back to the CPU.", true);
        return false;
    }

    float* destination = static_cast<float*>(dst->getPixelData()) +
        size_t(window.y1 - dstBounds.y1) * destinationPitch +
        size_t(window.x1 - dstBounds.x1) * size_t(components);

    _gpu.compose(settings.output, left, right, stream);
    _gpu.unpack(destination, destinationPitch, components, stream);

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

    if (settings.method == iw3::Method::MonoBwInpaint ||
        settings.method == iw3::Method::MonoBwInpaintVideo)
    {
        // Deliberately GPU-only. The CPU path below implements the backward
        // warp, not this pipeline, and the missing half is a 2.26M-parameter
        // network at full frame resolution -- on CPU ONNX Runtime that is
        // seconds a frame, which is not a fallback anyone would choose over
        // switching the Model parameter back.
        report("The monobw_inpaint models need the GPU render path (NVIDIA, "
               "and Fusion's GPU processing enabled). Pick row_flow_v2 or "
               "row_flow_v3 for the CPU.", true);
        passSourceThrough(args, dst.get(), src.get(), window, components);
        return;
    }

    // --- unpack the two inputs into planar float ---------------------------
    const auto cpuStarted = std::chrono::steady_clock::now();

    _image.resize(pixels * 3);
    _depthFull.resize(pixels);

    const int depthComponents = componentCount(depth->getPixelComponents());
    const iw3::Mapper mapper(settings.foregroundScale, settings.mapperType);

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
    switch (settings.output)
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
    model->setHint("row_flow_v2 is a small convolution stack and the faster of the two warps. "
                   "row_flow_v3 is a windowed-attention model, roughly four times the parameters. "
                   "monobw_inpaint is a different pipeline: it warps forwards, finds the holes "
                   "that opens, and fills them with a network instead of smearing an edge into "
                   "them. Better at occlusions, several times slower, and NVIDIA only. "
                   "monobw_inpaint_video is the same pipeline with a temporal model that sees "
                   "twelve frames at once, so the fills stop flickering; it costs about twice "
                   "as much again and needs frames either side of the one being rendered.");
    model->appendOption("row_flow_v2");
    model->appendOption("row_flow_v3");
    model->appendOption("monobw_inpaint");
    model->appendOption("monobw_inpaint_video");
    model->setDefault(0);
    page->addChild(*model);

    // The two mask dilations, which only monobw_inpaint reads. Left visible
    // rather than hidden behind the Model choice: an OFX host may or may not
    // honour a dynamic enable, and a control that silently does nothing is
    // less confusing than one that appears and disappears.
    OFX::IntParamDescriptor* innerDilation = desc.defineIntParam("maskInnerDilation");
    innerDilation->setLabels("Mask Inner Dilation", "Mask Inner", "Mask Inner Dilation");
    innerDilation->setHint("monobw_inpaint only. Grows the hole mask towards the occluding edge "
                           "before filling, which helps when the depth map's edge sits slightly "
                           "inside the object's. Counted against the depth width, so it means the "
                           "same at any resolution.");
    innerDilation->setRange(0, 16);
    innerDilation->setDisplayRange(0, 8);
    innerDilation->setDefault(0);
    page->addChild(*innerDilation);

    OFX::IntParamDescriptor* outerDilation = desc.defineIntParam("maskOuterDilation");
    outerDilation->setLabels("Mask Outer Dilation", "Mask Outer", "Mask Outer Dilation");
    outerDilation->setHint("monobw_inpaint only. Grows the hole mask away from the occluding "
                           "edge, giving the network more room to invent into.");
    outerDilation->setRange(0, 16);
    outerDilation->setDisplayRange(0, 8);
    outerDilation->setDefault(0);
    page->addChild(*outerDilation);

    OFX::IntParamDescriptor* inpaintWidth = desc.defineIntParam("inpaintMaxWidth");
    inpaintWidth->setLabels("Inpaint Max Width", "Inpaint Width", "Inpaint Max Width");
    inpaintWidth->setHint("Both monobw models only. Caps the width the inpaint network runs at, "
                          "0 for the frame's own. Its memory scales with area -- at HD the "
                          "temporal model needs about 9 GB, 4.5 at 1280 wide and 3.4 at 960 -- "
                          "so this is the setting that fits it on a smaller card. The output "
                          "stays full resolution either way: only the invented pixels are "
                          "computed small, and everything outside a hole keeps its own detail.");
    inpaintWidth->setRange(0, 8192);
    inpaintWidth->setDisplayRange(0, 3840);
    inpaintWidth->setDefault(0);
    page->addChild(*inpaintWidth);

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
    // No Full SBS: an OFX effect cannot output an image larger than its input
    // unless the host supports multiple resolutions, and Resolve's reports that
    // it does not. The bundle README has the Fusion setup that does work.
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
