// Phase 0e probe: run the real stereo warp, through ONNX Runtime, inside
// Resolve's own process.
//
// Two questions, neither answerable from a standalone script:
//
//   1. Does a CUDA execution provider come up inside a host that already holds
//      its own CUDA context? This is the classic failure mode.
//   2. What does Resolve's colour management do to a depth clip on the way in?
//      (Phase 0f -- the probe logs raw pixel statistics for both inputs, to be
//      compared against the file on disk.)
//
// It is also a first cut of the real plugin: Source + Depth in, a genuine
// stereo pair out. Everything lands in %LOCALAPPDATA%\iw3probe\probe.log.

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "ort_runtime.h"
#include "probe_log.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#define kPluginName "iw3 Stereo (ORT Probe)"
#define kPluginGrouping "iw3"
#define kPluginIdentifier "com.nunif.iw3.ort"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kDepthClipName "Depth"

namespace
{

// Where this .ofx lives, so the runtime and the model are found relative to the
// bundle rather than relative to Resolve's working directory.
std::wstring bundleDirectory()
{
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&bundleDirectory), &self);
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(self, path, MAX_PATH);
    std::wstring full(path);
    const size_t slash = full.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : full.substr(0, slash);
}

struct Stats
{
    float minimum = 0.0f;
    float maximum = 0.0f;
    double mean = 0.0;
};

// Phase 0f depends on these numbers: if Resolve applies a colour transform to
// the depth clip, the values here will not match the file.
Stats channelStats(const OFX::Image* image, const OfxRectI& window, int channel, int components)
{
    Stats stats;
    bool first = true;
    double total = 0.0;
    size_t count = 0;
    for (int y = window.y1; y < window.y2; y += 4)
    {
        for (int x = window.x1; x < window.x2; x += 4)
        {
            const float* pixel = static_cast<const float*>(
                const_cast<OFX::Image*>(image)->getPixelAddress(x, y));
            if (!pixel)
            {
                continue;
            }
            const float value = pixel[std::min(channel, components - 1)];
            if (first)
            {
                stats.minimum = stats.maximum = value;
                first = false;
            }
            stats.minimum = std::min(stats.minimum, value);
            stats.maximum = std::max(stats.maximum, value);
            total += value;
            ++count;
        }
    }
    stats.mean = count ? total / double(count) : 0.0;
    return stats;
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

}  // namespace

////////////////////////////////////////////////////////////////////////////////

class OrtStereoEffect : public OFX::ImageEffect
{
public:
    explicit OrtStereoEffect(OfxImageEffectHandle handle)
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
            _depthClip = nullptr;  // filter context: no second input exists
        }
        _divergence = fetchDoubleParam("divergence");
        _convergence = fetchDoubleParam("convergence");
        _output = fetchChoiceParam("output");
        _depthInverted = fetchBooleanParam("depthInverted");
        _stereoWidth = fetchIntParam("stereoWidth");
    }

    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments&, OFX::Clip*&, double&) override { return false; }

private:
    iw3::OrtRuntime& runtime();

    OFX::Clip* _dstClip = nullptr;
    OFX::Clip* _srcClip = nullptr;
    OFX::Clip* _depthClip = nullptr;
    OFX::DoubleParam* _divergence = nullptr;
    OFX::DoubleParam* _convergence = nullptr;
    OFX::ChoiceParam* _output = nullptr;
    OFX::BooleanParam* _depthInverted = nullptr;
    OFX::IntParam* _stereoWidth = nullptr;

    std::vector<float> _image, _x, _left, _right;
};

// One runtime for the whole process. Sessions are expensive to build and
// Resolve creates and destroys effect instances freely, so this must not be
// per-instance -- the same trap as iw3's create_stereo_model(), which re-reads
// its checkpoint on every call.
// Leaked on purpose -- see the note on sharedRuntime() in plugin/iw3stereo.cpp.
// Destroying this at process exit hangs Resolve.
iw3::OrtRuntime& OrtStereoEffect::runtime()
{
    static iw3::OrtRuntime* instance = []()
    {
        auto* runtime = new iw3::OrtRuntime();
        const std::wstring directory = bundleDirectory();
        const bool ok = runtime->open(directory + L"\\ort", {directory + L"\\stereo_warp.onnx"}, true);
        probe::logf("---- ONNX Runtime bring-up (%s) ----", ok ? "OK" : "FAILED");
        for (const std::string& line : runtime->report())
        {
            probe::logf("    %s", line.c_str());
        }
        return runtime;
    }();
    return *instance;
}

void OrtStereoEffect::render(const OFX::RenderArguments& args)
{
    double divergence = 2.0, convergence = 0.5;
    int output = 0, stereoWidth = 0;
    bool depthInverted = false;
    _divergence->getValueAtTime(args.time, divergence);
    _convergence->getValueAtTime(args.time, convergence);
    _output->getValueAtTime(args.time, output);
    _depthInverted->getValueAtTime(args.time, depthInverted);
    _stereoWidth->getValueAtTime(args.time, stereoWidth);

    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(_srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> depth(
        _depthClip && _depthClip->isConnected() ? _depthClip->fetchImage(args.time) : nullptr);

    if (!dst || !src)
    {
        return;
    }

    const int components = componentCount(dst->getPixelComponents());
    if (components == 0 || dst->getPixelDepth() != OFX::eBitDepthFloat)
    {
        return;
    }

    const OfxRectI& window = args.renderWindow;
    const int width = window.x2 - window.x1;
    const int height = window.y2 - window.y1;
    const size_t pixels = size_t(width) * size_t(height);

    // Log the input statistics once every 30 frames; every frame would swamp
    // the log during playback.
    static std::atomic<int> frameCounter{0};
    const bool verbose = (frameCounter++ % 30) == 0;
    if (verbose)
    {
        const Stats srcStats = channelStats(src.get(), window, 0, components);
        probe::logf("render t=%.1f %dx%d src R: min=%.5f max=%.5f mean=%.5f",
                    args.time, width, height, srcStats.minimum, srcStats.maximum, srcStats.mean);
        if (depth)
        {
            const Stats depthStats = channelStats(depth.get(), window, 0,
                                                  componentCount(depth->getPixelComponents()));
            probe::logf("    depth R: min=%.5f max=%.5f mean=%.5f   <- compare against the file for 0f",
                        depthStats.minimum, depthStats.maximum, depthStats.mean);
        }
        else
        {
            probe::logf("    depth: not connected");
        }
    }

    iw3::OrtRuntime& ort = runtime();
    if (!ort.ready() || !depth)
    {
        // Pass the source through so a failure is visible as "no effect"
        // rather than as black.
        for (int y = window.y1; y < window.y2; ++y)
        {
            for (int x = window.x1; x < window.x2; ++x)
            {
                float* out = static_cast<float*>(dst->getPixelAddress(x, y));
                const float* in = static_cast<const float*>(src->getPixelAddress(x, y));
                if (out && in)
                {
                    for (int c = 0; c < components; ++c) out[c] = in[c];
                }
            }
        }
        return;
    }

    // --- pack: Resolve's interleaved RGBA -> the graph's planar NCHW --------
    _image.resize(pixels * 3);
    _x.resize(pixels * 3);
    _left.resize(pixels * 3);
    _right.resize(pixels * 3);

    const double baseSize = double(std::max(width, height));
    const double divergencePix = divergence * 0.5 * 0.01 * baseSize;
    const float divergenceValue = float(divergencePix / 32.0);
    const float convergenceValue = float((-divergencePix * convergence) / 32.0);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const float* pixel = static_cast<const float*>(
                src->getPixelAddress(window.x1 + x, window.y1 + y));
            const size_t index = size_t(y) * size_t(width) + size_t(x);
            for (int c = 0; c < 3; ++c)
            {
                _image[size_t(c) * pixels + index] = pixel ? pixel[c] : 0.0f;
            }

            const float* depthPixel = static_cast<const float*>(
                depth->getPixelAddress(window.x1 + x, window.y1 + y));
            float value = depthPixel ? depthPixel[0] : 0.0f;
            value = std::min(std::max(value, 0.0f), 1.0f);
            _x[index] = depthInverted ? 1.0f - value : value;
            _x[pixels + index] = divergenceValue;
            _x[2 * pixels + index] = convergenceValue;
        }
    }

    const int64_t imageShape[4] = {1, 3, height, width};
    const int64_t xShape[4] = {1, 3, height, width};
    // Note the *depth's* width, which here equals the image's because the
    // stereo_width resize is not implemented in this probe.
    const float deltaScale = float(1.0 / double(width / 2 - 1));

    if (!ort.run(0, _image.data(), imageShape, _x.data(), xShape, deltaScale,
                 _left.data(), _right.data(), pixels * 3))
    {
        probe::logf("    Run FAILED");
        return;
    }
    if (verbose)
    {
        probe::logf("    inference %.2f ms on %s (%dx%d, stereoWidth param=%d ignored in probe)",
                    ort.lastRunMilliseconds(), ort.provider().c_str(), width, height, stereoWidth);
    }

    // --- unpack ------------------------------------------------------------
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float* out = static_cast<float*>(dst->getPixelAddress(window.x1 + x, window.y1 + y));
            if (!out)
            {
                continue;
            }
            const size_t index = size_t(y) * size_t(width) + size_t(x);
            const float* eye = (output == 1) ? _right.data() : _left.data();

            if (output == 2)
            {
                // Anaglyph: left eye's red, right eye's green and blue. Makes a
                // correct warp obvious at a glance.
                out[0] = _left[index];
                out[1] = _right[pixels + index];
                out[2] = _right[2 * pixels + index];
            }
            else
            {
                for (int c = 0; c < 3; ++c)
                {
                    out[c] = eye[size_t(c) * pixels + index];
                }
            }
            if (components > 3)
            {
                out[3] = 1.0f;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

mDeclarePluginFactory(OrtStereoFactory, {}, {});

void OrtStereoFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription("Phase 0e probe: iw3's row_flow_v2 warp via ONNX Runtime.");

    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
}

void OrtStereoFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context)
{
    probe::logf("iw3ort describeInContext: context=%d", int(context));

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

    OFX::DoubleParamDescriptor* divergence = desc.defineDoubleParam("divergence");
    divergence->setLabels("Divergence", "Divergence", "Divergence");
    divergence->setHint("Stereo strength, as a percentage of image width");
    divergence->setDefault(2.0);
    divergence->setRange(0.0, 20.0);
    divergence->setDisplayRange(0.0, 10.0);
    page->addChild(*divergence);

    OFX::DoubleParamDescriptor* convergence = desc.defineDoubleParam("convergence");
    convergence->setLabels("Convergence", "Convergence", "Convergence");
    convergence->setHint("Depth value that sits on the screen plane");
    convergence->setDefault(0.5);
    convergence->setRange(-1.0, 2.0);
    convergence->setDisplayRange(0.0, 1.0);
    page->addChild(*convergence);

    OFX::ChoiceParamDescriptor* output = desc.defineChoiceParam("output");
    output->setLabels("Output", "Output", "Output");
    output->appendOption("Left eye");
    output->appendOption("Right eye");
    output->appendOption("Anaglyph");
    output->setDefault(2);
    page->addChild(*output);

    OFX::BooleanParamDescriptor* inverted = desc.defineBooleanParam("depthInverted");
    inverted->setLabels("Depth Is Inverted", "Depth Is Inverted", "Depth Is Inverted");
    inverted->setHint("External depth tools disagree on whether white is near or far");
    inverted->setDefault(false);
    page->addChild(*inverted);

    OFX::IntParamDescriptor* stereoWidth = desc.defineIntParam("stereoWidth");
    stereoWidth->setLabels("Stereo Width", "Stereo Width", "Stereo Width");
    stereoWidth->setHint("Depth is resized to this width before warping (not yet applied in this probe)");
    stereoWidth->setDefault(0);
    stereoWidth->setRange(0, 8192);
    stereoWidth->setDisplayRange(0, 3840);
    page->addChild(*stereoWidth);
}

OFX::ImageEffect* OrtStereoFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new OrtStereoEffect(handle);
}

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static OrtStereoFactory factory(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&factory);
}
