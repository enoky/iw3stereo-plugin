// Phase 0 probe plugins for the iw3 Resolve project.
//
// This binary answers the questions in Phase 0 of the plan that can only be
// answered by a plugin actually running inside Resolve:
//
//   * does Resolve load a third-party OFX plugin we built ourselves?
//   * does it expose a *second* input clip, and where (Edit / Color / Fusion)?
//   * what pixel format, bit depth and bounds does render() actually receive?
//   * how often, in what frame order, and on how many threads is render() called?
//
// It registers two effects so the two questions stay separable:
//
//   iw3 Probe (Single)  - one Source clip, inverts it.
//   iw3 Probe (Dual)    - Source + Depth clips, renders them side by side.
//
// Everything it learns is appended to %LOCALAPPDATA%\iw3probe\probe.log,
// because Resolve gives a plugin no console to print to.

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "probe_log.h"

#include <algorithm>
#include <sstream>
#include <thread>

#define kPluginGrouping "iw3"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSingleIdentifier "com.nunif.iw3.probe.single"
#define kDualIdentifier "com.nunif.iw3.probe.dual"

#define kDepthClipName "Depth"

namespace
{

const char* contextName(OFX::ContextEnum context)
{
    switch (context)
    {
        case OFX::eContextNone: return "none";
        case OFX::eContextGenerator: return "generator";
        case OFX::eContextFilter: return "filter";
        case OFX::eContextTransition: return "transition";
        case OFX::eContextPaint: return "paint";
        case OFX::eContextGeneral: return "general";
        case OFX::eContextRetimer: return "retimer";
        default: return "unknown";
    }
}

const char* bitDepthName(OFX::BitDepthEnum depth)
{
    switch (depth)
    {
        case OFX::eBitDepthNone: return "none";
        case OFX::eBitDepthUByte: return "uint8";
        case OFX::eBitDepthUShort: return "uint16";
        case OFX::eBitDepthHalf: return "half";
        case OFX::eBitDepthFloat: return "float";
        default: return "custom";
    }
}

const char* componentsName(OFX::PixelComponentEnum components)
{
    switch (components)
    {
        case OFX::ePixelComponentNone: return "none";
        case OFX::ePixelComponentRGBA: return "RGBA";
        case OFX::ePixelComponentRGB: return "RGB";
        case OFX::ePixelComponentAlpha: return "Alpha";
        default: return "custom";
    }
}

std::string threadId()
{
    std::ostringstream out;
    out << std::this_thread::get_id();
    return out.str();
}

void logImage(const char* label, OFX::Image* image)
{
    if (!image)
    {
        probe::logf("    %-6s : <null>", label);
        return;
    }
    const OfxRectI& bounds = image->getBounds();
    probe::logf("    %-6s : bounds=(%d,%d)-(%d,%d) %dx%d depth=%s comps=%s rowBytes=%d par=%.4f",
                label,
                bounds.x1, bounds.y1, bounds.x2, bounds.y2,
                bounds.x2 - bounds.x1, bounds.y2 - bounds.y1,
                bitDepthName(image->getPixelDepth()),
                componentsName(image->getPixelComponents()),
                image->getRowBytes(),
                image->getPixelAspectRatio());
}

// Copies one pixel, whatever its layout, from src to dst. Both images are
// assumed to share a pixel depth and component count, which Resolve guarantees
// for clips on the same effect unless supportsMultipleClipDepths is set.
void copyPixel(OFX::Image* dst, OFX::Image* src, int dstX, int dstY, int srcX, int srcY, int nComponents)
{
    float* out = static_cast<float*>(dst->getPixelAddress(dstX, dstY));
    const float* in = src ? static_cast<const float*>(src->getPixelAddress(srcX, srcY)) : nullptr;
    if (!out)
    {
        return;
    }
    for (int c = 0; c < nComponents; ++c)
    {
        out[c] = in ? in[c] : 0.0f;
    }
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
// The effect instance. One class serves both plugins; _hasDepth says which.

class ProbeEffect : public OFX::ImageEffect
{
public:
    ProbeEffect(OfxImageEffectHandle handle, bool hasDepth)
        : OFX::ImageEffect(handle)
        , _dstClip(nullptr)
        , _srcClip(nullptr)
        , _depthClip(nullptr)
        , _hasDepth(hasDepth)
    {
        _dstClip = fetchClip(kOfxImageEffectOutputClipName);
        _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
        if (_hasDepth)
        {
            _depthClip = fetchClip(kDepthClipName);
        }
        _divergence = fetchDoubleParam("divergence");
        _convergence = fetchDoubleParam("convergence");
        _syntheticView = fetchChoiceParam("syntheticView");
        _depthInverted = fetchBooleanParam("depthInverted");

        probe::logf("createInstance: dual=%d src=%p depth=%p", int(_hasDepth),
                    static_cast<void*>(_srcClip), static_cast<void*>(_depthClip));
        if (_depthClip)
        {
            probe::logf("    depth clip fetched OK, connected=%d", int(_depthClip->isConnected()));
        }
    }

    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments& args, OFX::Clip*& identityClip, double& identityTime) override;
    virtual void changedClip(const OFX::InstanceChangedArgs& args, const std::string& clipName) override;

private:
    OFX::Clip* _dstClip;
    OFX::Clip* _srcClip;
    OFX::Clip* _depthClip;
    OFX::DoubleParam* _divergence;
    OFX::DoubleParam* _convergence;
    OFX::ChoiceParam* _syntheticView;
    OFX::BooleanParam* _depthInverted;
    bool _hasDepth;
};

bool ProbeEffect::isIdentity(const OFX::IsIdentityArguments& /*args*/, OFX::Clip*& /*identityClip*/, double& /*identityTime*/)
{
    return false;
}

void ProbeEffect::changedClip(const OFX::InstanceChangedArgs& /*args*/, const std::string& clipName)
{
    OFX::Clip* clip = (clipName == kDepthClipName) ? _depthClip : _srcClip;
    probe::logf("changedClip: '%s' connected=%d", clipName.c_str(),
                clip ? int(clip->isConnected()) : -1);
}

void ProbeEffect::render(const OFX::RenderArguments& args)
{
    double divergence = 0.0, convergence = 0.0;
    int syntheticView = 0;
    bool depthInverted = false;
    _divergence->getValueAtTime(args.time, divergence);
    _convergence->getValueAtTime(args.time, convergence);
    _syntheticView->getValueAtTime(args.time, syntheticView);
    _depthInverted->getValueAtTime(args.time, depthInverted);

    probe::logf("render: t=%.3f window=(%d,%d)-(%d,%d) scale=%.3fx%.3f field=%d "
                "interactive=%d thread=%s | divergence=%.3f convergence=%.3f view=%d inverted=%d",
                args.time,
                args.renderWindow.x1, args.renderWindow.y1, args.renderWindow.x2, args.renderWindow.y2,
                args.renderScale.x, args.renderScale.y,
                int(args.fieldToRender), int(args.interactiveRenderStatus),
                threadId().c_str(),
                divergence, convergence, syntheticView, int(depthInverted));

    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(_srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);
    std::unique_ptr<OFX::Image> depth(_depthClip && _depthClip->isConnected() ? _depthClip->fetchImage(args.time) : nullptr);

    logImage("dst", dst.get());
    logImage("src", src.get());
    logImage("depth", depth.get());

    if (!dst)
    {
        return;
    }

    const int n = componentCount(dst->getPixelComponents());
    if (n == 0 || dst->getPixelDepth() != OFX::eBitDepthFloat)
    {
        probe::logf("    unsupported destination format, leaving black");
        return;
    }

    const OfxRectI& window = args.renderWindow;
    const int width = window.x2 - window.x1;

    for (int y = window.y1; y < window.y2; ++y)
    {
        if (abort())
        {
            probe::logf("    aborted at y=%d", y);
            return;
        }
        for (int x = window.x1; x < window.x2; ++x)
        {
            float* out = static_cast<float*>(dst->getPixelAddress(x, y));
            if (!out)
            {
                continue;
            }

            if (!_hasDepth)
            {
                // Single-input probe: invert, so "it ran" is unmistakable.
                const float* in = src ? static_cast<const float*>(src->getPixelAddress(x, y)) : nullptr;
                for (int c = 0; c < n; ++c)
                {
                    out[c] = in ? (c == 3 ? in[c] : 1.0f - in[c]) : 0.0f;
                }
                continue;
            }

            // Dual-input probe: source on the left half, depth on the right.
            // If depth never arrives the right half stays flat magenta, which
            // is impossible to mistake for a real depth map.
            const bool leftHalf = (x - window.x1) < width / 2;
            const int srcX = window.x1 + ((x - window.x1) % std::max(1, width / 2)) * 2;

            if (leftHalf)
            {
                copyPixel(dst.get(), src.get(), x, y, srcX, y, n);
            }
            else if (depth)
            {
                const float* in = static_cast<const float*>(depth->getPixelAddress(srcX, y));
                const float value = in ? (depthInverted ? 1.0f - in[0] : in[0]) : 0.0f;
                for (int c = 0; c < n; ++c)
                {
                    out[c] = (c == 3) ? 1.0f : value;
                }
            }
            else
            {
                for (int c = 0; c < n; ++c)
                {
                    out[c] = (c == 1) ? 0.0f : 1.0f;  // magenta
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Factories

static void describeCommon(OFX::ImageEffectDescriptor& desc, const char* label, const char* description)
{
    desc.setLabels(label, label, label);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(description);

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

    OFX::ImageEffectHostDescription* host = OFX::getImageEffectHostDescription();
    if (host)
    {
        probe::logf("host: name='%s' label='%s' version=%d.%d.%d ('%s') api=%d.%d",
                    host->hostName.c_str(), host->hostLabel.c_str(),
                    host->versionMajor, host->versionMinor, host->versionMicro,
                    host->versionLabel.c_str(),
                    host->APIVersionMajor, host->APIVersionMinor);
        probe::logf("host: multiRes=%d tiles=%d temporalClipAccess=%d multipleClipDepths=%d "
                    "multipleClipPARs=%d overlays=%d",
                    int(host->supportsMultiResolution), int(host->supportsTiles),
                    int(host->temporalClipAccess), int(host->supportsMultipleClipDepths),
                    int(host->supportsMultipleClipPARs), int(host->supportsOverlays));
        probe::logf("host: openCL=%d cuda=%d cudaStream=%d metal=%d",
                    int(host->supportsOpenCLRender), int(host->supportsCudaRender),
                    int(host->supportsCudaStream), int(host->supportsMetalRender));
        std::ostringstream contexts;
        for (size_t i = 0; i < host->_supportedContexts.size(); ++i)
        {
            contexts << (i ? ", " : "") << contextName(host->_supportedContexts[i]);
        }
        probe::logf("host: contexts=[%s]", contexts.str().c_str());
    }
    probe::logf("describe: '%s'", label);
}

// The parameter set is a dry run of the interface the real plugin will need,
// so Phase 3 does not discover a param type Resolve renders badly.
static void describeParams(OFX::ImageEffectDescriptor& desc)
{
    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

    OFX::DoubleParamDescriptor* divergence = desc.defineDoubleParam("divergence");
    divergence->setLabels("Divergence", "Divergence", "Divergence");
    divergence->setHint("Strength of the stereo effect, as a percentage of image width");
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

    OFX::ChoiceParamDescriptor* view = desc.defineChoiceParam("syntheticView");
    view->setLabels("Synthetic View", "Synthetic View", "Synthetic View");
    view->appendOption("both");
    view->appendOption("right");
    view->appendOption("left");
    view->setDefault(0);
    page->addChild(*view);

    OFX::BooleanParamDescriptor* inverted = desc.defineBooleanParam("depthInverted");
    inverted->setLabels("Depth Is Inverted", "Depth Is Inverted", "Depth Is Inverted");
    inverted->setHint("External depth tools disagree on whether white is near or far");
    inverted->setDefault(false);
    page->addChild(*inverted);
}

static void describeClips(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context, bool withDepth)
{
    probe::logf("describeInContext: context=%s withDepth=%d", contextName(context), int(withDepth));

    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    src->setIsMask(false);

    if (withDepth)
    {
        OFX::ClipDescriptor* depth = desc.defineClip(kDepthClipName);
        depth->addSupportedComponent(OFX::ePixelComponentRGBA);
        depth->addSupportedComponent(OFX::ePixelComponentAlpha);
        depth->setTemporalClipAccess(false);
        depth->setSupportsTiles(false);
        depth->setIsMask(false);
        // Optional so the effect still renders with nothing wired to it, which
        // is how we find out whether Resolve ever offers to wire it at all.
        depth->setOptional(true);
    }

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    describeParams(desc);
}

mDeclarePluginFactory(SingleProbeFactory, {}, {});
mDeclarePluginFactory(DualProbeFactory, {}, {});

void SingleProbeFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    describeCommon(desc, "iw3 Probe (Single)", "Phase 0 probe: one input clip, inverted.");
}

void SingleProbeFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context)
{
    describeClips(desc, context, false);
}

OFX::ImageEffect* SingleProbeFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum /*context*/)
{
    return new ProbeEffect(handle, false);
}

void DualProbeFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    describeCommon(desc, "iw3 Probe (Dual)", "Phase 0 probe: Source + Depth clips, shown side by side.");
}

void DualProbeFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context)
{
    // A second clip is only legal in the general context; in the filter context
    // OFX permits exactly one input. Whether Resolve *offers* the general
    // context, and where, is the thing being measured.
    describeClips(desc, context, context == OFX::eContextGeneral);
}

OFX::ImageEffect* DualProbeFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context)
{
    return new ProbeEffect(handle, context == OFX::eContextGeneral);
}

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids)
{
    probe::logf("---- getPluginIDs (binary loaded) ----");
    static SingleProbeFactory single(kSingleIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    static DualProbeFactory dual(kDualIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&single);
    ids.push_back(&dual);
}
