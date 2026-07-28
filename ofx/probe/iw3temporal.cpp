// Does Resolve actually give an OFX plugin frames other than the current one?
//
// The host advertises temporalClipAccess=1, and this project has been wrong
// before about taking an advertised capability at its word -- Resolve ships an
// onnxruntime.dll that would have been silently bound to, and DirectML claims
// to run row_flow_v2 and miscomputes it by whole units. So this asks the
// question the only way that settles it: fetch the source at other times and
// check that what comes back is a *different picture*.
//
// It matters because light_video_inpaint_v1 is the fix for the image inpaint's
// temporal flicker, and it needs exactly twelve frames -- the count is baked
// into its weights as a (12, 12, 1) convolution over the frame axis. iw3 feeds
// it through a stateful queue that assumes frames arrive consecutively, which
// Phase 0 measured that Resolve does not do (in order, but with gaps and
// repeats). A stateless window that asks for t-5..t+6 explicitly is the only
// shape that fits, and it only exists if this probe says yes.
//
// Three things have to hold, and the third is the one that a host can quietly
// fail:
//   1. fetchImage(t + n) returns an image rather than null
//   2. the images differ from each other, i.e. they really are other frames
//   3. it still holds during playback and render, not just a single scrub
//
// Deliberately does not opt into CUDA render, so getPixelData() is host memory
// and can be checksummed without a device copy. The temporal question is
// independent of where the pixels live.
//
// Add "iw3 Temporal Probe" to a clip with *moving* content -- on a freeze frame
// every offset is legitimately identical and the probe cannot tell you anything.

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "probe_log.h"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define kPluginGrouping "iw3"
#define kIdentifier "com.nunif.iw3.temporalprobe"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

namespace
{

// The offsets a twelve-frame window centred on t would need, plus the
// immediate neighbours, which are the ones most likely to work if anything does.
const int kOffsets[] = {-6, -5, -3, -1, 0, 1, 3, 6};

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

// A cheap content fingerprint: a few dozen samples spread over the frame.
//
// Not a hash -- the point is only to tell one frame from another, and a sum of
// widely spaced pixels does that on any real footage while costing nothing.
double fingerprint(OFX::Image* image)
{
    if (!image || image->getPixelDepth() != OFX::eBitDepthFloat)
    {
        return 0.0;
    }
    const OfxRectI bounds = image->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    const int components = componentCount(image->getPixelComponents());
    if (width <= 0 || height <= 0 || components < 3)
    {
        return 0.0;
    }

    const char* base = static_cast<const char*>(image->getPixelData());
    const size_t rowBytes = size_t(image->getRowBytes());
    double sum = 0.0;
    const int steps = 8;
    for (int j = 0; j < steps; ++j)
    {
        const int y = (height * (2 * j + 1)) / (2 * steps);
        const float* row = reinterpret_cast<const float*>(base + size_t(y) * rowBytes);
        for (int i = 0; i < steps; ++i)
        {
            const int x = (width * (2 * i + 1)) / (2 * steps);
            const float* pixel = row + size_t(x) * size_t(components);
            // Weighted so a channel swap does not cancel out.
            sum += double(pixel[0]) + 2.0 * double(pixel[1]) + 3.0 * double(pixel[2]);
        }
    }
    return sum;
}

class TemporalProbeEffect : public OFX::ImageEffect
{
public:
    explicit TemporalProbeEffect(OfxImageEffectHandle handle)
        : OFX::ImageEffect(handle)
    {
        _dstClip = fetchClip(kOfxImageEffectOutputClipName);
        _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    }

    virtual void render(const OFX::RenderArguments& args) override;

    // Declaring the range up front is the documented way to ask a host for
    // other frames. A host that honours only this and not ad-hoc fetches would
    // still be usable, so both are exercised: this declares, render() fetches.
    virtual void getFramesNeeded(const OFX::FramesNeededArguments& args,
                                 OFX::FramesNeededSetter& frames) override
    {
        OfxRangeD range;
        range.min = args.time + kOffsets[0];
        range.max = args.time + kOffsets[sizeof(kOffsets) / sizeof(kOffsets[0]) - 1];
        if (_srcClip)
        {
            frames.setFramesNeeded(*_srcClip, range);
        }
        if (_framesNeededCalls++ == 0)
        {
            probe::logf("getFramesNeeded: CALLED, t=%.3f range=[%.3f, %.3f]",
                        args.time, range.min, range.max);
        }
    }

private:
    OFX::Clip* _dstClip = nullptr;
    OFX::Clip* _srcClip = nullptr;
    int _framesNeededCalls = 0;
    int _renders = 0;
};

void TemporalProbeEffect::render(const OFX::RenderArguments& args)
{
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
    if (!dst)
    {
        return;
    }

    // Only the first few renders are logged in detail. Playback would otherwise
    // produce thousands of lines and bury the answer.
    const bool verbose = _renders < 8;
    ++_renders;

    std::ostringstream summary;
    double baseline = 0.0;
    int fetched = 0;
    int distinct = 0;

    for (int offset : kOffsets)
    {
        const double time = args.time + double(offset);
        std::unique_ptr<OFX::Image> image;
        try
        {
            image.reset(_srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(time) : nullptr);
        }
        catch (const std::exception& error)
        {
            if (verbose)
            {
                probe::logf("    t%+d: THREW %s", offset, error.what());
            }
            continue;
        }
        catch (...)
        {
            if (verbose)
            {
                probe::logf("    t%+d: THREW (unknown)", offset);
            }
            continue;
        }

        if (!image)
        {
            summary << " " << offset << ":null";
            if (verbose)
            {
                probe::logf("    t%+d: null", offset);
            }
            continue;
        }

        ++fetched;
        const double print = fingerprint(image.get());
        if (offset == 0)
        {
            baseline = print;
        }
        else if (std::abs(print - baseline) > 1e-6)
        {
            ++distinct;
        }
        const OfxRectI bounds = image->getBounds();
        summary << " " << offset << ":" << (offset != 0 && std::abs(print - baseline) > 1e-6 ? "diff" : "same");
        if (verbose)
        {
            probe::logf("    t%+d: ok bounds=(%d,%d)-(%d,%d) fingerprint=%.6f%s",
                        offset, bounds.x1, bounds.y1, bounds.x2, bounds.y2, print,
                        offset == 0 ? "  <- baseline" : "");
        }
    }

    if (verbose)
    {
        const int wanted = int(sizeof(kOffsets) / sizeof(kOffsets[0]));
        probe::logf("render t=%.3f: fetched %d/%d, %d differ from t+0 --%s",
                    args.time, fetched, wanted, distinct, summary.str().c_str());
        if (_renders == 1)
        {
            probe::logf("  VERDICT so far: %s",
                        distinct >= 2 ? "temporal access WORKS (other frames really are other frames)"
                      : fetched > 1 ? "fetch succeeds but every frame is IDENTICAL -- either a freeze "
                                      "frame, or the host is returning the current frame for every time"
                                    : "temporal access UNAVAILABLE");
        }
    }

    // Pass the source through so the node is visible in the viewer.
    std::unique_ptr<OFX::Image> src(
        _srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);
    const int components = componentCount(dst->getPixelComponents());
    if (dst->getPixelDepth() != OFX::eBitDepthFloat || components < 3)
    {
        return;
    }
    const OfxRectI dstBounds = dst->getBounds();
    for (int y = args.renderWindow.y1; y < args.renderWindow.y2; ++y)
    {
        if (y < dstBounds.y1 || y >= dstBounds.y2) continue;
        float* out = reinterpret_cast<float*>(
            static_cast<char*>(dst->getPixelData()) +
            size_t(y - dstBounds.y1) * size_t(dst->getRowBytes()));
        const float* in = nullptr;
        if (src)
        {
            const OfxRectI srcBounds = src->getBounds();
            if (y >= srcBounds.y1 && y < srcBounds.y2)
            {
                in = reinterpret_cast<const float*>(
                    static_cast<const char*>(src->getPixelData()) +
                    size_t(y - srcBounds.y1) * size_t(src->getRowBytes()));
            }
        }
        for (int x = args.renderWindow.x1; x < args.renderWindow.x2; ++x)
        {
            if (x < dstBounds.x1 || x >= dstBounds.x2) continue;
            float* pixel = out + size_t(x - dstBounds.x1) * size_t(components);
            for (int c = 0; c < components; ++c)
            {
                pixel[c] = in ? in[size_t(x - dstBounds.x1) * size_t(components) + c]
                              : (c == 3 ? 1.0f : 0.0f);
            }
        }
    }
}

mDeclarePluginFactory(TemporalProbeFactory, {}, {});

void TemporalProbeFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    desc.setLabels("iw3 Temporal Probe", "iw3 Temporal Probe", "iw3 Temporal Probe");
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(
        "Asks whether Resolve will hand an OFX plugin frames other than the current one. "
        "Passes the source through; the answer goes to the probe log.");

    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    // The whole point.
    desc.setTemporalClipAccess(true);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);

    OFX::ImageEffectHostDescription* host = OFX::getImageEffectHostDescription();
    probe::logf("---- iw3 Temporal Probe: describe ----");
    if (host)
    {
        probe::logf("host advertises temporalClipAccess=%d", int(host->temporalClipAccess));
    }
}

void TemporalProbeFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentRGB);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");
    (void)page;
}

OFX::ImageEffect* TemporalProbeFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new TemporalProbeEffect(handle);
}

}  // namespace

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static TemporalProbeFactory factory(kIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&factory);
}
