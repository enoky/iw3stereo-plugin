// The CUDA half of monobw_inpaint: the forward warp, its hole mask, and the
// mask morphology that runs before the inpaint graph. All in device memory.
//
// This is the half of monobw_inpaint that cannot be an ONNX graph. torch.export
// handles it, but ONNX has no operator for either torch.cummax or
// torch.searchsorted, and those two are not incidental to the algorithm -- the
// cummax is the monotonisation that makes the mapping invertible and the
// searchsorted is the inversion. docs/monobw-inpaint.md has the evidence.
//
// The arithmetic is not reimplemented here. monobw_math.h is compiled for the
// device, so these kernels run the same lines tests/cpp/test_pipeline.cpp
// validates on the CPU against Python that matches stock iw3 at difference 0.
// tests/cpp/test_monobw_gpu.cu then runs this against the same reference data,
// so the GPU path is covered rather than assumed.

#pragma once

#include <string>

namespace iw3
{

class MonoBwGpu
{
public:
    MonoBwGpu() = default;
    ~MonoBwGpu();

    MonoBwGpu(const MonoBwGpu&) = delete;
    MonoBwGpu& operator=(const MonoBwGpu&) = delete;

    // Allocates for this frame geometry. Cheap when nothing changed.
    bool prepare(int width, int height, int depthWidth, int depthHeight);

    // `image` is three planes of width * height, `depth` is depthWidth *
    // depthHeight, both device pointers. Results land in eyeDevice() and
    // maskDevice().
    //
    // `fixScreenBorderMask` is iw3's: 0 leaves the mask alone, 1 clears the
    // uninpaintable side, 2 clears both. It applies only when
    // preserveScreenBorder is off. The image model uses 1.
    // `mirrorOutput` writes the eye and mask mirrored horizontally. It is free
    // -- the sample kernel just writes to the opposite column -- and it is what
    // prepareEye() uses to hand the inpaint network the handedness it wants
    // without a separate pass. The mask's screen-border fix still applies in
    // warp coordinates, before the mirror.
    void forward(const float* image, const float* depth,
                 double divergence, double convergence,
                 bool preserveScreenBorder, int fixScreenBorderMask,
                 bool mirrorOutput, void* stream);

    // Everything one eye needs before the inpaint graph: the warp, the mask,
    // the morphology, and all the mirroring.
    //
    // iw3 does this with four flips whose order is easy to get wrong, so it
    // lives here rather than at the call site. The right eye is warped in
    // mirrored coordinates and inpainted in frame coordinates; the left eye is
    // warped in frame coordinates and inpainted in mirrored ones. Both end up
    // handing the network holes that open the same way, which is the only
    // handedness it was trained for.
    //
    // Leaves the eye in inpaintEyeDevice() and the mask in
    // processedMaskDevice(). Feed the graph's output back through finishEye().
    bool prepareEye(const float* image, const float* depth, bool rightEye,
                    double divergence, double convergence, bool preserveScreenBorder,
                    int innerDilation, int outerDilation, int baseWidth, void* stream);

    // Puts the graph's output back into frame orientation. Returns `filled`
    // itself for the right eye, which is already there.
    const float* finishEye(const float* filled, bool rightEye, void* stream);

    // Half-precision copies of what prepareEye() produced, for a graph that
    // wants them. The inpaint graphs do: in fp32 the twelve-frame window does
    // not fit in 17 GiB, and halving it also halves the time.
    //
    // Everything this class computes stays fp32 -- the warp's coordinate
    // arithmetic needs it -- so the conversion is a pass at the boundary rather
    // than a change of type throughout.
    void castEyeAndMaskToHalf(void* stream);
    const void* inpaintEyeHalfDevice() const { return _eyeHalf; }
    const void* processedMaskHalfDevice() const { return _maskHalf; }

    // The graph's half output back to float, into a buffer this owns, so
    // finishEye() and the compose kernel see what they expect.
    const float* halfToFloat(const void* filled, void* stream);

    const float* inpaintEyeDevice() const { return _eye; }

    // preprocess_mask on the mask forward() produced: mask_closing, then the
    // two directional dilations. Outside the ONNX graph because the counts are
    // plugin parameters -- the same rule that keeps preserve_screen_border out
    // of the warp's graph.
    //
    // `baseWidth` is the depth's width, which is what iw3 quotes the dilations
    // against. Result lands in processedMaskDevice().
    //
    // `mask` is taken explicitly rather than read from maskDevice() so the
    // dataflow is visible at the call site, and so a test can drive this with a
    // mask it supplies rather than one forward() happened to leave behind.
    void preprocessMask(const float* mask, int innerDilation, int outerDilation,
                        int baseWidth, void* stream);

    // The warped eye, three planes of width * height.
    const float* eyeDevice() const { return _eye; }
    // The raw hole mask, width * height, 0.0 or 1.0.
    const float* maskDevice() const { return _mask; }
    // The mask after preprocessMask(), which is what the inpaint graph takes.
    const float* processedMaskDevice() const { return _processedMask; }

    bool ok() const { return _error.empty(); }
    const std::string& error() const { return _error; }

private:
    bool allocate(float** pointer, size_t floats, size_t& held);

    int _width = 0, _height = 0, _depthWidth = 0, _depthHeight = 0;

    float* _gridX = nullptr;    // depthWidth * depthHeight
    float* _gridY = nullptr;    // depthHeight
    float* _scratch = nullptr;  // 3 * depthWidth * depthHeight: dest, moved, smoothed
    float* _eye = nullptr;      // 3 * width * height
    float* _mask = nullptr;     // width * height, the raw stretch mask
    float* _maskA = nullptr;    // the morphology's ping-pong pair; the finished
    float* _maskB = nullptr;    // mask ends in one of them
    const float* _processedMask = nullptr;  // whichever that was
    float* _flipImage = nullptr;  // 3 * width * height, the right eye's mirrored source
    float* _flipDepth = nullptr;  // depthWidth * depthHeight, likewise
    float* _final = nullptr;      // 3 * width * height, the left eye mirrored back
    void* _eyeHalf = nullptr;     // 3 * width * height, __half
    void* _maskHalf = nullptr;    // width * height, __half
    float* _fromHalf = nullptr;   // 3 * width * height, the graph's output widened

    size_t _gridXHeld = 0, _gridYHeld = 0, _scratchHeld = 0, _eyeHeld = 0, _maskHeld = 0;
    size_t _maskAHeld = 0, _maskBHeld = 0;
    size_t _flipImageHeld = 0, _flipDepthHeld = 0, _finalHeld = 0;
    size_t _eyeHalfHeld = 0, _maskHalfHeld = 0, _fromHalfHeld = 0;

    std::string _error;
};

// The twelve-frame window for light_video_inpaint_v1: the sequence buffers the
// graph reads, and the cache of finished frames that makes it affordable.
//
// The graph costs 234 ms a call at HD and produces twelve frames, so the whole
// economics of the temporal model rest on keeping what it produced. Phase 0
// measured Resolve rendering in order but with gaps and repeats, so the cache
// is keyed by frame number and the window each frame belongs to is derived from
// the frame number alone -- any render order hits the same window.
class MonoBwVideoGpu
{
public:
    // Baked into the checkpoint: enc2.1 and enc2.3 convolve the frame axis with
    // (12, 12, 1) weights, so the window is twelve and not a preference.
    static constexpr int kSequence = 12;
    // How many of the twelve are kept. iw3 pads three at each end and emits the
    // middle six, and the reason to copy that is continuity: consecutive output
    // frames then come from window positions 3..8, which weight their
    // neighbours similarly. Emitting all twelve would halve the cost and make
    // every twelfth frame jump from position 11 to position 0, trading
    // per-frame flicker for a periodic one.
    static constexpr int kStride = 6;
    static constexpr int kPad = (kSequence - kStride) / 2;

    MonoBwVideoGpu() = default;
    ~MonoBwVideoGpu();

    MonoBwVideoGpu(const MonoBwVideoGpu&) = delete;
    MonoBwVideoGpu& operator=(const MonoBwVideoGpu&) = delete;

    bool prepare(int width, int height);

    // Which window a frame belongs to, and which frame that window starts at.
    static long long windowIndex(long long frame);
    static long long windowFirstFrame(long long frame);

    // True when the cache already holds this window under these settings.
    bool holds(long long window, unsigned long long fingerprint) const
    {
        return _valid && _window == window && _fingerprint == fingerprint;
    }
    void markHeld(long long window, unsigned long long fingerprint)
    {
        _window = window;
        _fingerprint = fingerprint;
        _valid = true;
    }
    void invalidate() { _valid = false; }

    // Half-precision sequence buffers, laid out (frame, plane, y, x), which is
    // what the graph's (12, 3, H, W) input wants.
    void* eyesDevice() const { return _eyes; }
    void* masksDevice() const { return _masks; }

    // Copy one prepared frame into slot `index` of the sequence.
    void storeFrame(int index, const void* eyeHalf, const void* maskHalf, void* stream);

    // Keep the middle `kStride` frames of the graph's output for one eye.
    void cacheOutput(bool rightEye, const void* filledHalf, void* stream);
    // A cached frame in frame orientation, ready for compose(). `offset` is
    // 0..kStride-1 within the window.
    const float* cachedEye(bool rightEye, int offset, bool mirrorBack, void* stream);

    bool ok() const { return _error.empty(); }
    const std::string& error() const { return _error; }

private:
    bool allocate(void** pointer, size_t bytes, size_t& held);

    int _width = 0, _height = 0;
    void* _eyes = nullptr;    // kSequence * 3 * width * height, __half
    void* _masks = nullptr;   // kSequence * width * height, __half
    void* _cache[2] = {nullptr, nullptr};  // [eye] kStride * 3 * w * h, __half
    float* _scratch = nullptr;             // one frame widened, and mirrored
    float* _mirror = nullptr;

    size_t _eyesHeld = 0, _masksHeld = 0, _cacheHeld[2] = {0, 0};
    size_t _scratchHeld = 0, _mirrorHeld = 0;

    long long _window = -1;
    unsigned long long _fingerprint = 0;
    bool _valid = false;

    std::string _error;
};

}  // namespace iw3
