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

    size_t _gridXHeld = 0, _gridYHeld = 0, _scratchHeld = 0, _eyeHeld = 0, _maskHeld = 0;
    size_t _maskAHeld = 0, _maskBHeld = 0;
    size_t _flipImageHeld = 0, _flipDepthHeld = 0, _finalHeld = 0;

    std::string _error;
};

}  // namespace iw3
