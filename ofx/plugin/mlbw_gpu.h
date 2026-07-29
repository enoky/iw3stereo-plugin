// The CUDA half of mlbw_l2_inpaint: the two backward warps, the softmax blend,
// and the mask post-process that runs before the inpaint graph. All in device
// memory.
//
// This is the half of mlbw_l2_inpaint that cannot be an ONNX graph -- and
// unlike monobw, that is not because an operator is missing. Every operation
// here has an ONNX equivalent. One of them does not have the *same* equivalent:
// iw3 resizes the layer weights with PyTorch's antialiased kernel, which
// ignores align_corners and uses a half-pixel centre regardless, and ONNX
// Runtime honours align_corners. They disagree by 3.6e-2 on a finished eye.
// docs/mlbw-inpaint-plan.md has the numbers; the same wall was already hit for
// the depth resize, and the answer is the same one: buildResampleAxis on the
// CPU, resample_gpu.cuh on the device.
//
// The arithmetic is not reimplemented here. mlbw_math.h is compiled for the
// device, so these kernels run the same lines tests/cpp/test_pipeline.cpp
// validates on the CPU against Python that matches stock iw3 at difference 0.
// tests/cpp/test_mlbw_gpu.cu then runs this against the same reference data.

#pragma once

#include <string>
#include <vector>

namespace iw3
{

class MlbwGpu
{
public:
    MlbwGpu() = default;
    ~MlbwGpu();

    MlbwGpu(const MlbwGpu&) = delete;
    MlbwGpu& operator=(const MlbwGpu&) = delete;

    // Allocates for this frame geometry and rebuilds the resample weights.
    // Cheap when nothing changed.
    bool prepare(int width, int height, int depthWidth, int depthHeight);

    // Everything one eye needs between the network and the inpaint graph.
    //
    // `image` is the frame in frame orientation, three planes. `delta`,
    // `layerWeight` and `maskLogits` are the network's three outputs at the
    // depth's resolution, in whatever orientation the network saw -- that is,
    // mirrored for the right eye, because iw3 flips the depth before building
    // the model's input.
    //
    // Both eyes come out of here in the orientation the inpaint network wants,
    // which is the same orientation for both. Working that through is the part
    // of this path most likely to be wrong and least likely to be caught by
    // anything else, so it is written down rather than left to the call site:
    //
    //   left  -- the model saw the frame; apply_divergence returns the eye in
    //            frame coordinates; _inpaint_single then flips it.
    //   right -- the model saw the mirror; apply_divergence flips the eye back
    //            to frame coordinates; _inpaint_single leaves it alone.
    //
    // Both therefore hand the network the *mirror of the warp's own
    // coordinates*, so the warp writes mirrored in both cases and the logits
    // are flipped in both cases. The only per-eye difference left is which
    // image the warp samples: the frame for the left eye, its mirror for the
    // right.
    //
    // Leaves the eye in inpaintEyeDevice() and the mask in
    // processedMaskDevice(). Feed the graph's output back through finishEye().
    bool prepareEye(const float* image, const float* delta, const float* layerWeight,
                    const float* maskLogits, bool rightEye,
                    int innerDilation, int outerDilation, void* stream);

    // What the inpaint graph is fed: half precision, and optionally at a
    // reduced resolution. Identical in every respect to monobw's -- the same
    // kernels, from inpaint_boundary.cuh -- because neither the cast nor the
    // reduce-and-composite has anything to do with which warp opened the holes,
    // and both pipelines feed the same two LightInpaint graphs.
    //
    // `maxWidth` of 0, or anything at least the frame's width, leaves the
    // resolution alone.
    void prepareInpaintInput(int maxWidth, void* stream);
    int inpaintWidth() const { return _workWidth; }
    int inpaintHeight() const { return _workHeight; }
    const void* inpaintEyeHalfDevice() const { return _eyeHalf; }
    const void* processedMaskHalfDevice() const { return _maskHalf; }

    // The graph's output back to a full-resolution float frame: a widening at
    // full resolution, or a widening plus an upscale plus a feathered composite
    // against the full-resolution eye when it ran reduced.
    const float* finishInpaintOutput(const void* filled, void* stream);

    // Puts the graph's output back into frame orientation. Returns `filled`
    // itself for the right eye, which is already there.
    const float* finishEye(const float* filled, bool rightEye, void* stream);

    // The blended eye, three planes of width * height, in the orientation the
    // inpaint network wants.
    const float* inpaintEyeDevice() const { return _eye; }
    // The finished binary mask, width * height, 0.0 or 1.0, same orientation.
    const float* processedMaskDevice() const { return _mask; }

    bool ok() const { return _error.empty(); }
    const std::string& error() const { return _error; }

private:
    bool allocate(float** pointer, size_t floats, size_t& held);
    bool uploadInts(int** pointer, const std::vector<int>& values);
    void buildWeightAxes();

    int _width = 0, _height = 0, _depthWidth = 0, _depthHeight = 0;

    float* _flipImage = nullptr;   // 3 * width * height, the right eye's source
    float* _gridX = nullptr;       // kMlbwLayers * depthWidth * depthHeight
    float* _gridY = nullptr;       // depthWidth * depthHeight, row-constant
    float* _weights = nullptr;     // kMlbwLayers * width * height, resized
    float* _weightScratch = nullptr;  // kMlbwLayers * width * depthHeight
    float* _eye = nullptr;         // 3 * width * height
    float* _logits = nullptr;      // depthWidth * depthHeight, flipped
    float* _closed = nullptr;      // depthWidth * depthHeight, and its scratch
    float* _closedTmp = nullptr;
    float* _mask = nullptr;        // width * height
    float* _final = nullptr;       // 3 * width * height, the left eye flipped back
    void* _eyeHalf = nullptr;      // 3 * workWidth * workHeight, __half
    void* _maskHalf = nullptr;     // workWidth * workHeight, __half
    float* _fromHalf = nullptr;    // 3 * width * height, the graph's output widened
    float* _maskBlur = nullptr;    // width * height, the feather for the composite
    float* _maskBlurTmp = nullptr;
    int _workWidth = 0, _workHeight = 0;

    // The resample tables, exactly as the CPU builds them. Horizontal entries
    // first, then vertical, which is the packing resizeDepth already uses.
    float* _axisWeights = nullptr;
    int* _axisStart = nullptr;
    int* _axisCount = nullptr;
    int* _axisOffset = nullptr;
    bool _axesValid = false;

    size_t _flipImageHeld = 0, _gridXHeld = 0, _gridYHeld = 0;
    size_t _weightsHeld = 0, _weightScratchHeld = 0, _eyeHeld = 0;
    size_t _logitsHeld = 0, _closedHeld = 0, _closedTmpHeld = 0;
    size_t _maskHeld = 0, _finalHeld = 0;
    size_t _eyeHalfHeld = 0, _maskHalfHeld = 0, _fromHalfHeld = 0;
    size_t _maskBlurHeld = 0, _maskBlurTmpHeld = 0;
    size_t _axisWeightsHeld = 0, _axisIntsHeld = 0;

    std::string _error;
};

}  // namespace iw3
