// mask_mlbw_l2's geometry and mask post-process, shared verbatim between the
// CPU path and the CUDA kernels -- the same discipline as numeric_math.h and
// monobw_math.h, and for the same reason: tests/cpp/test_pipeline.cpp can only
// run this on the CPU, and having the GPU compile the same lines is what
// extends that coverage to the GPU path.
//
// This is the half of mlbw_l2_inpaint that is *not* the network. The split is
// not the one docs/mlbw-inpaint-plan.md originally drew: the plan had all of
// this inside the ONNX graph, since -- unlike MonoBW, which needs cummax and
// searchsorted -- every operation here does have an ONNX equivalent. One of
// them does not have the *same* equivalent.
//
// iw3 resizes the layer weights with align_corners=True, antialias=True, and
// PyTorch's antialiased kernel ignores align_corners and uses a half-pixel
// centre regardless. ONNX Runtime honours align_corners. The two disagree by
// 3.6e-2 on a finished eye, and no Resize configuration reconciles them,
// because the disagreement is in the coordinate transform rather than in the
// filter. The same wall was already hit for the depth resize -- see
// buildResampleAxis in stereo_pipeline.h and docs/phase2-onnx.md -- and the
// answer is the same: build the weights on the CPU and resample here.
//
// So the graph is the network alone, and what remains is:
//
//   1. resize the layer weights to the frame     -> buildResampleAxis, reused
//   2. one backward warp per layer               -> mlbwWarpSampleAt
//   3. blend them by the softmax weights         -> mlbwBlendAt
//   4. turn the mask logits into a binary mask   -> mlbwMask*
//
// Steps 2 and 3 are per output pixel and independent; step 4 is per pixel with
// a small neighbourhood. All of it is one thread per pixel on the GPU.

#pragma once

#include "monobw_math.h"

namespace iw3
{
namespace math
{

// MASK_MLBW_THRESHOLD. The head is calibrated to fire well below an even split,
// so this is not 0.5, and moving it towards 0.5 loses holes.
constexpr float kMaskMlbwThreshold = 0.15f;

// The model emits one x-shift and one softmax weight per layer. Two, for
// mask_mlbw_l2 -- baked into the checkpoint, not a setting.
constexpr int kMlbwLayers = 2;

// One layer's sampling grid at the *depth's* resolution, before it is lifted to
// the frame.
//
// iw3 adds the delta to the grid at the depth's size and resizes the sum, not
// the other way round. That ordering is load-bearing: resizing a grid and then
// adding a delta sampled at a different resolution is a different picture.
//
// delta_y is always zero -- pad_delta_y interleaves it and the model never
// predicts one -- so the y grid is just the ramp, and it is still resized
// through the same path so the two axes round identically.
IW3_HD inline float mlbwGridXAt(const float* deltaX, int width, int height,
                                int x, int y, float deltaScale)
{
    const float ramp = linspaceAt(-1.0f, 1.0f, width, x);
    return ramp + deltaX[size_t(y) * size_t(width) + size_t(x)] * deltaScale;
}

IW3_HD inline float mlbwGridYAt(int height, int y)
{
    return linspaceAt(-1.0f, 1.0f, height, y);
}

// One layer's contribution at one output pixel of one channel.
//
// `gridX`/`gridY` are the grid planes built above, still at the depth's size;
// they are lifted to the frame here with the ordinary align_corners bilinear
// resize -- *no* antialias. That asymmetry against the weight resize is iw3's
// and is deliberate on its part: a weight map is an image being resampled and a
// grid is a coordinate field being resampled, and they are not the same
// operation.
//
// The clamp is backward_warp's own, applied per layer before the blend rather
// than once at the end. Clamping after the sum instead would be a different
// answer wherever a layer overshoots.
IW3_HD inline float mlbwWarpSampleAt(const float* plane, int imageWidth, int imageHeight,
                                     const float* gridX, const float* gridY,
                                     int depthWidth, int depthHeight,
                                     int x, int y)
{
    const float gx = bilinearResizeAt(gridX, depthWidth, depthHeight,
                                      imageWidth, imageHeight, x, y);
    const float gy = bilinearResizeAt(gridY, depthWidth, depthHeight,
                                      imageWidth, imageHeight, x, y);
    const float value = bilinearSampleBorder(plane, imageWidth, imageHeight, gx, gy);
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

// The finished pixel: the weighted sum of the layers, clamped.
//
// `weights` is the layer weight already resized to the frame, one plane per
// layer. It is a softmax, so the weights sum to one and the blend is a convex
// combination of values already in 0..1 -- the outer clamp cannot fire on exact
// arithmetic. It is iw3's, and it stays, because in float the sum of two
// weights is not exactly one.
IW3_HD inline float mlbwBlendAt(const float* layerValues, const float* weights,
                                size_t weightStride, size_t index, int layers)
{
    float sum = 0.0f;
    for (int layer = 0; layer < layers; ++layer)
    {
        sum += layerValues[layer] * weights[size_t(layer) * weightStride + index];
    }
    return sum < 0.0f ? 0.0f : (sum > 1.0f ? 1.0f : sum);
}

// --- the mask post-process ---------------------------------------------------
//
// postprocess_hole_mask, which is NOT MonoBWInpaintImage::preprocess_mask with
// different constants. Three things differ and each changes the answer:
//
//   1. the closing runs on the LOGITS, before any sigmoid, so it is a greyscale
//      morphological closing rather than a binary one;
//   2. the upscale to the frame is bilinear rather than nearest, and happens
//      while the value is still continuous;
//   3. the threshold is a sigmoid against 0.15, not "> 0".
//
// A fourth difference is only apparent: iw3 dilates inner then outer, the
// opposite order to the monobw path. Dilation by opposite unit structuring
// elements commutes -- measured over 3200 cases including ones saturating
// against the frame edge, the disagreement is zero -- so the single-window form
// below is exact for either order, as it already is for monobw.
//
// Step 1 uses maskDilateAt/maskErodeAt from monobw_math.h unchanged. They are
// 3x3 max and min that skip out-of-range neighbours, which is what max_pool2d's
// -inf padding does, and closing(n_iter=1) is one of each in that order.

// Step 3, at one pixel of the already-resized logit plane.
//
// sigmoid(v) > 0.15 rearranges to v > logit(0.15) = -1.7346..., and comparing
// that way would avoid an expf per pixel. It is not done: the comparison has to
// land on the same side as PyTorch's for a value sitting on the boundary, and
// the rearrangement moves the boundary by an ulp or two. At HD roughly one
// pixel in two million sits that close -- measured -- and each one that flips
// gets filled, so it is a visible pixel rather than a rounding difference.
IW3_HD inline bool mlbwMaskThreshold(float logit)
{
    return (1.0f / (1.0f + expf(-logit))) > kMaskMlbwThreshold;
}

// Steps 3 and 4 together, at one pixel.
//
// `logits` is the closed logit plane at the *depth's* size; it is resized here
// with the ordinary bilinear (align_corners=True, antialias=False) that iw3
// uses for the mask, then thresholded, then dilated over a single window.
//
// One window rather than two passes, for the reason maskFinishAt gives: or is
// idempotent and associative, so oring over [x - outer, x + inner] is the same
// answer as dilating left by `outer` and right by `inner` in either order.
//
// The dilation counts arrive already scaled -- see maskDilateIterations -- but
// note the base width is the *logits'*, not the depth's, because iw3 reads it
// off the tensor it is handed. Those are the same number here and for different
// reasons, so do not let the two drift.
IW3_HD inline float mlbwMaskAt(const float* logits, int depthWidth, int depthHeight,
                               int width, int height, int x, int y,
                               int outer, int inner)
{
    const int first = x - outer < 0 ? 0 : x - outer;
    const int last = x + inner >= width ? width - 1 : x + inner;
    for (int sx = first; sx <= last; ++sx)
    {
        const float logit = bilinearResizeAt(logits, depthWidth, depthHeight,
                                             width, height, sx, y);
        if (mlbwMaskThreshold(logit))
        {
            return 1.0f;
        }
    }
    return 0.0f;
}

}  // namespace math
}  // namespace iw3
