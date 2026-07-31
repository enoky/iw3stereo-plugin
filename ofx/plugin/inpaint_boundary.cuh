// The boundary either side of an inpaint graph: float32 at the frame's
// resolution on this side, half precision at the graph's working resolution on
// the other.
//
// One definition, two callers -- monobw_gpu.cu and mlbw_gpu.cu. Neither the
// casts nor the reduce-and-composite has anything to do with which warp opened
// the holes, and both pipelines feed the same two LightInpaint graphs, so a
// second copy would only be a second place for the composite to drift.
//
// Half precision is not an optimisation for the temporal graph: in fp32 the
// twelve-frame window exhausts 17 GiB at HD. The reduced working resolution is
// the only lever that brings it onto a smaller card, and it applies to the
// invented pixels alone -- everything outside a hole keeps its own detail,
// which is what compositeUpscaledKernel is for.
//
// `static` so each translation unit gets its own instantiation without a
// duplicate symbol. The point is a single source of the arithmetic.

#pragma once

#include "monobw_math.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace iw3
{


static __global__ void toHalfKernel(const float* __restrict__ source, size_t count,
                             __half* __restrict__ destination)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    destination[i] = __float2half(source[i]);
}

// Area-average downscale of the eye, straight to half.
//
// An average over the source footprint rather than a bilinear sample, because
// this is a reduction and point sampling one would alias -- and aliasing in the
// eye is aliasing in what the network is asked to continue.
// The eye, reduced to the graph's working size, weighted by which pixels are
// valid.
//
// A plain box average has no idea which of the pixels it is averaging sit inside
// a hole, so invalid content bleeds into the valid pixels next to every hole
// edge and the network is handed that as though it were real. Premultiplying by
// validity and dividing by the coverage is the standard fix.
//
// Note what "invalid" means here, because it is not what the upstream patch
// this came from describes. iw3 leaves holes black and gets dark halos; both
// warps here leave *smeared* content -- monobw stretches an edge across the hole
// and the backward warps clamp at the border -- so the artefact is edge content
// bleeding inwards rather than a halo. Same cause, same fix, less dramatic.
//
// It matters in proportion to the downscale ratio: not at all at Full, where
// this kernel does not run, and most at 720 on a 4K timeline where one output
// pixel averages a 5x5 footprint that may be half hole.
static __global__ void downscaleEyeKernel(const float* __restrict__ source, int width, int height,
                                   const float* __restrict__ mask,
                                   int outWidth, int outHeight, __half* __restrict__ destination)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outWidth || y >= outHeight) return;

    const int x0 = int((long long)x * width / outWidth);
    const int x1 = max(x0 + 1, int((long long)(x + 1) * width / outWidth));
    const int y0 = int((long long)y * height / outHeight);
    const int y1 = max(y0 + 1, int((long long)(y + 1) * height / outHeight));

    const size_t pixels = size_t(width) * size_t(height);
    const size_t outPixels = size_t(outWidth) * size_t(outHeight);
    const float count = float((x1 - x0) * (y1 - y0));

    // Coverage is a property of the footprint, not of a plane, so it is
    // accumulated once rather than three times.
    float coverage = 0.0f;
    for (int sy = y0; sy < y1; ++sy)
    {
        for (int sx = x0; sx < x1; ++sx)
        {
            coverage += 1.0f - mask[size_t(sy) * size_t(width) + size_t(sx)];
        }
    }
    // Entirely inside a hole: no valid content to average, so fall back to the
    // plain mean. That pixel is about to be replaced by the network; what
    // matters is that it is finite rather than a division by zero.
    const bool anyValid = coverage > 0.0f;
    const float divisor = anyValid ? coverage : count;

    for (int plane = 0; plane < 3; ++plane)
    {
        float total = 0.0f;
        for (int sy = y0; sy < y1; ++sy)
        {
            for (int sx = x0; sx < x1; ++sx)
            {
                const size_t index = size_t(sy) * size_t(width) + size_t(sx);
                const float valid = anyValid ? (1.0f - mask[index]) : 1.0f;
                total += source[size_t(plane) * pixels + index] * valid;
            }
        }
        destination[size_t(plane) * outPixels + size_t(y) * size_t(outWidth) + size_t(x)] =
            __float2half(total / divisor);
    }
}

// The mask reduces by a maximum, not an average. A hole that survives in any
// source pixel of the footprint has to survive in the reduced mask, or the
// network is never told to fill it.
static __global__ void downscaleMaskKernel(const float* __restrict__ source, int width, int height,
                                    int outWidth, int outHeight, __half* __restrict__ destination)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outWidth || y >= outHeight) return;

    const int x0 = int((long long)x * width / outWidth);
    const int x1 = max(x0 + 1, int((long long)(x + 1) * width / outWidth));
    const int y0 = int((long long)y * height / outHeight);
    const int y1 = max(y0 + 1, int((long long)(y + 1) * height / outHeight));

    float best = 0.0f;
    for (int sy = y0; sy < y1; ++sy)
    {
        for (int sx = x0; sx < x1; ++sx)
        {
            const float value = source[size_t(sy) * size_t(width) + size_t(sx)];
            best = value > best ? value : best;
        }
    }
    destination[size_t(y) * size_t(outWidth) + size_t(x)] = __float2half(best);
}

// The separable 15-tap feather, the same one LightInpaintV1 applies to its own
// mask. Two passes, replicate-padded, matching SeparableGaussianFilter2d.
static __global__ void maskBlurKernel(const float* __restrict__ source, int width, int height,
                               int horizontal, float* __restrict__ destination)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float kernel[math::kMaskBlurKernel];
    math::gaussianKernel(kernel, math::kMaskBlurKernel);

    float total = 0.0f;
    for (int k = 0; k < math::kMaskBlurKernel; ++k)
    {
        const int offset = k - math::kMaskBlurRadius;
        const int sx = horizontal ? math::clampInt(x + offset, 0, width - 1) : x;
        const int sy = horizontal ? y : math::clampInt(y + offset, 0, height - 1);
        total += source[size_t(sy) * size_t(width) + size_t(sx)] * kernel[k];
    }
    destination[size_t(y) * size_t(width) + size_t(x)] = total;
}

// Upscale the graph's reduced output and composite it into the full-resolution
// eye by the feathered mask.
//
// LightInpaintV1's own composite is src * (1 - m) + x * m with m the blurred
// mask, and this is that same expression at full resolution: where the mask is
// zero the original eye survives untouched, so only the invented pixels are the
// ones that were computed small.
static __global__ void compositeUpscaledKernel(const float* __restrict__ eye,
                                        const __half* __restrict__ filled,
                                        const float* __restrict__ maskBlur,
                                        const float* __restrict__ maskHard,
                                        int width, int height, int inWidth, int inHeight,
                                        float* __restrict__ destination)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t index = size_t(y) * size_t(width) + size_t(x);

    float m = maskBlur[index] + maskHard[index];
    m = m < 0.0f ? 0.0f : (m > 1.0f ? 1.0f : m);

    if (m <= 0.0f)
    {
        for (int plane = 0; plane < 3; ++plane)
        {
            destination[size_t(plane) * pixels + index] = eye[size_t(plane) * pixels + index];
        }
        return;
    }

    const float scaleX = inWidth > 1 ? float(inWidth - 1) / float(max(width - 1, 1)) : 0.0f;
    const float scaleY = inHeight > 1 ? float(inHeight - 1) / float(max(height - 1, 1)) : 0.0f;
    const float sx = float(x) * scaleX;
    const float sy = float(y) * scaleY;
    const int x0 = math::clampInt(int(sx), 0, inWidth - 1);
    const int y0 = math::clampInt(int(sy), 0, inHeight - 1);
    const int x1 = math::clampInt(x0 + 1, 0, inWidth - 1);
    const int y1 = math::clampInt(y0 + 1, 0, inHeight - 1);
    const float fx = sx - float(x0);
    const float fy = sy - float(y0);

    const size_t inPixels = size_t(inWidth) * size_t(inHeight);
    for (int plane = 0; plane < 3; ++plane)
    {
        const __half* p = filled + size_t(plane) * inPixels;
        const float v00 = __half2float(p[size_t(y0) * size_t(inWidth) + size_t(x0)]);
        const float v01 = __half2float(p[size_t(y0) * size_t(inWidth) + size_t(x1)]);
        const float v10 = __half2float(p[size_t(y1) * size_t(inWidth) + size_t(x0)]);
        const float v11 = __half2float(p[size_t(y1) * size_t(inWidth) + size_t(x1)]);
        const float top = v00 + (v01 - v00) * fx;
        const float bottom = v10 + (v11 - v10) * fx;
        const float value = top + (bottom - top) * fy;
        const float original = eye[size_t(plane) * pixels + index];
        destination[size_t(plane) * pixels + index] = original * (1.0f - m) + value * m;
    }
}

static __global__ void fromHalfKernel(const __half* __restrict__ source, size_t count,
                               float* __restrict__ destination)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    destination[i] = __half2float(source[i]);
}

}  // namespace iw3
