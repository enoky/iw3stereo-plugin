// The separable antialiased resample, shared by every path that needs it.
//
// One definition, two callers: stereo_gpu.cu resizes the depth to the model's
// input size, and mlbw_gpu.cu resizes mask_mlbw_l2's layer weights to the
// frame. They are the same filter -- PyTorch's antialiased bilinear, which no
// ONNX Resize reproduces -- and the weights come from buildResampleAxis on the
// CPU rather than being derived here, which is what keeps the CPU and GPU
// paths identical.
//
// `static` so each translation unit gets its own instantiation without a
// duplicate symbol. The point is a single source of the arithmetic, not a
// single copy of the code.

#pragma once

#include <cuda_runtime.h>

namespace iw3
{

// The separable antialiased resample, in the same order as the CPU version:
// horizontal into scratch, then vertical.
static __global__ void resizeHorizontalKernel(const float* __restrict__ source, int sourceWidth, int height,
                                       const int* __restrict__ start, const int* __restrict__ count,
                                       const int* __restrict__ offset, const float* __restrict__ weights,
                                       int targetWidth, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= targetWidth || y >= height) return;

    const float* row = source + size_t(y) * size_t(sourceWidth);
    const int first = start[x];
    const int taps = count[x];
    const float* w = weights + offset[x];
    float sum = 0.0f;
    for (int k = 0; k < taps; ++k)
    {
        sum += row[first + k] * w[k];
    }
    out[size_t(y) * size_t(targetWidth) + size_t(x)] = sum;
}

static __global__ void resizeVerticalKernel(const float* __restrict__ scratch, int targetWidth, int sourceHeight,
                                     const int* __restrict__ start, const int* __restrict__ count,
                                     const int* __restrict__ offset, const float* __restrict__ weights,
                                     int targetHeight, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= targetWidth || y >= targetHeight) return;

    const int first = start[y];
    const int taps = count[y];
    const float* w = weights + offset[y];
    float sum = 0.0f;
    for (int k = 0; k < taps; ++k)
    {
        sum += scratch[size_t(first + k) * size_t(targetWidth) + size_t(x)] * w[k];
    }
    out[size_t(y) * size_t(targetWidth) + size_t(x)] = fminf(fmaxf(sum, 0.0f), 1.0f);
}

}  // namespace iw3
