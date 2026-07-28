#include "monobw_gpu.h"

#include "monobw_math.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace iw3
{

namespace
{

constexpr int kBlock = 16;
constexpr int kRowBlock = 64;

dim3 grid2d(int width, int height)
{
    return dim3((unsigned(width) + kBlock - 1) / kBlock, (unsigned(height) + kBlock - 1) / kBlock);
}

// Stage 1, one thread per depth row.
//
// This is the only sequential part -- a running maximum along the row -- and it
// cannot be parallelised without becoming a different algorithm, which would be
// a second transcription no test covers. It is also cheap: one max and one
// compare per element.
//
// It is launched on its own rather than fused with the two stages below because
// fusing them costs more than the scan does. Measured at HD with everything in
// one thread per row: 1.85 ms of a 1.92 ms total, scaling with row length
// rather than with pixels, because a few hundred threads cannot hide a
// thousand dependent steps.
__global__ void destIndexKernel(const float* __restrict__ depth, int depthWidth, int depthHeight,
                                float shiftPx, float convergence, int borderPix,
                                float* __restrict__ dest, float* __restrict__ moved)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= depthHeight) return;

    const size_t offset = size_t(row) * size_t(depthWidth);
    math::monobwDestIndexRow(depth + offset, depthWidth, shiftPx, convergence, borderPix,
                             dest + offset, moved + offset);
}

// Stages 2 and 3, one thread per element. Both read only the previous stage's
// output, so they parallelise completely; the binary search in stage 3 is ten
// dependent loads rather than a scan of the whole row.
__global__ void smoothKernel2d(const float* __restrict__ dest, const float* __restrict__ moved,
                               int depthWidth, int depthHeight, float* __restrict__ smoothed)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= depthWidth || y >= depthHeight) return;

    const size_t offset = size_t(y) * size_t(depthWidth);
    smoothed[offset + size_t(x)] = math::monobwSmoothAt(dest + offset, moved + offset,
                                                        depthWidth, x);
}

__global__ void invertKernel(const float* __restrict__ smoothed,
                             int depthWidth, int depthHeight, float* __restrict__ gridX)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= depthWidth || y >= depthHeight) return;

    const size_t offset = size_t(y) * size_t(depthWidth);
    gridX[offset + size_t(x)] = math::monobwInvertAt(smoothed + offset, depthWidth, x);
}

__global__ void gridYKernel(int depthHeight, float* __restrict__ gridY)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= depthHeight) return;
    gridY[y] = math::linspaceAt(-1.0f, 1.0f, depthHeight, y);
}

// Lift the grid to the frame's resolution, sample, and mark the holes.
//
// The resize is recomputed for the two horizontal neighbours rather than the
// full-resolution grid being materialised: bilinearResizeAt is a pure function
// of its inputs, so this is bit-identical to storing it, and it saves a buffer
// the size of the frame plus a pass over it. The stretch test needs exactly
// those two neighbours and nothing else.
__global__ void sampleKernel(const float* __restrict__ image, int width, int height,
                             const float* __restrict__ gridX, const float* __restrict__ gridY,
                             int depthWidth, int depthHeight,
                             int maskBorderPix, int fixScreenBorderMask,
                             float* __restrict__ eye, float* __restrict__ mask)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t index = size_t(y) * size_t(width) + size_t(x);

    const float gx = math::bilinearResizeAt(gridX, depthWidth, depthHeight, width, height, x, y);
    const float gy = math::bilinearResizeAt(gridY, 1, depthHeight, 1, height, 0, y);

    for (int c = 0; c < 3; ++c)
    {
        eye[size_t(c) * pixels + index] =
            math::bilinearSampleBorder(image + size_t(c) * pixels, width, height, gx, gy);
    }

    // compute_stretch_mask, inlined so the neighbours can be recomputed rather
    // than read back. The threshold is at the *frame's* width, because that is
    // the resolution the grid has been lifted to.
    const float threshold = (2.0f / float(width - 1)) * math::kStretchThreshold;
    bool stretched = false;
    if (x > 0)
    {
        const float left = math::bilinearResizeAt(gridX, depthWidth, depthHeight,
                                                  width, height, x - 1, y);
        stretched = stretched || ((gx - left) < threshold);
    }
    if (x + 1 < width)
    {
        const float right = math::bilinearResizeAt(gridX, depthWidth, depthHeight,
                                                   width, height, x + 1, y);
        stretched = stretched || ((right - gx) < threshold);
    }

    if (maskBorderPix > 0)
    {
        if (x < maskBorderPix)
        {
            stretched = false;
        }
        if (fixScreenBorderMask == 2 && x >= width - maskBorderPix)
        {
            stretched = false;
        }
    }

    mask[index] = stretched ? 1.0f : 0.0f;
}

// mask_closing's four passes. One kernel with a flag rather than two, because
// the only difference is which way the reduction goes and the launch pattern is
// identical.
__global__ void morphologyKernel(const float* __restrict__ mask, int width, int height,
                                 int dilating, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    out[size_t(y) * size_t(width) + size_t(x)] =
        dilating ? math::maskDilateAt(mask, width, height, x, y)
                 : math::maskErodeAt(mask, width, height, x, y);
}

// Putting the isolated pixels back, and both directional dilations in one
// window pass. See maskFinishAt for why the two dilations collapse into one.
__global__ void maskFinishKernel(const float* __restrict__ closed,
                                 const float* __restrict__ original,
                                 int width, int height, int outer, int inner,
                                 float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    out[size_t(y) * size_t(width) + size_t(x)] =
        math::maskFinishAt(closed, original, width, x, y, outer, inner);
}

}  // namespace

MonoBwGpu::~MonoBwGpu()
{
    for (float* pointer : {_gridX, _gridY, _scratch, _eye, _mask, _maskA, _maskB})
    {
        if (pointer) cudaFree(pointer);
    }
}

bool MonoBwGpu::allocate(float** pointer, size_t floats, size_t& held)
{
    if (held >= floats && *pointer)
    {
        return true;
    }
    if (*pointer)
    {
        cudaFree(*pointer);
        *pointer = nullptr;
    }
    const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(pointer), floats * sizeof(float));
    if (status != cudaSuccess)
    {
        _error = std::string("cudaMalloc failed: ") + cudaGetErrorString(status);
        held = 0;
        return false;
    }
    held = floats;
    return true;
}

bool MonoBwGpu::prepare(int width, int height, int depthWidth, int depthHeight)
{
    _error.clear();
    _width = width;
    _height = height;
    _depthWidth = depthWidth;
    _depthHeight = depthHeight;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);

    if (!allocate(&_gridX, depthPixels, _gridXHeld)) return false;
    if (!allocate(&_gridY, size_t(depthHeight), _gridYHeld)) return false;
    // dest, moved and smoothed, laid end to end: each stage reads the previous
    // one's output, so none of them can share a buffer.
    if (!allocate(&_scratch, depthPixels * 3, _scratchHeld)) return false;
    if (!allocate(&_eye, pixels * 3, _eyeHeld)) return false;
    if (!allocate(&_mask, pixels, _maskHeld)) return false;
    // The morphology ping-pongs between these; the finished mask ends in _maskA.
    if (!allocate(&_maskA, pixels, _maskAHeld)) return false;
    if (!allocate(&_maskB, pixels, _maskBHeld)) return false;

    return _error.empty();
}

void MonoBwGpu::preprocessMask(const float* mask, int innerDilation, int outerDilation,
                               int baseWidth, void* stream)
{
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    const dim3 blocks = grid2d(_width, _height);
    const dim3 block2d(kBlock, kBlock);

    const int outer = math::maskDilateIterations(outerDilation, _width, baseWidth);
    const int inner = math::maskDilateIterations(innerDilation, _width, baseWidth);

    // dilate, dilate, erode, erode -- ping-ponging, because each pass reads its
    // input's whole neighbourhood and cannot run in place. `mask` is left alone
    // throughout: the finish step needs the original back.
    const float* source = mask;
    float* destination = _maskA;
    for (int pass = 0; pass < 4; ++pass)
    {
        morphologyKernel<<<blocks, block2d, 0, cudaStream>>>(
            source, _width, _height, pass < 2 ? 1 : 0, destination);
        source = destination;
        destination = (destination == _maskA) ? _maskB : _maskA;
    }

    // Four passes leaves the closed mask in _maskB, so the finish writes _maskA
    // and there is no aliasing with the window it reads.
    maskFinishKernel<<<blocks, block2d, 0, cudaStream>>>(
        source, mask, _width, _height, outer, inner, destination);
    _processedMask = destination;

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("mask kernel launch failed: ") + cudaGetErrorString(status);
    }
}

void MonoBwGpu::forward(const float* image, const float* depth,
                        double divergence, double convergence,
                        bool preserveScreenBorder, int fixScreenBorderMask,
                        void* stream)
{
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    const float shiftPx = float(math::monobwShiftPx(divergence, _depthWidth, _depthHeight));
    int borderPix = 0;
    if (preserveScreenBorder)
    {
        borderPix = std::min(math::monobwBorderPix(divergence, _width, _depthWidth), _depthWidth);
    }

    int maskBorderPix = 0;
    if (!preserveScreenBorder && fixScreenBorderMask > 0)
    {
        maskBorderPix = std::min(math::monobwMaskBorderPix(divergence, _width, _depthWidth), _width);
    }

    const dim3 rowGrid((unsigned(_depthHeight) + kRowBlock - 1) / kRowBlock);
    const dim3 depthGrid = grid2d(_depthWidth, _depthHeight);
    const dim3 block2d(kBlock, kBlock);

    float* dest = _scratch;
    float* moved = _scratch + size_t(_depthWidth) * size_t(_depthHeight);
    float* smoothed = moved + size_t(_depthWidth) * size_t(_depthHeight);

    destIndexKernel<<<rowGrid, kRowBlock, 0, cudaStream>>>(
        depth, _depthWidth, _depthHeight, shiftPx, float(convergence), borderPix, dest, moved);

    smoothKernel2d<<<depthGrid, block2d, 0, cudaStream>>>(
        dest, moved, _depthWidth, _depthHeight, smoothed);

    invertKernel<<<depthGrid, block2d, 0, cudaStream>>>(
        smoothed, _depthWidth, _depthHeight, _gridX);

    gridYKernel<<<rowGrid, kRowBlock, 0, cudaStream>>>(_depthHeight, _gridY);

    sampleKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
        image, _width, _height, _gridX, _gridY, _depthWidth, _depthHeight,
        maskBorderPix, fixScreenBorderMask, _eye, _mask);

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("kernel launch failed: ") + cudaGetErrorString(status);
    }
}

}  // namespace iw3
