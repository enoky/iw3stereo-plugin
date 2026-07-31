#include "monobw_gpu.h"

#include "inpaint_boundary.cuh"
#include "monobw_math.h"

#include <cuda_fp16.h>
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
                             int maskBorderPix, int fixScreenBorderMask, int mirrorOutput,
                             float* __restrict__ eye, float* __restrict__ mask)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    // Everything below reads and decides in warp coordinates -- including the
    // screen-border fix, which is defined on the warped frame. Only the write
    // is mirrored, which makes the mirror exactly a permutation.
    const size_t index = size_t(y) * size_t(width) +
                         size_t(mirrorOutput ? (width - 1 - x) : x);

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

// Horizontal mirror of a planar buffer. Exactly a permutation of the samples,
// so it introduces no arithmetic difference of its own.
__global__ void flipKernel(const float* __restrict__ source, int width, int height, int planes,
                           float* __restrict__ destination)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t from = size_t(y) * size_t(width) + size_t(x);
    const size_t to = size_t(y) * size_t(width) + size_t(width - 1 - x);
    for (int plane = 0; plane < planes; ++plane)
    {
        destination[size_t(plane) * pixels + to] = source[size_t(plane) * pixels + from];
    }
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
    for (float* pointer : {_gridX, _gridY, _scratch, _eye, _mask, _maskA, _maskB,
                           _flipImage, _flipDepth, _final, _fromHalf,
                           _maskBlur, _maskBlurTmp})
    {
        if (pointer) cudaFree(pointer);
    }
    for (void* pointer : {_eyeHalf, _maskHalf})
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
    if (!allocate(&_flipImage, pixels * 3, _flipImageHeld)) return false;
    if (!allocate(&_flipDepth, depthPixels, _flipDepthHeld)) return false;
    if (!allocate(&_final, pixels * 3, _finalHeld)) return false;
    // Half buffers are allocated as floats of half the count, which is the same
    // number of bytes and saves a second allocator.
    if (!allocate(reinterpret_cast<float**>(&_eyeHalf), (pixels * 3 + 1) / 2, _eyeHalfHeld))
        return false;
    if (!allocate(reinterpret_cast<float**>(&_maskHalf), (pixels + 1) / 2, _maskHalfHeld))
        return false;
    if (!allocate(&_fromHalf, pixels * 3, _fromHalfHeld)) return false;
    if (!allocate(&_maskBlur, pixels, _maskBlurHeld)) return false;
    if (!allocate(&_maskBlurTmp, pixels, _maskBlurTmpHeld)) return false;

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
                        bool mirrorOutput, void* stream)
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
        maskBorderPix, fixScreenBorderMask, mirrorOutput ? 1 : 0, _eye, _mask);

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("kernel launch failed: ") + cudaGetErrorString(status);
    }
}

bool MonoBwGpu::prepareEye(const float* image, const float* depth, bool rightEye,
                           double divergence, double convergence, bool preserveScreenBorder,
                           int innerDilation, int outerDilation, int baseWidth, void* stream)
{
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    const float* warpImage = image;
    const float* warpDepth = depth;
    if (rightEye)
    {
        // iw3 mirrors the inputs, warps, and mirrors the result back. The
        // mirror of the result is folded into the sample kernel's write below,
        // so only the inputs need a pass of their own.
        flipKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
            image, _width, _height, 3, _flipImage);
        flipKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0, cudaStream>>>(
            depth, _depthWidth, _depthHeight, 1, _flipDepth);
        warpImage = _flipImage;
        warpDepth = _flipDepth;
    }

    // The mirror on the way out serves two different purposes and happens to be
    // the same operation for both eyes. For the right eye it undoes the input
    // mirror, putting the eye back in frame orientation, which is the
    // orientation the network wants for that side. For the left eye it is the
    // flip _inpaint_single does before inferring. Either way what comes out of
    // here is what the network should see.
    forward(warpImage, warpDepth, divergence, convergence, preserveScreenBorder,
            1 /* fix the uninpaintable side, as the image model does */,
            true /* mirrorOutput */, stream);
    if (!ok())
    {
        return false;
    }

    // The morphology runs on the mask in that same orientation, which matters:
    // dilate_inner and dilate_outer are directional.
    preprocessMask(_mask, innerDilation, outerDilation, baseWidth, stream);
    return ok();
}

void MonoBwGpu::prepareInpaintInput(int maxWidth, void* stream)
{
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    const size_t pixels = size_t(_width) * size_t(_height);
    constexpr int kThreads = 256;

    math::inpaintWorkingSize(_width, _height, maxWidth, _workWidth, _workHeight);

    if (_workWidth == _width && _workHeight == _height)
    {
        // Nothing to reduce: the same straight cast this always did.
        const size_t eyeCount = pixels * 3;
        toHalfKernel<<<unsigned((eyeCount + kThreads - 1) / kThreads), kThreads, 0, cudaStream>>>(
            _eye, eyeCount, static_cast<__half*>(_eyeHalf));
        toHalfKernel<<<unsigned((pixels + kThreads - 1) / kThreads), kThreads, 0, cudaStream>>>(
            _processedMask, pixels, static_cast<__half*>(_maskHalf));
    }
    else
    {
        const dim3 blocks = grid2d(_workWidth, _workHeight);
        const dim3 block2d(kBlock, kBlock);
        downscaleEyeKernel<<<blocks, block2d, 0, cudaStream>>>(
            _eye, _width, _height, _processedMask, _workWidth, _workHeight,
            static_cast<__half*>(_eyeHalf));
        downscaleMaskKernel<<<blocks, block2d, 0, cudaStream>>>(
            _processedMask, _width, _height, _workWidth, _workHeight,
            static_cast<__half*>(_maskHalf));

        // The feather for the full-resolution composite, built here so it
        // overlaps the inference rather than being computed after it.
        const dim3 fullBlocks = grid2d(_width, _height);
        maskBlurKernel<<<fullBlocks, block2d, 0, cudaStream>>>(
            _processedMask, _width, _height, 1, _maskBlurTmp);
        maskBlurKernel<<<fullBlocks, block2d, 0, cudaStream>>>(
            _maskBlurTmp, _width, _height, 0, _maskBlur);
    }

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("inpaint input preparation failed: ") + cudaGetErrorString(status);
    }
}

const float* MonoBwGpu::finishInpaintOutput(const void* filled, void* stream)
{
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    const size_t count = size_t(_width) * size_t(_height) * 3;
    constexpr int kThreads = 256;

    if (_workWidth == _width && _workHeight == _height)
    {
        // The graph already composited against the eye it was given, so this is
        // only a widening.
        fromHalfKernel<<<unsigned((count + kThreads - 1) / kThreads), kThreads, 0, cudaStream>>>(
            static_cast<const __half*>(filled), count, _fromHalf);
        return _fromHalf;
    }

    compositeUpscaledKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
        _eye, static_cast<const __half*>(filled), _maskBlur, _processedMask,
        _width, _height, _workWidth, _workHeight, _fromHalf);
    return _fromHalf;
}

const float* MonoBwGpu::finishEye(const float* filled, bool rightEye, void* stream)
{
    if (rightEye)
    {
        // Already in frame orientation: the input mirror and the output mirror
        // cancelled, and the network ran on the result of both.
        return filled;
    }
    flipKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0,
                 static_cast<cudaStream_t>(stream)>>>(filled, _width, _height, 3, _final);
    return _final;
}



// ---------------------------------------------------------------------------
// MonoBwVideoGpu

namespace
{

}  // namespace

MonoBwVideoGpu::~MonoBwVideoGpu()
{
    for (void* pointer : {_eyes, _masks, _cache[0], _cache[1]})
    {
        if (pointer) cudaFree(pointer);
    }
}

bool MonoBwVideoGpu::allocate(void** pointer, size_t bytes, size_t& held)
{
    if (held >= bytes && *pointer)
    {
        return true;
    }
    if (*pointer)
    {
        cudaFree(*pointer);
        *pointer = nullptr;
    }
    const cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess)
    {
        _error = std::string("cudaMalloc failed: ") + cudaGetErrorString(status);
        held = 0;
        return false;
    }
    held = bytes;
    return true;
}

long long MonoBwVideoGpu::windowIndex(long long frame)
{
    // Floor division, so negative frame numbers land in the window below rather
    // than being truncated towards zero into the one above.
    return (frame >= 0) ? (frame / kStride)
                        : -(((-frame) + kStride - 1) / kStride);
}

long long MonoBwVideoGpu::windowFirstFrame(long long frame)
{
    return windowIndex(frame) * kStride - kPad;
}

bool MonoBwVideoGpu::prepare(int width, int height)
{
    _error.clear();
    if (width != _width || height != _height)
    {
        _valid = false;
    }
    _width = width;
    _height = height;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t half = sizeof(unsigned short);

    if (!allocate(&_eyes, pixels * 3 * kSequence * half, _eyesHeld)) return false;
    if (!allocate(&_masks, pixels * kSequence * half, _masksHeld)) return false;
    for (int eye = 0; eye < 2; ++eye)
    {
        if (!allocate(&_cache[eye], pixels * 3 * kStride * half, _cacheHeld[eye])) return false;
    }
    return _error.empty();
}

void MonoBwVideoGpu::storeFrame(int index, const void* eyeHalf, const void* maskHalf, void* stream)
{
    const size_t pixels = size_t(_width) * size_t(_height);
    const size_t half = sizeof(unsigned short);
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    cudaMemcpyAsync(static_cast<char*>(_eyes) + size_t(index) * pixels * 3 * half,
                    eyeHalf, pixels * 3 * half, cudaMemcpyDeviceToDevice, cudaStream);
    cudaMemcpyAsync(static_cast<char*>(_masks) + size_t(index) * pixels * half,
                    maskHalf, pixels * half, cudaMemcpyDeviceToDevice, cudaStream);
}

void MonoBwVideoGpu::cacheOutput(bool rightEye, const void* filledHalf, void* stream)
{
    // Only the middle kStride frames are kept; the padding at each end exists
    // to give those frames neighbours, not to be used itself.
    const size_t pixels = size_t(_width) * size_t(_height);
    const size_t half = sizeof(unsigned short);
    const size_t frameBytes = pixels * 3 * half;

    cudaMemcpyAsync(_cache[rightEye ? 1 : 0],
                    static_cast<const char*>(filledHalf) + size_t(kPad) * frameBytes,
                    frameBytes * kStride, cudaMemcpyDeviceToDevice,
                    static_cast<cudaStream_t>(stream));
}

const void* MonoBwVideoGpu::cachedFrame(bool rightEye, int offset) const
{
    const size_t pixels = size_t(_width) * size_t(_height);
    const size_t half = sizeof(unsigned short);
    return static_cast<const char*>(_cache[rightEye ? 1 : 0]) + size_t(offset) * pixels * 3 * half;
}

}  // namespace iw3
