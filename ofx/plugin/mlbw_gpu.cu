#include "mlbw_gpu.h"

#include "inpaint_boundary.cuh"
#include "mlbw_math.h"
#include "resample_gpu.cuh"
#include "stereo_pipeline.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace iw3
{
namespace
{

constexpr int kBlock = 16;

dim3 grid2d(int width, int height)
{
    return dim3(unsigned((width + kBlock - 1) / kBlock),
                unsigned((height + kBlock - 1) / kBlock));
}

__global__ void flipPlanesKernel(const float* __restrict__ source, int width, int height,
                                 int planes, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t from = size_t(y) * size_t(width) + size_t(width - 1 - x);
    const size_t to = size_t(y) * size_t(width) + size_t(x);
    for (int plane = 0; plane < planes; ++plane)
    {
        out[size_t(plane) * pixels + to] = source[size_t(plane) * pixels + from];
    }
}

// The sampling grid at the depth's resolution, one x plane per layer plus the
// shared y ramp.
//
// The y ramp is stored as a full plane rather than one value per row. It is
// row-constant, so a column of it would do -- monobw takes that shortcut -- but
// carrying the plane lets this call the same bilinearResizeAt the CPU driver
// calls, which is the thing being validated. The buffer is at the depth's size,
// not the frame's.
__global__ void gridKernel(const float* __restrict__ delta, int depthWidth, int depthHeight,
                           float deltaScale, float* __restrict__ gridX,
                           float* __restrict__ gridY)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= depthWidth || y >= depthHeight) return;

    const size_t pixels = size_t(depthWidth) * size_t(depthHeight);
    const size_t index = size_t(y) * size_t(depthWidth) + size_t(x);
    for (int layer = 0; layer < math::kMlbwLayers; ++layer)
    {
        gridX[size_t(layer) * pixels + index] = math::mlbwGridXAt(
            delta + size_t(layer) * pixels, depthWidth, depthHeight, x, y, deltaScale);
    }
    gridY[index] = math::mlbwGridYAt(depthHeight, y);
}

// The two warps and the blend. `mirror` writes to the opposite column, which is
// what puts the eye into the inpaint network's orientation without a pass of
// its own -- see MlbwGpu::prepareEye for why both eyes want it.
__global__ void warpKernel(const float* __restrict__ image, int width, int height,
                           const float* __restrict__ gridX, const float* __restrict__ gridY,
                           int depthWidth, int depthHeight,
                           const float* __restrict__ weights, int mirror,
                           float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);
    const size_t index = size_t(y) * size_t(width) + size_t(x);
    const size_t target = mirror
        ? size_t(y) * size_t(width) + size_t(width - 1 - x)
        : index;

    for (int channel = 0; channel < 3; ++channel)
    {
        const float* plane = image + size_t(channel) * pixels;
        float layerValues[math::kMlbwLayers];
        for (int layer = 0; layer < math::kMlbwLayers; ++layer)
        {
            layerValues[layer] = math::mlbwWarpSampleAt(
                plane, width, height,
                gridX + size_t(layer) * depthPixels, gridY,
                depthWidth, depthHeight, x, y);
        }
        out[size_t(channel) * pixels + target] =
            math::mlbwBlendAt(layerValues, weights, pixels, index, math::kMlbwLayers);
    }
}

// closing(n_iter=1) on the logits: one 3x3 max then one 3x3 min, on the
// greyscale values, before any sigmoid.
__global__ void closingKernel(const float* __restrict__ source, int width, int height,
                              int dilating, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    out[size_t(y) * size_t(width) + size_t(x)] =
        dilating ? math::maskDilateAt(source, width, height, x, y)
                 : math::maskErodeAt(source, width, height, x, y);
}

__global__ void maskKernel(const float* __restrict__ closed, int depthWidth, int depthHeight,
                           int width, int height, int outer, int inner,
                           float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    out[size_t(y) * size_t(width) + size_t(x)] =
        math::mlbwMaskAt(closed, depthWidth, depthHeight, width, height, x, y, outer, inner);
}

}  // namespace

MlbwGpu::~MlbwGpu()
{
    for (float* pointer : {_flipImage, _gridX, _gridY, _weights, _weightScratch,
                           _eye, _logits, _closed, _closedTmp, _mask, _final,
                           _fromHalf, _maskBlur, _maskBlurTmp, _axisWeights})
    {
        if (pointer) cudaFree(pointer);
    }
    for (void* pointer : {_eyeHalf, _maskHalf})
    {
        if (pointer) cudaFree(pointer);
    }
    for (int* pointer : {_axisStart, _axisCount, _axisOffset})
    {
        if (pointer) cudaFree(pointer);
    }
}

bool MlbwGpu::allocate(float** pointer, size_t floats, size_t& held)
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

bool MlbwGpu::uploadInts(int** pointer, const std::vector<int>& values)
{
    if (*pointer == nullptr)
    {
        if (cudaMalloc(reinterpret_cast<void**>(pointer), values.size() * sizeof(int)) != cudaSuccess)
        {
            _error = "cudaMalloc failed for resample indices";
            return false;
        }
    }
    if (cudaMemcpy(*pointer, values.data(), values.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        _error = "cudaMemcpy failed for resample indices";
        return false;
    }
    return true;
}

// The resample tables, built by exactly the function the CPU path uses. Not
// derived here: deriving them twice is how the two paths would drift, and this
// filter is the reason the warp is not in the ONNX graph at all.
void MlbwGpu::buildWeightAxes()
{
    _axesValid = false;
    if (_width == _depthWidth && _height == _depthHeight)
    {
        // iw3 skips the resize entirely when the sizes match. Going through it
        // anyway would be the identity but would still round.
        _axesValid = true;
        return;
    }

    ResampleAxis horizontal, vertical;
    buildResampleAxis(horizontal, _depthWidth, _width);
    buildResampleAxis(vertical, _depthHeight, _height);

    std::vector<int> start, count, offset;
    std::vector<float> weights;
    for (const ResampleAxis* axis : {&horizontal, &vertical})
    {
        const int base = int(weights.size());
        for (size_t i = 0; i < axis->start.size(); ++i)
        {
            start.push_back(axis->start[i]);
            count.push_back(axis->count[i]);
            offset.push_back(base + axis->offset[i]);
        }
        weights.insert(weights.end(), axis->weights.begin(), axis->weights.end());
    }

    if (_axisIntsHeld < start.size())
    {
        for (int** pointer : {&_axisStart, &_axisCount, &_axisOffset})
        {
            if (*pointer) { cudaFree(*pointer); *pointer = nullptr; }
        }
        _axisIntsHeld = 0;
    }
    if (!uploadInts(&_axisStart, start) || !uploadInts(&_axisCount, count) ||
        !uploadInts(&_axisOffset, offset))
    {
        return;
    }
    _axisIntsHeld = std::max(_axisIntsHeld, start.size());

    if (_axisWeightsHeld < weights.size())
    {
        if (_axisWeights) cudaFree(_axisWeights);
        if (cudaMalloc(reinterpret_cast<void**>(&_axisWeights),
                       weights.size() * sizeof(float)) != cudaSuccess)
        {
            _error = "cudaMalloc failed for resample weights";
            return;
        }
        _axisWeightsHeld = weights.size();
    }
    if (cudaMemcpy(_axisWeights, weights.data(), weights.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        _error = "cudaMemcpy failed for resample weights";
        return;
    }
    _axesValid = true;
}

bool MlbwGpu::prepare(int width, int height, int depthWidth, int depthHeight)
{
    const bool geometryChanged = (width != _width || height != _height ||
                                  depthWidth != _depthWidth || depthHeight != _depthHeight);

    _width = width;
    _height = height;
    _depthWidth = depthWidth;
    _depthHeight = depthHeight;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);
    const size_t layers = size_t(math::kMlbwLayers);

    if (!allocate(&_flipImage, pixels * 3, _flipImageHeld)) return false;
    if (!allocate(&_gridX, depthPixels * layers, _gridXHeld)) return false;
    if (!allocate(&_gridY, depthPixels, _gridYHeld)) return false;
    if (!allocate(&_weights, pixels * layers, _weightsHeld)) return false;
    if (!allocate(&_weightScratch, size_t(width) * size_t(depthHeight) * layers,
                  _weightScratchHeld)) return false;
    if (!allocate(&_eye, pixels * 3, _eyeHeld)) return false;
    if (!allocate(&_logits, depthPixels, _logitsHeld)) return false;
    if (!allocate(&_closed, depthPixels, _closedHeld)) return false;
    if (!allocate(&_closedTmp, depthPixels, _closedTmpHeld)) return false;
    if (!allocate(&_mask, pixels, _maskHeld)) return false;
    if (!allocate(&_final, pixels * 3, _finalHeld)) return false;
    // The half buffers go through the same float allocator, rounded up: two
    // halves to a float, which is what monobw does and for the same reason --
    // one allocator rather than two.
    if (!allocate(reinterpret_cast<float**>(&_eyeHalf), (pixels * 3 + 1) / 2, _eyeHalfHeld))
        return false;
    if (!allocate(reinterpret_cast<float**>(&_maskHalf), (pixels + 1) / 2, _maskHalfHeld))
        return false;
    if (!allocate(&_fromHalf, pixels * 3, _fromHalfHeld)) return false;
    if (!allocate(&_maskBlur, pixels, _maskBlurHeld)) return false;
    if (!allocate(&_maskBlurTmp, pixels, _maskBlurTmpHeld)) return false;

    if (geometryChanged || !_axesValid)
    {
        buildWeightAxes();
    }
    return _axesValid && ok();
}

bool MlbwGpu::prepareEye(const float* image, const float* delta, const float* layerWeight,
                         const float* maskLogits, bool rightEye,
                         int innerDilation, int outerDilation, void* stream)
{
    if (!ok() || !_axesValid) return false;
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    const size_t pixels = size_t(_width) * size_t(_height);
    const size_t depthPixels = size_t(_depthWidth) * size_t(_depthHeight);
    const float deltaScale = float(1.0 / double(_depthWidth / 2 - 1));

    // The right eye's model saw the mirrored frame, so its warp must sample the
    // mirrored frame too.
    const float* source = image;
    if (rightEye)
    {
        flipPlanesKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
            image, _width, _height, 3, _flipImage);
        source = _flipImage;
    }

    gridKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0, cudaStream>>>(
        delta, _depthWidth, _depthHeight, deltaScale, _gridX, _gridY);

    // The weights up to the frame, one layer at a time through the shared
    // antialiased resampler. When the sizes already match iw3 does not resize,
    // so neither does this.
    const float* weights = layerWeight;
    if (_width != _depthWidth || _height != _depthHeight)
    {
        for (int layer = 0; layer < math::kMlbwLayers; ++layer)
        {
            const float* in = layerWeight + size_t(layer) * depthPixels;
            float* scratch = _weightScratch + size_t(layer) * size_t(_width) * size_t(_depthHeight);
            float* out = _weights + size_t(layer) * pixels;

            resizeHorizontalKernel<<<grid2d(_width, _depthHeight), dim3(kBlock, kBlock), 0,
                                     cudaStream>>>(
                in, _depthWidth, _depthHeight,
                _axisStart, _axisCount, _axisOffset, _axisWeights, _width, scratch);
            resizeVerticalKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0,
                                   cudaStream>>>(
                scratch, _width, _depthHeight,
                _axisStart + _width, _axisCount + _width, _axisOffset + _width, _axisWeights,
                _height, out);
        }
        weights = _weights;
    }

    // Mirrored for both eyes -- see the header. Getting this wrong produces a
    // picture rather than an error, which is why the mlbw_eye_* reference cases
    // record what the network is actually handed.
    warpKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
        source, _width, _height, _gridX, _gridY, _depthWidth, _depthHeight,
        weights, 1, _eye);

    // The logits are mirrored for both eyes too, and physically rather than by
    // a read index: the closing is symmetric and the resize is symmetric, but
    // the dilation window is not, so the flip has to happen before the chain
    // rather than after it.
    flipPlanesKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0, cudaStream>>>(
        maskLogits, _depthWidth, _depthHeight, 1, _logits);

    closingKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0, cudaStream>>>(
        _logits, _depthWidth, _depthHeight, 1, _closedTmp);
    closingKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0, cudaStream>>>(
        _closedTmp, _depthWidth, _depthHeight, 0, _closed);

    const int outer = math::maskDilateIterations(outerDilation, _width, _depthWidth);
    const int inner = math::maskDilateIterations(innerDilation, _width, _depthWidth);
    maskKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
        _closed, _depthWidth, _depthHeight, _width, _height, outer, inner, _mask);

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("mlbw kernel launch failed: ") + cudaGetErrorString(status);
        return false;
    }
    (void)pixels;
    return true;
}

// Identical to MonoBwGpu's, and deliberately so: the same kernels out of
// inpaint_boundary.cuh, in the same order, on the same two buffers. What differs
// between the pipelines is which warp opened the holes, and by this point that
// is behind us.
void MlbwGpu::prepareInpaintInput(int maxWidth, void* stream)
{
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    const size_t pixels = size_t(_width) * size_t(_height);
    constexpr int kThreads = 256;

    math::inpaintWorkingSize(_width, _height, maxWidth, _workWidth, _workHeight);

    if (_workWidth == _width && _workHeight == _height)
    {
        const size_t eyeCount = pixels * 3;
        toHalfKernel<<<unsigned((eyeCount + kThreads - 1) / kThreads), kThreads, 0, cudaStream>>>(
            _eye, eyeCount, static_cast<__half*>(_eyeHalf));
        toHalfKernel<<<unsigned((pixels + kThreads - 1) / kThreads), kThreads, 0, cudaStream>>>(
            _mask, pixels, static_cast<__half*>(_maskHalf));
    }
    else
    {
        const dim3 blocks = grid2d(_workWidth, _workHeight);
        const dim3 block2d(kBlock, kBlock);
        downscaleEyeKernel<<<blocks, block2d, 0, cudaStream>>>(
            _eye, _width, _height, _mask, _workWidth, _workHeight,
            static_cast<__half*>(_eyeHalf));
        downscaleMaskKernel<<<blocks, block2d, 0, cudaStream>>>(
            _mask, _width, _height, _workWidth, _workHeight,
            static_cast<__half*>(_maskHalf));

        // The feather for the full-resolution composite, built here so it
        // overlaps the inference rather than being computed after it.
        const dim3 fullBlocks = grid2d(_width, _height);
        maskBlurKernel<<<fullBlocks, block2d, 0, cudaStream>>>(
            _mask, _width, _height, 1, _maskBlurTmp);
        maskBlurKernel<<<fullBlocks, block2d, 0, cudaStream>>>(
            _maskBlurTmp, _width, _height, 0, _maskBlur);
    }

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("inpaint input preparation failed: ") + cudaGetErrorString(status);
    }
}

const float* MlbwGpu::finishInpaintOutput(const void* filled, void* stream)
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
        _eye, static_cast<const __half*>(filled), _maskBlur, _mask,
        _width, _height, _workWidth, _workHeight, _fromHalf);
    return _fromHalf;
}

const float* MlbwGpu::finishEye(const float* filled, bool rightEye, void* stream)
{
    if (!rightEye)
    {
        // The left eye was inpainted mirrored; put it back.
        cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
        flipPlanesKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
            filled, _width, _height, 3, _final);
        return _final;
    }
    return filled;
}

}  // namespace iw3
