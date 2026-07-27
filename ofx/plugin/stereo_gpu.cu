#include "stereo_gpu.h"

#include "numeric_math.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace iw3
{

namespace
{

constexpr int kBlock = 16;

dim3 grid2d(int width, int height)
{
    return dim3((unsigned(width) + kBlock - 1) / kBlock, (unsigned(height) + kBlock - 1) / kBlock);
}

__device__ inline bool insideSource(int x, int y, int offsetX, int offsetY,
                                    int sourceWidth, int sourceHeight, int& sx, int& sy)
{
    sx = x + offsetX;
    sy = y + offsetY;
    return sx >= 0 && sx < sourceWidth && sy >= 0 && sy < sourceHeight;
}

__global__ void packSourceKernel(const float* __restrict__ source, size_t rowPitch, int components,
                                 int offsetX, int offsetY, int sourceWidth, int sourceHeight,
                                 int width, int height, float* __restrict__ planar)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t index = size_t(y) * size_t(width) + size_t(x);

    int sx = 0, sy = 0;
    if (!source || !insideSource(x, y, offsetX, offsetY, sourceWidth, sourceHeight, sx, sy))
    {
        planar[index] = planar[pixels + index] = planar[2 * pixels + index] = 0.0f;
        return;
    }
    const float* pixel = source + size_t(sy) * rowPitch + size_t(sx) * size_t(components);
    planar[index] = pixel[0];
    planar[pixels + index] = pixel[1];
    planar[2 * pixels + index] = pixel[2];
}

__global__ void packDepthKernel(const float* __restrict__ source, size_t rowPitch, int components,
                                int offsetX, int offsetY, int sourceWidth, int sourceHeight,
                                int width, int height,
                                int undoVideoRange, int inverted,
                                math::MapperParams mapper,
                                float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int sx = 0, sy = 0;
    if (!source || !insideSource(x, y, offsetX, offsetY, sourceWidth, sourceHeight, sx, sy))
    {
        out[size_t(y) * size_t(width) + size_t(x)] = 0.0f;
        return;
    }
    float value = source[size_t(sy) * rowPitch + size_t(sx) * size_t(components)];
    if (undoVideoRange)
    {
        // Resolve expanded 16-235 to 0-1 on the way in; put it back.
        value = (value * 219.0f + 16.0f) / 255.0f;
    }
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    if (inverted)
    {
        value = 1.0f - value;
    }
    out[size_t(y) * size_t(width) + size_t(x)] = float(math::applyMapper(mapper, double(value)));
}

// The separable antialiased resample, in the same order as the CPU version:
// horizontal into scratch, then vertical.
__global__ void resizeHorizontalKernel(const float* __restrict__ source, int sourceWidth, int height,
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

__global__ void resizeVerticalKernel(const float* __restrict__ scratch, int targetWidth, int sourceHeight,
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

__global__ void inputTensorKernel(const float* __restrict__ depth, int width, int height,
                                  float divergenceValue, float convergenceValue,
                                  int borderPix, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t index = size_t(y) * size_t(width) + size_t(x);

    float divergence = divergenceValue;
    float convergence = convergenceValue;
    if (borderPix > 0)
    {
        // Taper to zero at both edges. The ramp is symmetric, so a pixel's
        // weight depends only on its distance from the nearer edge.
        const int distance = min(x, width - 1 - x);
        if (distance < borderPix)
        {
            const float weight = borderPix > 1 ? float(distance) / float(borderPix - 1) : 0.0f;
            divergence *= weight;
            convergence *= weight;
        }
    }

    out[index] = depth[index];
    out[pixels + index] = divergence;
    out[2 * pixels + index] = convergence;
}

__global__ void composeKernel(int mode, const float* __restrict__ left, const float* __restrict__ right,
                              int width, int height,
                              const float* __restrict__ depth, int depthWidth, int depthHeight,
                              float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t index = size_t(y) * size_t(width) + size_t(x);

    if (mode == 0)  // anaglyph
    {
        double r, g, b;
        math::duboisPixel(left[index], left[pixels + index], left[2 * pixels + index],
                          right[index], right[pixels + index], right[2 * pixels + index],
                          r, g, b);
        out[index] = float(math::clamp01(r));
        out[pixels + index] = float(math::clamp01(g));
        out[2 * pixels + index] = float(math::clamp01(b));
    }
    else if (mode == 1 || mode == 2)  // left eye / right eye
    {
        const float* eye = mode == 1 ? left : right;
        out[index] = eye[index];
        out[pixels + index] = eye[pixels + index];
        out[2 * pixels + index] = eye[2 * pixels + index];
    }
    else if (mode == 3)  // half SBS
    {
        const int half = width / 2;
        const float* eye = x < half ? left : right;
        const int local = x < half ? x : x - half;
        const int a = min(local * 2, width - 1);
        const int b = min(local * 2 + 1, width - 1);
        const size_t rowBase = size_t(y) * size_t(width);
        if (x < half * 2)
        {
            for (int c = 0; c < 3; ++c)
            {
                const float* plane = eye + size_t(c) * pixels + rowBase;
                out[size_t(c) * pixels + index] = 0.5f * (plane[a] + plane[b]);
            }
        }
        else
        {
            // An odd width leaves one column over; black is honest about it.
            for (int c = 0; c < 3; ++c)
            {
                out[size_t(c) * pixels + index] = 0.0f;
            }
        }
    }
    else  // depth, nearest-upscaled so the effect of Stereo Width is visible
    {
        const int sy = min(y * depthHeight / max(1, height), depthHeight - 1);
        const int sx = min(x * depthWidth / max(1, width), depthWidth - 1);
        const float value = depth[size_t(sy) * size_t(depthWidth) + size_t(sx)];
        out[index] = value;
        out[pixels + index] = value;
        out[2 * pixels + index] = value;
    }
}

__global__ void unpackKernel(const float* __restrict__ planar, int width, int height,
                             float* __restrict__ destination, size_t rowPitch, int components)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t index = size_t(y) * size_t(width) + size_t(x);
    float* pixel = destination + size_t(y) * rowPitch + size_t(x) * size_t(components);
    pixel[0] = planar[index];
    pixel[1] = planar[pixels + index];
    pixel[2] = planar[2 * pixels + index];
    if (components > 3)
    {
        pixel[3] = 1.0f;
    }
}

__global__ void passthroughKernel(const float* __restrict__ source, size_t sourcePitch,
                                  int sourceOffsetX, int sourceOffsetY,
                                  int sourceWidth, int sourceHeight,
                                  float* __restrict__ destination, size_t destinationPitch,
                                  int width, int height, int components)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float* out = destination + size_t(y) * destinationPitch + size_t(x) * size_t(components);
    const int sx = x + sourceOffsetX;
    const int sy = y + sourceOffsetY;

    if (source && sx >= 0 && sx < sourceWidth && sy >= 0 && sy < sourceHeight)
    {
        const float* in = source + size_t(sy) * sourcePitch + size_t(sx) * size_t(components);
        for (int c = 0; c < components; ++c)
        {
            out[c] = in[c];
        }
    }
    else
    {
        for (int c = 0; c < components; ++c)
        {
            out[c] = 0.0f;
        }
    }
}

}  // namespace

bool devicePassthrough(const float* source, size_t sourcePitch,
                       int sourceOffsetX, int sourceOffsetY,
                       int sourceWidth, int sourceHeight,
                       float* destination, size_t destinationPitch,
                       int width, int height, int components, void* stream)
{
    if (!destination || width <= 0 || height <= 0)
    {
        return false;
    }
    passthroughKernel<<<grid2d(width, height), dim3(kBlock, kBlock), 0,
                        static_cast<cudaStream_t>(stream)>>>(
        source, sourcePitch, sourceOffsetX, sourceOffsetY, sourceWidth, sourceHeight,
        destination, destinationPitch, width, height, components);
    return cudaGetLastError() == cudaSuccess;
}

bool GpuPipeline::deviceAvailable()
{
    static const bool available = []()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }();
    return available;
}

GpuPipeline::~GpuPipeline()
{
    for (float* pointer : {_image, _depthFull, _depthSmall, _scratch, _x, _composed, _axisWeights})
    {
        if (pointer) cudaFree(pointer);
    }
    for (int* pointer : {_axisStart, _axisCount, _axisOffset})
    {
        if (pointer) cudaFree(pointer);
    }
}

bool GpuPipeline::allocate(float** pointer, size_t floats, size_t& held)
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

bool GpuPipeline::prepare(int width, int height, int depthWidth, int depthHeight)
{
    _error.clear();
    if (depthWidth != _depthWidth || depthHeight != _depthHeight ||
        width != _width || height != _height)
    {
        _weightsCurrent = false;
    }
    _width = width;
    _height = height;
    _depthWidth = depthWidth;
    _depthHeight = depthHeight;

    const size_t pixels = size_t(width) * size_t(height);
    const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);

    if (!allocate(&_image, pixels * 3, _imageHeld)) return false;
    if (!allocate(&_depthFull, pixels, _depthFullHeld)) return false;
    if (!allocate(&_depthSmall, depthPixels, _depthSmallHeld)) return false;
    if (!allocate(&_scratch, size_t(height) * size_t(depthWidth), _scratchHeld)) return false;
    if (!allocate(&_x, depthPixels * 3, _xHeld)) return false;
    if (!allocate(&_composed, pixels * 3, _composedHeld)) return false;

    if (!_weightsCurrent)
    {
        uploadWeights();
    }
    return _error.empty();
}

void GpuPipeline::uploadWeights()
{
    if (_depthWidth == _width && _depthHeight == _height)
    {
        _weightsCurrent = true;  // no resize needed
        return;
    }

    // Exactly the host's weights, not a device-side rederivation: that is what
    // keeps this identical to the path the CPU test covers.
    buildResampleAxis(_horizontal, _width, _depthWidth);
    buildResampleAxis(_vertical, _height, _depthHeight);

    std::vector<int> start, count, offset;
    std::vector<float> weights;
    const auto append = [&](const ResampleAxis& axis)
    {
        const int weightBase = int(weights.size());
        for (size_t i = 0; i < axis.start.size(); ++i)
        {
            start.push_back(axis.start[i]);
            count.push_back(axis.count[i]);
            offset.push_back(weightBase + axis.offset[i]);
        }
        weights.insert(weights.end(), axis.weights.begin(), axis.weights.end());
    };
    append(_horizontal);
    append(_vertical);

    const auto uploadInts = [&](int** pointer, const std::vector<int>& data)
    {
        if (_axisIntsHeld < data.size())
        {
            if (*pointer) cudaFree(*pointer);
            if (cudaMalloc(reinterpret_cast<void**>(pointer), data.size() * sizeof(int)) != cudaSuccess)
            {
                _error = "cudaMalloc failed for resample indices";
                return false;
            }
        }
        return cudaMemcpy(*pointer, data.data(), data.size() * sizeof(int),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    };

    const size_t entries = start.size();
    if (_axisIntsHeld < entries)
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
    _axisIntsHeld = std::max(_axisIntsHeld, entries);

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
        _error = "resample weight upload failed";
        return;
    }
    _weightsCurrent = true;
}

void GpuPipeline::packSource(const float* source, size_t rowPitch, int components,
                             int offsetX, int offsetY, int sourceWidth, int sourceHeight,
                             void* stream)
{
    packSourceKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0,
                       static_cast<cudaStream_t>(stream)>>>(
        source, rowPitch, components, offsetX, offsetY, sourceWidth, sourceHeight,
        _width, _height, _image);
}

void GpuPipeline::packDepth(const float* depth, size_t rowPitch, int components,
                            int offsetX, int offsetY, int sourceWidth, int sourceHeight,
                            bool undoVideoRange, bool inverted,
                            const math::MapperParams& mapper, void* stream)
{
    packDepthKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0,
                      static_cast<cudaStream_t>(stream)>>>(
        depth, rowPitch, components, offsetX, offsetY, sourceWidth, sourceHeight,
        _width, _height, undoVideoRange ? 1 : 0, inverted ? 1 : 0, mapper, _depthFull);
}

void GpuPipeline::resizeDepth(void* stream)
{
    if (_depthWidth == _width && _depthHeight == _height)
    {
        return;  // _depthFull is already the model's input
    }
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    const int horizontalEntries = _depthWidth;

    resizeHorizontalKernel<<<grid2d(_depthWidth, _height), dim3(kBlock, kBlock), 0, cudaStream>>>(
        _depthFull, _width, _height,
        _axisStart, _axisCount, _axisOffset, _axisWeights,
        _depthWidth, _scratch);

    resizeVerticalKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0, cudaStream>>>(
        _scratch, _depthWidth, _height,
        _axisStart + horizontalEntries, _axisCount + horizontalEntries,
        _axisOffset + horizontalEntries, _axisWeights,
        _depthHeight, _depthSmall);
}

void GpuPipeline::buildInputTensor(double divergence, double convergence,
                                   bool preserveScreenBorder, void* stream)
{
    const double imageWidth = double(std::max(_depthWidth, _depthHeight));
    float divergenceValue = 0.0f, convergenceValue = 0.0f;
    math::featureValues(divergence, convergence, imageWidth, divergenceValue, convergenceValue);

    int borderPix = 0;
    if (preserveScreenBorder)
    {
        // Left as iw3 writes it; imageWidth cancels but the rounding does not.
        borderPix = int(std::lround(divergence * 0.75 * 0.01 * imageWidth *
                                    (double(_depthWidth) / imageWidth)));
        borderPix = std::min(borderPix, _depthWidth);
    }

    const float* depth = (_depthWidth == _width && _depthHeight == _height) ? _depthFull : _depthSmall;
    inputTensorKernel<<<grid2d(_depthWidth, _depthHeight), dim3(kBlock, kBlock), 0,
                        static_cast<cudaStream_t>(stream)>>>(
        depth, _depthWidth, _depthHeight, divergenceValue, convergenceValue, borderPix, _x);
}

void GpuPipeline::compose(OutputMode mode, const float* left, const float* right, void* stream)
{
    const int modeIndex = mode == OutputMode::Anaglyph ? 0
                        : mode == OutputMode::LeftEye ? 1
                        : mode == OutputMode::RightEye ? 2
                        : mode == OutputMode::HalfSbs ? 3
                                                      : 4;
    const float* depth = (_depthWidth == _width && _depthHeight == _height) ? _depthFull : _depthSmall;
    composeKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0,
                    static_cast<cudaStream_t>(stream)>>>(
        modeIndex, left, right, _width, _height, depth, _depthWidth, _depthHeight, _composed);
}

void GpuPipeline::unpack(float* destination, size_t rowPitch, int components, void* stream)
{
    unpackKernel<<<grid2d(_width, _height), dim3(kBlock, kBlock), 0,
                   static_cast<cudaStream_t>(stream)>>>(
        _composed, _width, _height, destination, rowPitch, components);

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        _error = std::string("kernel launch failed: ") + cudaGetErrorString(status);
    }
}

}  // namespace iw3
