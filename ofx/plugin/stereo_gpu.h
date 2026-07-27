// The GPU path: everything between Resolve's device buffers and ONNX Runtime,
// without a frame ever crossing PCIe.
//
// Resolve renders with CUDA when the plugin asks for it (cuda=1 cudaStream=1 in
// the host report), and then OFX::Image::getPixelData() is a *device* pointer
// and RenderArguments::pCudaStream is the stream to work on. BMD's own
// GainPlugin sample establishes that contract.
//
// The arithmetic is not reimplemented here. numeric_math.h is compiled for the
// device, and the resample weights come from buildResampleAxis() on the host,
// so the kernels run the same lines the CPU test validates.

#pragma once

#include "stereo_pipeline.h"

#include <string>

namespace iw3
{

class GpuPipeline
{
public:
    GpuPipeline() = default;
    ~GpuPipeline();

    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // Allocates for this frame geometry. Cheap when nothing changed.
    bool prepare(int width, int height, int depthWidth, int depthHeight);

    // Resolve's source -> planar RGB. rowPitch is in floats.
    void packSource(const float* source, size_t rowPitch, int components, void* stream);

    // Resolve's depth clip -> a single mapped channel at frame resolution.
    void packDepth(const float* depth, size_t rowPitch, int components,
                   bool undoVideoRange, bool inverted,
                   const math::MapperParams& mapper, void* stream);

    // The stereo_width resize. Skipped when the sizes already match.
    void resizeDepth(void* stream);

    void buildInputTensor(double divergence, double convergence,
                          bool preserveScreenBorder, void* stream);

    // left and right are device pointers from the ONNX session's output.
    void compose(OutputMode mode, const float* left, const float* right, void* stream);

    // Planar RGB -> Resolve's destination.
    void unpack(float* destination, size_t rowPitch, int components, void* stream);

    const float* imageDevice() const { return _image; }
    const float* inputTensorDevice() const { return _x; }

    bool ok() const { return _error.empty(); }
    const std::string& error() const { return _error; }

    // True if a CUDA device is usable at all. Checked once.
    static bool deviceAvailable();

private:
    void uploadWeights();
    bool allocate(float** pointer, size_t floats, size_t& held);

    int _width = 0, _height = 0, _depthWidth = 0, _depthHeight = 0;

    float* _image = nullptr;      // 3 * width * height
    float* _depthFull = nullptr;  // width * height
    float* _depthSmall = nullptr; // depthWidth * depthHeight
    float* _scratch = nullptr;    // height * depthWidth, for the separable resize
    float* _x = nullptr;          // 3 * depthWidth * depthHeight
    float* _composed = nullptr;   // 3 * width * height

    size_t _imageHeld = 0, _depthFullHeld = 0, _depthSmallHeld = 0;
    size_t _scratchHeld = 0, _xHeld = 0, _composedHeld = 0;

    ResampleAxis _horizontal, _vertical;
    int* _axisStart = nullptr;
    int* _axisCount = nullptr;
    int* _axisOffset = nullptr;
    float* _axisWeights = nullptr;
    size_t _axisIntsHeld = 0, _axisWeightsHeld = 0;
    bool _weightsCurrent = false;

    std::string _error;
};

}  // namespace iw3
