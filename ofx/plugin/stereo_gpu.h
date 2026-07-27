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
    //
    // An OFX image's bounds need not start at the render window's origin, nor
    // cover it. offsetX/Y locate the window inside the image and sourceWidth/
    // Height give its extent; anything outside reads as black rather than
    // running off the allocation. Ignoring this is an out-of-bounds device read,
    // which is a GPU fault, which takes Resolve with it.
    void packSource(const float* source, size_t rowPitch, int components,
                    int offsetX, int offsetY, int sourceWidth, int sourceHeight, void* stream);

    // Resolve's depth clip -> a single mapped channel at frame resolution.
    void packDepth(const float* depth, size_t rowPitch, int components,
                   int offsetX, int offsetY, int sourceWidth, int sourceHeight,
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

// Copy Resolve's source buffer straight to its destination, on the device.
//
// This exists because of a crash: when Resolve renders with CUDA enabled,
// OFX::Image::getPixelData() returns *device* pointers, and every path that
// gives up and passes the source through was reading them on the CPU. That is
// an access violation on a device address, and it takes Resolve down with it.
//
// Pointers must already be offset to the top-left of the region being written.
// `sourceOffsetX/Y` place that region within the source image, whose extent is
// `sourceWidth/Height`; anything outside it is written as black rather than
// read out of bounds. Pitches are in floats. A null `source` fills black.
bool devicePassthrough(const float* source, size_t sourcePitch,
                       int sourceOffsetX, int sourceOffsetY,
                       int sourceWidth, int sourceHeight,
                       float* destination, size_t destinationPitch,
                       int width, int height, int components, void* stream);

}  // namespace iw3
