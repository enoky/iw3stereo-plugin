// MonoBW on the GPU: the forward warp and its hole mask, in device memory.
//
// This is the half of monobw_inpaint that cannot be an ONNX graph. torch.export
// handles it, but ONNX has no operator for either torch.cummax or
// torch.searchsorted, and those two are not incidental to the algorithm -- the
// cummax is the monotonisation that makes the mapping invertible and the
// searchsorted is the inversion. docs/monobw-inpaint.md has the evidence.
//
// The arithmetic is not reimplemented here. monobw_math.h is compiled for the
// device, so these kernels run the same lines tests/cpp/test_pipeline.cpp
// validates on the CPU against Python that matches stock iw3 at difference 0.
// tests/cpp/test_monobw_gpu.cu then runs this against the same reference data,
// so the GPU path is covered rather than assumed.

#pragma once

#include <string>

namespace iw3
{

class MonoBwGpu
{
public:
    MonoBwGpu() = default;
    ~MonoBwGpu();

    MonoBwGpu(const MonoBwGpu&) = delete;
    MonoBwGpu& operator=(const MonoBwGpu&) = delete;

    // Allocates for this frame geometry. Cheap when nothing changed.
    bool prepare(int width, int height, int depthWidth, int depthHeight);

    // `image` is three planes of width * height, `depth` is depthWidth *
    // depthHeight, both device pointers. Results land in eyeDevice() and
    // maskDevice().
    //
    // `fixScreenBorderMask` is iw3's: 0 leaves the mask alone, 1 clears the
    // uninpaintable side, 2 clears both. It applies only when
    // preserveScreenBorder is off. The image model uses 1.
    void forward(const float* image, const float* depth,
                 double divergence, double convergence,
                 bool preserveScreenBorder, int fixScreenBorderMask,
                 void* stream);

    // The warped eye, three planes of width * height.
    const float* eyeDevice() const { return _eye; }
    // The hole mask, width * height, 0.0 or 1.0.
    const float* maskDevice() const { return _mask; }

    bool ok() const { return _error.empty(); }
    const std::string& error() const { return _error; }

private:
    bool allocate(float** pointer, size_t floats, size_t& held);

    int _width = 0, _height = 0, _depthWidth = 0, _depthHeight = 0;

    float* _gridX = nullptr;    // depthWidth * depthHeight
    float* _gridY = nullptr;    // depthHeight
    float* _scratch = nullptr;  // 3 * depthWidth * depthHeight: dest, moved, smoothed
    float* _eye = nullptr;      // 3 * width * height
    float* _mask = nullptr;     // width * height

    size_t _gridXHeld = 0, _gridYHeld = 0, _scratchHeld = 0, _eyeHeld = 0, _maskHeld = 0;

    std::string _error;
};

}  // namespace iw3
