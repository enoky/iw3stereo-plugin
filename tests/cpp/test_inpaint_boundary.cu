// The reduce-and-composite either side of an inpaint graph, against Python.
//
// Until this existed, nothing compared these kernels' output to anything. The
// Python golden tests do cover max_width at difference 0, but they test the
// *port*, which follows iw3 and resizes the whole frame before warping; these
// kernels are the plugin's own departure from that and were checked only by
// test_inpaint_ort, which asserts that a buffer came back and is not flat.
//
// It models the kernels as they are today, deliberately.
// docs/inpaint-downscale-plan.md changes them next -- a hole-aware downscale and
// a bicubic lift -- and a harness proven against the current behaviour first is
// what turns each of those into a visible, intended diff rather than a surprise.
//
// The kernels are launched directly rather than through MonoBwGpu or MlbwGpu,
// because what is under test is the boundary itself and both of those would
// require a warp to have run first.
//
//     ofx\build\tests\test_inpaint_boundary.exe tests\cpp\pipeline_reference.bin

#include "inpaint_boundary.cuh"

#include "reference_data.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

int failures = 0;
int checks = 0;
constexpr int kBlock = 16;

dim3 grid2d(int width, int height)
{
    return dim3(unsigned((width + kBlock - 1) / kBlock),
                unsigned((height + kBlock - 1) / kBlock));
}

template <typename T>
T* upload(const T* data, size_t count)
{
    T* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T)) != cudaSuccess)
    {
        return nullptr;
    }
    cudaMemcpy(device, data, count * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

void expectClose(const std::string& name, const std::vector<float>& got,
                 const std::vector<float>& want, double tolerance)
{
    ++checks;
    if (got.size() != want.size())
    {
        std::printf("  FAIL %-44s size %zu != %zu\n", name.c_str(), got.size(), want.size());
        ++failures;
        return;
    }
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
    {
        worst = std::max(worst, double(std::abs(got[i] - want[i])));
    }
    if (worst > tolerance)
    {
        std::printf("  FAIL %-44s max abs diff %.3e > %.1e\n", name.c_str(), worst, tolerance);
        ++failures;
    }
    else
    {
        std::printf("  ok   %-44s max abs diff %.3e\n", name.c_str(), worst);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        std::printf("no CUDA device; skipping\n");
        return 0;
    }

    const std::string path = argc > 1 ? argv[1] : "tests/cpp/pipeline_reference.bin";
    std::vector<iw3test::Case> cases;
    if (!iw3test::load(path, cases))
    {
        return 2;
    }

    int ran = 0;
    for (const iw3test::Case& entry : cases)
    {
        if (entry.name.rfind("inpaint_boundary_", 0) != 0)
        {
            continue;
        }
        ++ran;

        const int width = entry.ints[0], height = entry.ints[1];
        const int outWidth = entry.ints[2], outHeight = entry.ints[3];
        const size_t pixels = size_t(width) * size_t(height);
        const size_t outPixels = size_t(outWidth) * size_t(outHeight);
        const bool reduced = (outWidth != width || outHeight != height);

        const float* eye = entry.inputs.data();
        const float* mask = eye + pixels * 3;
        const float* filled = mask + pixels;

        // The graph's output arrives as __half, so the stand-in does too.
        std::vector<__half> filledHalf(outPixels * 3);
        for (size_t i = 0; i < filledHalf.size(); ++i)
        {
            filledHalf[i] = __float2half(filled[i]);
        }

        float* eyeDevice = upload(eye, pixels * 3);
        float* maskDevice = upload(mask, pixels);
        __half* filledDevice = upload(filledHalf.data(), filledHalf.size());
        __half* eyeHalf = nullptr;
        __half* maskHalf = nullptr;
        float* blur = nullptr;
        float* blurTmp = nullptr;
        float* out = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&eyeHalf), outPixels * 3 * sizeof(__half));
        cudaMalloc(reinterpret_cast<void**>(&maskHalf), outPixels * sizeof(__half));
        cudaMalloc(reinterpret_cast<void**>(&blur), pixels * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&blurTmp), pixels * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&out), pixels * 3 * sizeof(float));

        if (!eyeDevice || !maskDevice || !filledDevice || !eyeHalf || !maskHalf || !out)
        {
            std::printf("  FAIL %-44s allocation\n", entry.name.c_str());
            ++failures;
            continue;
        }

        // Mirrors prepareInpaintInput's branch exactly, including the fact that
        // the unreduced case is a straight cast and not a one-to-one box.
        constexpr int kThreads = 256;
        if (!reduced)
        {
            iw3::toHalfKernel<<<unsigned((pixels * 3 + kThreads - 1) / kThreads), kThreads>>>(
                eyeDevice, pixels * 3, eyeHalf);
            iw3::toHalfKernel<<<unsigned((pixels + kThreads - 1) / kThreads), kThreads>>>(
                maskDevice, pixels, maskHalf);
        }
        else
        {
            iw3::downscaleEyeKernel<<<grid2d(outWidth, outHeight), dim3(kBlock, kBlock)>>>(
                eyeDevice, width, height, maskDevice, outWidth, outHeight, eyeHalf);
            iw3::downscaleMaskKernel<<<grid2d(outWidth, outHeight), dim3(kBlock, kBlock)>>>(
                maskDevice, width, height, outWidth, outHeight, maskHalf);
            iw3::maskBlurKernel<<<grid2d(width, height), dim3(kBlock, kBlock)>>>(
                maskDevice, width, height, 1, blurTmp);
            iw3::maskBlurKernel<<<grid2d(width, height), dim3(kBlock, kBlock)>>>(
                blurTmp, width, height, 0, blur);
        }
        cudaDeviceSynchronize();

        std::vector<__half> gotEyeHalf(outPixels * 3), gotMaskHalf(outPixels);
        cudaMemcpy(gotEyeHalf.data(), eyeHalf, gotEyeHalf.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(gotMaskHalf.data(), maskHalf, gotMaskHalf.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        std::vector<float> gotEye(gotEyeHalf.size()), gotMask(gotMaskHalf.size());
        for (size_t i = 0; i < gotEye.size(); ++i) gotEye[i] = __half2float(gotEyeHalf[i]);
        for (size_t i = 0; i < gotMask.size(); ++i) gotMask[i] = __half2float(gotMaskHalf[i]);

        const float* wantEye = entry.outputs.data();
        const float* wantMask = wantEye + outPixels * 3;
        const float* wantOut = wantMask + outPixels;

        // Half precision either side, so one ulp near 1.0 is about 1e-3. The box
        // sums sequentially here and pairwise in numpy, and either order can tip
        // the rounding -- an exact bar would be testing numpy's summation, not
        // this kernel.
        expectClose(entry.name + "/eye",
                    gotEye, std::vector<float>(wantEye, wantEye + outPixels * 3), 1e-3);
        // The mask reduces by max, which selects rather than sums, so it is exact.
        expectClose(entry.name + "/mask",
                    gotMask, std::vector<float>(wantMask, wantMask + outPixels), 0.0);

        if (!reduced)
        {
            iw3::fromHalfKernel<<<unsigned((pixels * 3 + kThreads - 1) / kThreads), kThreads>>>(
                filledDevice, pixels * 3, out);
        }
        else
        {
            iw3::compositeUpscaledKernel<<<grid2d(width, height), dim3(kBlock, kBlock)>>>(
                eyeDevice, filledDevice, blur, maskDevice,
                width, height, outWidth, outHeight, out);
        }
        cudaDeviceSynchronize();

        std::vector<float> gotOut(pixels * 3);
        cudaMemcpy(gotOut.data(), out, gotOut.size() * sizeof(float), cudaMemcpyDeviceToHost);
        expectClose(entry.name + "/composite",
                    gotOut, std::vector<float>(wantOut, wantOut + pixels * 3), 1e-5);

        for (void* p : {(void*)eyeDevice, (void*)maskDevice, (void*)filledDevice,
                        (void*)eyeHalf, (void*)maskHalf, (void*)blur, (void*)blurTmp, (void*)out})
        {
            if (p) cudaFree(p);
        }
    }

    if (ran == 0)
    {
        std::printf("no inpaint_boundary cases in %s -- regenerate with "
                    "tools/dump_pipeline_reference.py\n", path.c_str());
        return 2;
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
