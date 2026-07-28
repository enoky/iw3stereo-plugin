// Checks the MonoBW CUDA kernels against the same reference data the CPU core
// is checked against.
//
// The kernels call straight into monobw_math.h, which is what test_pipeline.cpp
// validates on the CPU against Python that matches stock iw3 at difference 0.
// That shared header is the argument that the GPU path is correct; this is the
// evidence. Without it the argument rests on the kernels being wired up right,
// which is exactly the part a header cannot guarantee.
//
//     ofx\build\tests\test_monobw_gpu.exe tests\cpp\pipeline_reference.bin

#include "monobw_gpu.h"

#include "reference_data.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

int failures = 0;
int checks = 0;

float* upload(const float* data, size_t count)
{
    float* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(float)) != cudaSuccess)
    {
        return nullptr;
    }
    if (cudaMemcpy(device, data, count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(device);
        return nullptr;
    }
    return device;
}

void download(std::vector<float>& out, const float* device, size_t count)
{
    out.resize(count);
    cudaMemcpy(out.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost);
}

void expectClose(const std::string& name, const std::vector<float>& got,
                 const std::vector<float>& want, double tolerance)
{
    ++checks;
    if (got.size() != want.size())
    {
        std::printf("  FAIL %-46s size %zu != %zu\n", name.c_str(), got.size(), want.size());
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
        std::printf("  FAIL %-46s max abs diff %.3e > %.1e\n", name.c_str(), worst, tolerance);
        ++failures;
    }
    else
    {
        std::printf("  ok   %-46s max abs diff %.3e\n", name.c_str(), worst);
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

    iw3::MonoBwGpu gpu;
    int ran = 0;

    for (const iw3test::Case& entry : cases)
    {
        if (entry.name.rfind("monobw_forward_", 0) != 0)
        {
            continue;
        }
        ++ran;

        const int width = entry.ints[0], height = entry.ints[1];
        const int depthWidth = entry.ints[2], depthHeight = entry.ints[3];
        const bool border = entry.ints[4] != 0;
        const int fixMask = entry.ints[5];
        const double divergence = double(entry.ints[6]) / 1000.0;
        const double convergence = double(entry.ints[7]) / 1000.0;

        const size_t pixels = size_t(width) * size_t(height);
        const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);

        float* image = upload(entry.inputs.data(), pixels * 3);
        float* depth = upload(entry.inputs.data() + pixels * 3, depthPixels);
        if (!image || !depth)
        {
            std::printf("  FAIL %s: upload failed\n", entry.name.c_str());
            ++failures;
            ++checks;
            continue;
        }

        if (!gpu.prepare(width, height, depthWidth, depthHeight))
        {
            std::printf("  FAIL %s: %s\n", entry.name.c_str(), gpu.error().c_str());
            ++failures;
            ++checks;
            cudaFree(image);
            cudaFree(depth);
            continue;
        }
        gpu.forward(image, depth, divergence, convergence, border, fixMask, nullptr);
        cudaDeviceSynchronize();

        if (!gpu.ok())
        {
            std::printf("  FAIL %s: %s\n", entry.name.c_str(), gpu.error().c_str());
            ++failures;
            ++checks;
            cudaFree(image);
            cudaFree(depth);
            continue;
        }

        std::vector<float> eye, mask;
        download(eye, gpu.eyeDevice(), pixels * 3);
        download(mask, gpu.maskDevice(), pixels);

        const std::vector<float> wantEye(entry.outputs.begin(),
                                         entry.outputs.begin() + ptrdiff_t(pixels * 3));
        const std::vector<float> wantMask(entry.outputs.begin() + ptrdiff_t(pixels * 3),
                                          entry.outputs.end());

        expectClose(entry.name + "/eye", eye, wantEye, 1e-5);

        // The mask is a threshold on a difference, so a tolerance says nothing
        // about it: a pixel is either on the right side of the line or it is
        // not. Count disagreements.
        ++checks;
        size_t wrong = 0;
        for (size_t i = 0; i < mask.size() && i < wantMask.size(); ++i)
        {
            if (mask[i] != wantMask[i]) ++wrong;
        }
        if (mask.size() != wantMask.size() || wrong != 0)
        {
            std::printf("  FAIL %-46s %zu of %zu mask pixels differ\n",
                        (entry.name + "/mask").c_str(), wrong, wantMask.size());
            ++failures;
        }
        else
        {
            size_t set = 0;
            for (float value : wantMask) if (value != 0.0f) ++set;
            std::printf("  ok   %-46s %zu of %zu mask pixels, exact\n",
                        (entry.name + "/mask").c_str(), set, wantMask.size());
        }

        cudaFree(image);
        cudaFree(depth);
    }

    // The mask morphology. Its reference cases carry the raw mask directly
    // rather than the image and depth that produced it, so this drives
    // preprocessMask with an uploaded mask; the depth geometry passed to
    // prepare() is unused by this path and is kept small.
    for (const iw3test::Case& entry : cases)
    {
        if (entry.name.rfind("mask_preprocess_", 0) != 0)
        {
            continue;
        }
        ++ran;

        const int width = entry.ints[0], height = entry.ints[1];
        const int inner = entry.ints[2], outer = entry.ints[3];
        const int baseWidth = entry.ints[4];
        const size_t pixels = size_t(width) * size_t(height);

        float* raw = upload(entry.inputs.data(), pixels);
        if (!raw || !gpu.prepare(width, height, 8, 8))
        {
            std::printf("  FAIL %s: setup failed\n", entry.name.c_str());
            ++failures;
            ++checks;
            if (raw) cudaFree(raw);
            continue;
        }

        gpu.preprocessMask(raw, inner, outer, baseWidth, nullptr);
        cudaDeviceSynchronize();

        std::vector<float> got;
        download(got, gpu.processedMaskDevice(), pixels);

        // A mask is 0 or 1 and every operation on it is a max, a min or an or.
        // Nothing rounds, so the bar is equality.
        ++checks;
        size_t wrong = 0;
        for (size_t i = 0; i < got.size() && i < entry.outputs.size(); ++i)
        {
            if (got[i] != entry.outputs[i]) ++wrong;
        }
        if (got.size() != entry.outputs.size() || wrong != 0)
        {
            std::printf("  FAIL %-46s %zu of %zu pixels differ\n",
                        entry.name.c_str(), wrong, entry.outputs.size());
            ++failures;
        }
        else
        {
            size_t set = 0;
            for (float value : entry.outputs) if (value != 0.0f) ++set;
            std::printf("  ok   %-46s %zu of %zu pixels, exact\n",
                        entry.name.c_str(), set, entry.outputs.size());
        }

        cudaFree(raw);
    }

    if (ran == 0)
    {
        std::printf("no monobw cases in %s -- regenerate it\n", path.c_str());
        return 2;
    }

    // Timing, at the geometry the plugin actually runs: an HD frame with depth
    // at the model working size autoDepthSize() picks. Reported rather than
    // asserted -- a timing assertion on someone else's GPU is a flaky test.
    //
    // The number to watch is whether one-thread-per-row is fast enough to keep
    // the arithmetic in CUDA rather than emulating a scan in ONNX. The row scan
    // does not coalesce, which is the design's one known cost.
    // Three geometries, chosen to separate the two kernels rather than just
    // report a total: holding the depth fixed while shrinking the frame isolates
    // the per-pixel sampling, and holding the frame fixed while shrinking the
    // depth isolates the row scan.
    const struct { int width, height, depthWidth, depthHeight; } timings[] = {
        {1920, 1036, 938, 392},
        {384, 216, 938, 392},    // same depth, 1/25 the pixels
        {1920, 1036, 192, 108},  // same frame, 1/20 the depth
    };
    for (const auto& config : timings)
    {
        const int width = config.width, height = config.height;
        const int depthWidth = config.depthWidth, depthHeight = config.depthHeight;
        std::vector<float> image(size_t(width) * size_t(height) * 3, 0.5f);
        std::vector<float> depthHost(size_t(depthWidth) * size_t(depthHeight));
        for (size_t i = 0; i < depthHost.size(); ++i)
        {
            depthHost[i] = float((i * 37) % 1000) / 1000.0f;
        }
        float* deviceImage = upload(image.data(), image.size());
        float* deviceDepth = upload(depthHost.data(), depthHost.size());
        if (deviceImage && deviceDepth && gpu.prepare(width, height, depthWidth, depthHeight))
        {
            // The whole CUDA half of an eye: warp, mask, and morphology.
            const auto oneEye = [&]()
            {
                gpu.forward(deviceImage, deviceDepth, 2.0, 0.5, false, 1, nullptr);
                gpu.preprocessMask(gpu.maskDevice(), 2, 2, depthWidth, nullptr);
            };
            for (int i = 0; i < 5; ++i)
            {
                oneEye();
            }
            cudaDeviceSynchronize();

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            const int iterations = 50;
            cudaEventRecord(start);
            for (int i = 0; i < iterations; ++i)
            {
                oneEye();
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::printf("  %dx%d frame, %dx%d depth: %.3f ms per eye\n",
                        width, height, depthWidth, depthHeight,
                        double(milliseconds) / double(iterations));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        cudaFree(deviceImage);
        cudaFree(deviceDepth);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
