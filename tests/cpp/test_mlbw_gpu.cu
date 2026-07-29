// Checks the mlbw_l2_inpaint CUDA kernels against the same reference data the
// CPU core is checked against.
//
// The kernels call straight into mlbw_math.h, which test_pipeline.cpp validates
// on the CPU against Python that matches stock iw3 at difference 0. That shared
// header is the argument that the GPU path is correct; this is the evidence.
// Without it the argument rests on the kernels being wired up right, which is
// exactly the part a header cannot guarantee.
//
// The mlbw_eye_* cases are the ones that only exist here. They record what the
// inpaint network is actually handed for each eye, which pins down the
// mirroring -- and the mirroring has no CPU equivalent to check it against,
// because monobw hides its own inside a kernel's write index while this path
// has to do it explicitly. Get it wrong and the result is a picture rather than
// an error.
//
//     ofx\build\tests\test_mlbw_gpu.exe tests\cpp\pipeline_reference.bin

#include "mlbw_gpu.h"
#include "stereo_pipeline.h"

#include "reference_data.h"

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

// A mask is 0 or 1 and the last thing done to it is a threshold, so a tolerance
// would let a whole misplaced pixel through. Count disagreements instead.
void expectExact(const std::string& name, const std::vector<float>& got,
                 const std::vector<float>& want)
{
    ++checks;
    size_t wrong = 0;
    for (size_t i = 0; i < got.size() && i < want.size(); ++i)
    {
        if (got[i] != want[i]) ++wrong;
    }
    if (got.size() != want.size() || wrong != 0)
    {
        std::printf("  FAIL %-46s %zu of %zu pixels differ\n",
                    name.c_str(), wrong, want.size());
        ++failures;
        return;
    }
    size_t set = 0;
    for (float value : want) if (value != 0.0f) ++set;
    std::printf("  ok   %-46s %zu of %zu pixels, exact\n", name.c_str(), set, want.size());
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

    iw3::MlbwGpu gpu;
    int ran = 0;

    for (const iw3test::Case& entry : cases)
    {
        const bool isEye = entry.name.rfind("mlbw_eye_", 0) == 0;
        const bool isWarp = entry.name.rfind("mlbw_warp_", 0) == 0;
        if (!isEye && !isWarp)
        {
            continue;
        }
        ++ran;

        const int width = entry.ints[0], height = entry.ints[1];
        const int depthWidth = entry.ints[2], depthHeight = entry.ints[3];
        const size_t pixels = size_t(width) * size_t(height);
        const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);

        const float* image = entry.inputs.data();
        const float* delta = image + pixels * 3;
        const float* layerWeight = delta + depthPixels * 2;

        float* imageDevice = upload(image, pixels * 3);
        float* deltaDevice = upload(delta, depthPixels * 2);
        float* weightDevice = upload(layerWeight, depthPixels * 2);
        if (!imageDevice || !deltaDevice || !weightDevice)
        {
            std::printf("  FAIL %-46s upload failed\n", entry.name.c_str());
            ++failures;
            continue;
        }

        if (isWarp)
        {
            // The warp on its own, with no mirroring, against the CPU driver's
            // own case. The reference is in frame orientation, so the eye case
            // below is what covers the mirror.
            const float* logits = layerWeight + depthPixels * 2;
            (void)logits;

            // prepareEye always mirrors, so undo it here to compare against a
            // frame-oriented reference. Flipping the answer back is legitimate
            // for the warp -- unlike the mask, nothing downstream of it is
            // direction-dependent.
            if (!gpu.prepare(width, height, depthWidth, depthHeight))
            {
                std::printf("  FAIL %-46s prepare: %s\n", entry.name.c_str(),
                            gpu.error().c_str());
                ++failures;
            }
            else
            {
                std::vector<float> zeroLogits(depthPixels, 0.0f);
                float* logitsDevice = upload(zeroLogits.data(), depthPixels);
                gpu.prepareEye(imageDevice, deltaDevice, weightDevice, logitsDevice,
                               /*rightEye=*/false, 0, 0, nullptr);
                cudaDeviceSynchronize();

                std::vector<float> mirrored;
                download(mirrored, gpu.inpaintEyeDevice(), pixels * 3);
                std::vector<float> got(pixels * 3);
                for (int channel = 0; channel < 3; ++channel)
                {
                    for (int y = 0; y < height; ++y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            got[size_t(channel) * pixels + size_t(y) * size_t(width) + size_t(x)] =
                                mirrored[size_t(channel) * pixels + size_t(y) * size_t(width) +
                                         size_t(width - 1 - x)];
                        }
                    }
                }
                expectClose(entry.name, got, entry.outputs, 1e-5);
                if (logitsDevice) cudaFree(logitsDevice);
            }
        }
        else
        {
            const bool rightEye = entry.ints[4] != 0;
            const int inner = entry.ints[5], outer = entry.ints[6];
            const float* logits = layerWeight + depthPixels * 2;
            float* logitsDevice = upload(logits, depthPixels);

            if (!logitsDevice || !gpu.prepare(width, height, depthWidth, depthHeight))
            {
                std::printf("  FAIL %-46s prepare: %s\n", entry.name.c_str(),
                            gpu.error().c_str());
                ++failures;
            }
            else if (!gpu.prepareEye(imageDevice, deltaDevice, weightDevice, logitsDevice,
                                     rightEye, inner, outer, nullptr))
            {
                std::printf("  FAIL %-46s prepareEye: %s\n", entry.name.c_str(),
                            gpu.error().c_str());
                ++failures;
            }
            else
            {
                cudaDeviceSynchronize();
                std::vector<float> eye, mask;
                download(eye, gpu.inpaintEyeDevice(), pixels * 3);
                download(mask, gpu.processedMaskDevice(), pixels);

                std::vector<float> wantEye(entry.outputs.begin(),
                                           entry.outputs.begin() + ptrdiff_t(pixels * 3));
                std::vector<float> wantMask(entry.outputs.begin() + ptrdiff_t(pixels * 3),
                                            entry.outputs.end());
                expectClose(entry.name + "/eye", eye, wantEye, 1e-5);
                expectExact(entry.name + "/mask", mask, wantMask);

                // finishEye has to undo exactly what prepareEye did, or a
                // round trip through the network would come back mirrored. The
                // network is not here, so feed the eye straight back: for the
                // left that must return it to frame orientation, and for the
                // right it must not touch it.
                const float* restored = gpu.finishEye(gpu.inpaintEyeDevice(), rightEye, nullptr);
                cudaDeviceSynchronize();
                std::vector<float> back;
                download(back, restored, pixels * 3);

                std::vector<float> expected = eye;
                if (!rightEye)
                {
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        for (int y = 0; y < height; ++y)
                        {
                            for (int x = 0; x < width; ++x)
                            {
                                expected[size_t(channel) * pixels +
                                         size_t(y) * size_t(width) + size_t(x)] =
                                    eye[size_t(channel) * pixels +
                                        size_t(y) * size_t(width) + size_t(width - 1 - x)];
                            }
                        }
                    }
                }
                expectClose(entry.name + "/finish", back, expected, 0.0);
            }
            if (logitsDevice) cudaFree(logitsDevice);
        }

        cudaFree(imageDevice);
        cudaFree(deltaDevice);
        cudaFree(weightDevice);
    }

    if (ran == 0)
    {
        std::printf("no mlbw cases in %s -- regenerate with tools/dump_pipeline_reference.py\n",
                    path.c_str());
        return 2;
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
