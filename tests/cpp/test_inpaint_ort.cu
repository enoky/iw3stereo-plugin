// The plugin's monobw_inpaint render path, outside Resolve.
//
// Everything this runs is the production code: OrtRuntime opening the same
// three graphs from the same bundle, MonoBwGpu producing the eye and mask on
// the device, and runInpaintDevice() binding those device pointers. If the
// plugin fails in Resolve and this passes, the fault is in the OFX glue; if
// this fails too, the fault is here and can be found without a host.
//
//     ofx\build\tests\test_inpaint_ort.exe [ort-dir] [model-dir]
//
// The two directories are separate because the graphs under test are not
// always staged into a bundle yet -- the ONNX Runtime comes from an installed
// bundle, the graphs from models/.

#include "monobw_gpu.h"
#include "ort_runtime.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

void dump(iw3::OrtRuntime& ort, const char* what)
{
    const std::vector<std::string> lines = ort.takeNewReport();
    if (lines.empty())
    {
        return;
    }
    std::printf("  -- %s --\n", what);
    for (const std::string& line : lines)
    {
        std::printf("     %s\n", line.c_str());
    }
}

float* upload(const std::vector<float>& data)
{
    float* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), data.size() * sizeof(float)) != cudaSuccess)
    {
        return nullptr;
    }
    cudaMemcpy(device, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return device;
}

std::wstring widen(const std::string& text)
{
    return std::wstring(text.begin(), text.end());
}

// IEEE half to float, on the host. Written out rather than borrowed from
// cuda_fp16 because whether __half2float is host-callable varies by version,
// and this is a dozen lines.
float halfToFloat(unsigned short bits)
{
    const unsigned sign = unsigned(bits >> 15) << 31;
    const unsigned exponent = (bits >> 10) & 0x1fu;
    const unsigned mantissa = bits & 0x3ffu;
    unsigned out = 0;
    if (exponent == 0)
    {
        out = mantissa ? sign | ((127u - 15u + 1u) << 23) | (mantissa << 13) : sign;
    }
    else if (exponent == 31)
    {
        out = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        out = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
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

    const std::string ortDir = argc > 1 ? argv[1]
        : "C:\\Program Files\\Common Files\\OFX\\Plugins\\iw3stereo.ofx.bundle\\Contents\\Win64";
    const std::string modelDir = argc > 2 ? argv[2] : "models";
    const std::wstring runtimeDir = widen(ortDir);
    const std::wstring models = widen(modelDir);

    iw3::OrtRuntime ort;
    const std::vector<std::wstring> graphs = {
        models + L"\\stereo_warp.onnx",
        models + L"\\stereo_warp_v3.onnx",
        models + L"\\light_inpaint_v1.onnx",
        models + L"\\light_video_inpaint_v1.onnx",
    };
    // The same per-graph session options the plugin uses. Without them the
    // twelve-frame graph takes 15.7 GiB and sixteen seconds instead of 9.0 and
    // a quarter of a second, and a reproduction that does not reproduce the
    // configuration is not one.
    const std::vector<bool> conserve = {false, false, true, true};
    const bool opened = ort.open(runtimeDir + L"\\ort", graphs, true, conserve);
    dump(ort, "bring-up");
    if (!opened || ort.modelCount() < 4)
    {
        std::printf("FAIL: only %zu graph(s) opened\n", ort.modelCount());
        return 2;
    }
    std::printf("provider %s, %zu graphs, inpaint outputs %zu, deviceCapable %d\n\n",
                ort.provider().c_str(), ort.modelCount(), ort.outputCount(2),
                ort.deviceCapable() ? 1 : 0);

    // The geometry from the user's log: an HD frame with depth at the model
    // working size.
    const struct { int width, height, depthWidth, depthHeight; } cases[] = {
        {1920, 1036, 714, 392},
        {384, 216, 192, 108},
    };

    int failures = 0;
    for (const auto& c : cases)
    {
        const size_t pixels = size_t(c.width) * size_t(c.height);
        std::vector<float> image(pixels * 3, 0.5f);
        std::vector<float> depth(size_t(c.depthWidth) * size_t(c.depthHeight));
        for (size_t i = 0; i < depth.size(); ++i)
        {
            depth[i] = float((i * 37) % 1000) / 1000.0f;
        }
        float* deviceImage = upload(image);
        float* deviceDepth = upload(depth);

        iw3::MonoBwGpu gpu;
        if (!gpu.prepare(c.width, c.height, c.depthWidth, c.depthHeight))
        {
            std::printf("FAIL %dx%d: prepare: %s\n", c.width, c.height, gpu.error().c_str());
            ++failures;
            continue;
        }

        for (int pass = 0; pass < 2; ++pass)
        {
            const bool rightEye = pass == 1;
            if (!gpu.prepareEye(deviceImage, deviceDepth, rightEye, 2.0, 0.5, false,
                                0, 0, c.depthWidth, nullptr))
            {
                std::printf("FAIL %dx%d %s: prepareEye: %s\n", c.width, c.height,
                            rightEye ? "right" : "left", gpu.error().c_str());
                ++failures;
                break;
            }
            cudaDeviceSynchronize();

            const int64_t shape[4] = {1, 3, c.height, c.width};
            const float* filled = nullptr;
            const bool half = ort.inputIsHalf(2);
            if (half)
            {
                gpu.castEyeAndMaskToHalf(nullptr);
                cudaDeviceSynchronize();
            }
            const bool ok = ort.runInpaintDevice(
                2,
                half ? reinterpret_cast<const float*>(gpu.inpaintEyeHalfDevice())
                     : gpu.inpaintEyeDevice(),
                half ? reinterpret_cast<const float*>(gpu.processedMaskHalfDevice())
                     : gpu.processedMaskDevice(),
                shape, &filled);
            if (ok && half)
            {
                filled = gpu.halfToFloat(filled, nullptr);
                cudaDeviceSynchronize();
            }
            std::printf("%s %dx%d %s eye: runInpaintDevice %s (%.2f ms)\n",
                        ok ? "ok  " : "FAIL", c.width, c.height,
                        rightEye ? "right" : "left ",
                        ok ? "returned a buffer" : "FAILED",
                        ort.lastRunMilliseconds());
            dump(ort, "ort");
            if (!ok)
            {
                ++failures;
                break;
            }

            // Read one pixel back, so a silently-wrong buffer is not counted a
            // success.
            float sample[3] = {-1.0f, -1.0f, -1.0f};
            if (cudaMemcpy(sample, filled, sizeof(sample), cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                std::printf("FAIL: output is not readable device memory\n");
                ++failures;
                break;
            }
            std::printf("     first pixel %.4f %.4f %.4f\n", sample[0], sample[1], sample[2]);
            (void)gpu.finishEye(filled, rightEye, nullptr);
            cudaDeviceSynchronize();
        }

        cudaFree(deviceImage);
        cudaFree(deviceDepth);
    }

    // The twelve-frame window. Same call shape the plugin will make: one warp
    // per frame into a slice of a sequence buffer, then one inference over the
    // whole window.
    {
        const int width = 1920, height = 1036;
        const int depthWidth = 714, depthHeight = 392;
        const size_t pixels = size_t(width) * size_t(height);
        const int seq = 12;

        std::vector<float> image(pixels * 3, 0.5f);
        std::vector<float> depth(size_t(depthWidth) * size_t(depthHeight));
        for (size_t i = 0; i < depth.size(); ++i)
        {
            depth[i] = float((i * 37) % 1000) / 1000.0f;
        }
        float* deviceImage = upload(image);
        float* deviceDepth = upload(depth);

        float* eyes = nullptr;
        float* masks = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&eyes), pixels * 3 * size_t(seq) * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&masks), pixels * size_t(seq) * sizeof(float));
        // Allocated at float width, which is more than a half window needs.

        iw3::MonoBwGpu gpu;
        if (deviceImage && deviceDepth && eyes && masks &&
            gpu.prepare(width, height, depthWidth, depthHeight))
        {
            // Half, so the sequence buffers are half-width too.
            const size_t element = ort.inputIsHalf(3) ? sizeof(uint16_t) : sizeof(float);
            for (int frame = 0; frame < seq; ++frame)
            {
                gpu.prepareEye(deviceImage, deviceDepth, false, 2.0, 0.5, false,
                               0, 0, depthWidth, nullptr);
                gpu.castEyeAndMaskToHalf(nullptr);
                cudaDeviceSynchronize();
                cudaMemcpy(reinterpret_cast<char*>(eyes) + size_t(frame) * pixels * 3 * element,
                           gpu.inpaintEyeHalfDevice(), pixels * 3 * element,
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(reinterpret_cast<char*>(masks) + size_t(frame) * pixels * element,
                           gpu.processedMaskHalfDevice(), pixels * element,
                           cudaMemcpyDeviceToDevice);
            }
            cudaDeviceSynchronize();

            const int64_t shape[4] = {seq, 3, height, width};
            const float* filled = nullptr;
            const bool ok = ort.runInpaintDevice(3, eyes, masks, shape, &filled);
            std::printf("%s video window %dx%d x%d frames: %s (%.2f ms, %.2f ms/frame)\n",
                        ok ? "ok  " : "FAIL", width, height, seq,
                        ok ? "returned a buffer" : "FAILED",
                        ort.lastRunMilliseconds(), ort.lastRunMilliseconds() / seq);
            dump(ort, "ort");
            if (!ok)
            {
                ++failures;
            }
            else
            {
                // The graph is half, so this reads halves. Printing them as
                // floats showed 0.0001 for an input that was 0.5 everywhere,
                // which looked like a pipeline fault and was a decoding one.
                // Worth a real check rather than a decorative print.
                // A row's mean rather than one pixel. The input is 0.5
                // everywhere, so most of the frame is passed straight through
                // and the mean lands near 0.5 whatever the filled pixels do --
                // whereas reading these halves as floats, which is what this
                // did at first, gives about 1e-4 and would be caught.
                //
                // Not a single pixel: prepareEye mirrors its output, so pixel
                // zero of this buffer is the far edge of the warp, which is
                // exactly where the holes are. It reads 0.588 and that is
                // correct.
                const size_t sampled = 512;
                std::vector<unsigned short> bits(sampled, 0);
                if (cudaMemcpy(bits.data(), filled, sampled * sizeof(unsigned short),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                {
                    std::printf("FAIL: video output is not readable device memory\n");
                    ++failures;
                }
                else
                {
                    double total = 0.0;
                    for (unsigned short value : bits)
                    {
                        total += double(halfToFloat(value));
                    }
                    const double mean = total / double(sampled);
                    std::printf("     first %zu samples mean %.4f (input was 0.5)\n",
                                sampled, mean);
                    if (!(mean > 0.4 && mean < 0.6))
                    {
                        std::printf("FAIL: mean %.4f is not near the 0.5 that went in\n", mean);
                        ++failures;
                    }
                }
            }
        }
        else
        {
            std::printf("FAIL: video window setup failed\n");
            ++failures;
        }

        if (eyes) cudaFree(eyes);
        if (masks) cudaFree(masks);
        cudaFree(deviceImage);
        cudaFree(deviceDepth);
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
