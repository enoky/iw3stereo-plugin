// The plugin's inpaint render paths, outside Resolve.
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

#include "mlbw_gpu.h"
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
        models + L"\\mlbw_net.onnx",
    };
    // The same per-graph session options the plugin uses. Without them the
    // twelve-frame graph takes 15.7 GiB and sixteen seconds instead of 9.0 and
    // a quarter of a second, and a reproduction that does not reproduce the
    // configuration is not one.
    const std::vector<bool> conserve = {false, false, true, true, false};
    const bool opened = ort.open(runtimeDir + L"\\ort", graphs, true, conserve);
    dump(ort, "bring-up");
    if (!opened || ort.modelCount() < 5)
    {
        std::printf("FAIL: only %zu graph(s) opened\n", ort.modelCount());
        return 2;
    }
    std::printf("provider %s, %zu graphs, inpaint outputs %zu, deviceCapable %d\n\n",
                ort.provider().c_str(), ort.modelCount(), ort.outputCount(2),
                ort.deviceCapable() ? 1 : 0);

    // The geometry from the user's log: an HD frame with depth at the model
    // working size.
    // The last case exercises Inpaint Max Width: the graph runs at 960 wide and
    // the fill is composited back into the full-resolution eye, so the output
    // is still 1920 and everything outside a hole is untouched.
    const struct { int width, height, depthWidth, depthHeight, maxWidth; } cases[] = {
        {1920, 1036, 714, 392, 0},
        {384, 216, 192, 108, 0},
        {1920, 1036, 714, 392, 960},
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

            const float* filled = nullptr;
            const bool half = ort.inputIsHalf(2);
            gpu.prepareInpaintInput(c.maxWidth, nullptr);
            cudaDeviceSynchronize();
            const int64_t reduced[4] = {1, 3, gpu.inpaintHeight(), gpu.inpaintWidth()};
            const bool ok = ort.runInpaintDevice(
                2,
                half ? reinterpret_cast<const float*>(gpu.inpaintEyeHalfDevice())
                     : gpu.inpaintEyeDevice(),
                half ? reinterpret_cast<const float*>(gpu.processedMaskHalfDevice())
                     : gpu.processedMaskDevice(),
                reduced, &filled);
            if (ok)
            {
                filled = gpu.finishInpaintOutput(filled, nullptr);
                cudaDeviceSynchronize();
            }
            std::printf("%s frame %dx%d, inpaint %dx%d, %s eye: %s (%.2f ms)\n",
                        ok ? "ok  " : "FAIL", c.width, c.height,
                        gpu.inpaintWidth(), gpu.inpaintHeight(),
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
                gpu.prepareInpaintInput(0, nullptr);
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

    // Is the fill for frame N actually derived from frame N?
    //
    // The indexing survives three hops -- slot j of the window is absolute
    // firstFrame + j, the graph's output j pairs with its input j, and the cache
    // keeps slots kPad..kPad+kStride-1 -- and an off-by-one anywhere in that
    // chain would show as an inpaint that lags or leads the picture. Reasoning
    // says it lines up; this checks it, because two bugs in this path have
    // already looked fine on paper.
    //
    // Each frame of the window is a different flat grey. The warp of a constant
    // field is that constant and so is a fill of it, so whatever comes out of
    // slot j should still be frame j's grey. That makes the alignment readable
    // as a number.
    {
        const int width = 384, height = 216;
        const int depthWidth = 192, depthHeight = 108;
        const size_t pixels = size_t(width) * size_t(height);
        const int seq = iw3::MonoBwVideoGpu::kSequence;

        std::vector<float> depth(size_t(depthWidth) * size_t(depthHeight));
        for (size_t i = 0; i < depth.size(); ++i)
        {
            depth[i] = float((i * 37) % 1000) / 1000.0f;
        }
        float* deviceDepth = upload(depth);

        iw3::MonoBwGpu gpu;
        iw3::MonoBwVideoGpu video;
        std::vector<float> greys(size_t(seq), 0.0f);
        bool built = deviceDepth != nullptr &&
                     gpu.prepare(width, height, depthWidth, depthHeight) &&
                     video.prepare(width, height);

        for (int slot = 0; slot < seq && built; ++slot)
        {
            // Values well apart in fp16 and clear of 0 and 1.
            greys[size_t(slot)] = 0.2f + 0.05f * float(slot);
            std::vector<float> frame(pixels * 3, greys[size_t(slot)]);
            float* deviceFrame = upload(frame);
            built = deviceFrame != nullptr &&
                    gpu.prepareEye(deviceFrame, deviceDepth, false, 2.0, 0.5, false,
                                   0, 0, depthWidth, nullptr);
            if (built)
            {
                gpu.prepareInpaintInput(0, nullptr);
                video.storeFrame(slot, gpu.inpaintEyeHalfDevice(),
                                 gpu.processedMaskHalfDevice(), nullptr);
                cudaDeviceSynchronize();
            }
            if (deviceFrame) cudaFree(deviceFrame);
        }

        const int64_t alignShape[4] = {seq, 3, height, width};
        const float* aligned = nullptr;
        if (built && ort.runInpaintDevice(3, reinterpret_cast<const float*>(video.eyesDevice()),
                                          reinterpret_cast<const float*>(video.masksDevice()),
                                          alignShape, &aligned))
        {
            video.cacheOutput(false, aligned, nullptr);
            cudaDeviceSynchronize();

            std::printf("\n  frame alignment through the window:\n");
            int wrong = 0;
            for (int offset = 0; offset < iw3::MonoBwVideoGpu::kStride; ++offset)
            {
                // The middle of the frame, well away from the screen border the
                // mask fix clears and from the holes at the edges.
                const size_t middle = size_t(height / 2) * size_t(width) + size_t(width / 2);
                unsigned short bits = 0;
                cudaMemcpy(&bits,
                           static_cast<const char*>(video.cachedFrame(false, offset)) +
                               middle * sizeof(unsigned short),
                           sizeof(bits), cudaMemcpyDeviceToHost);
                const float got = halfToFloat(bits);
                const int expectedSlot = iw3::MonoBwVideoGpu::kPad + offset;
                const float want = greys[size_t(expectedSlot)];
                const bool ok = std::abs(got - want) < 0.02f;
                std::printf("    %s cache[%d] = %.3f, slot %d went in as %.3f\n",
                            ok ? "ok  " : "FAIL", offset, got, expectedSlot, want);
                if (!ok) ++wrong;
            }
            if (wrong)
            {
                std::printf("FAIL: %d of %d cached frames came from the wrong input frame\n",
                            wrong, iw3::MonoBwVideoGpu::kStride);
                ++failures;
            }
        }
        else
        {
            std::printf("FAIL: frame alignment case did not run\n");
            ++failures;
        }
        if (deviceDepth) cudaFree(deviceDepth);
    }

    // --- the mlbw_l2_inpaint path -------------------------------------------
    //
    // Two graphs a frame per eye: the warp network in slot 4, then the same
    // LightInpaintV1 fill in slot 2. The plugin does exactly this, so what is
    // under test is the sequencing as much as the arithmetic -- in particular
    // that the fill's bound output buffer, which the next call overwrites, is
    // consumed before the second eye runs.
    std::printf("\n-- mlbw_l2_inpaint --\n");
    if (ort.outputCount(4) != 3)
    {
        std::printf("FAIL: mlbw_net has %zu outputs, expected 3\n", ort.outputCount(4));
        ++failures;
    }
    else
    {
        const struct { int width, height, depthWidth, depthHeight, maxWidth; } mlbwCases[] = {
            {1920, 1036, 1920, 1036, 0},      // Full: no weight resize at all
            {1920, 1036, 1920, 1036, 1280},   // Inpaint Max Width: the fill runs small
            {384, 216, 192, 108, 0},          // depth smaller than the frame
            {384, 216, 384, 216, 0},
        };
        for (const auto& c : mlbwCases)
        {
            const size_t pixels = size_t(c.width) * size_t(c.height);
            const size_t depthPixels = size_t(c.depthWidth) * size_t(c.depthHeight);
            std::vector<float> image(pixels * 3);
            for (size_t i = 0; i < image.size(); ++i)
            {
                image[i] = float((i * 17) % 997) / 997.0f;
            }
            // The model's input tensor, built here rather than by the plugin's
            // kernel: this test is about the graph and kernel sequencing, and a
            // plausible tensor is enough for that.
            std::vector<float> x(depthPixels * 3);
            for (size_t i = 0; i < depthPixels; ++i)
            {
                x[i] = float((i * 37) % 1000) / 1000.0f;   // depth
                x[depthPixels + i] = 0.6f;                 // divergence feature
                x[2 * depthPixels + i] = -0.3f;            // convergence feature
            }
            float* deviceImage = upload(image);
            float* deviceX = upload(x);

            iw3::MlbwGpu gpu;
            if (!deviceImage || !deviceX ||
                !gpu.prepare(c.width, c.height, c.depthWidth, c.depthHeight))
            {
                std::printf("FAIL: %dx%d setup: %s\n", c.width, c.height, gpu.error().c_str());
                ++failures;
                if (deviceImage) cudaFree(deviceImage);
                if (deviceX) cudaFree(deviceX);
                continue;
            }

            const int64_t xShape[4] = {1, 3, c.depthHeight, c.depthWidth};
            bool ok = true;
            std::vector<float> eyes[2];
            for (int pass = 0; pass < 2 && ok; ++pass)
            {
                const bool rightEye = pass == 1;
                const float* delta = nullptr;
                const float* weight = nullptr;
                const float* logits = nullptr;
                if (!ort.runMlbwDevice(4, deviceX, xShape, &delta, &weight, &logits))
                {
                    dump(ort, "mlbw net");
                    std::printf("FAIL: %dx%d mlbw net inference\n", c.width, c.height);
                    ok = false;
                    break;
                }
                if (!gpu.prepareEye(deviceImage, delta, weight, logits, rightEye, 2, 3, nullptr))
                {
                    std::printf("FAIL: %dx%d prepareEye: %s\n", c.width, c.height,
                                gpu.error().c_str());
                    ok = false;
                    break;
                }
                cudaDeviceSynchronize();

                gpu.prepareInpaintInput(c.maxWidth, nullptr);
                cudaDeviceSynchronize();
                const int64_t shape[4] = {1, 3, gpu.inpaintHeight(), gpu.inpaintWidth()};
                const float* filled = nullptr;
                if (!ort.runInpaintDevice(
                        2, reinterpret_cast<const float*>(gpu.inpaintEyeHalfDevice()),
                        reinterpret_cast<const float*>(gpu.processedMaskHalfDevice()),
                        shape, &filled))
                {
                    dump(ort, "mlbw fill");
                    std::printf("FAIL: %dx%d fill inference\n", c.width, c.height);
                    ok = false;
                    break;
                }
                filled = gpu.finishInpaintOutput(filled, nullptr);
                const float* eye = gpu.finishEye(filled, rightEye, nullptr);
                cudaDeviceSynchronize();

                // Read the frame back and insist it is a picture rather than a
                // buffer nothing wrote. Every bug on this path in the monobw
                // work produced an image, so "it ran without an error" is not
                // the test: a flat or black frame has to fail here.
                eyes[pass].resize(pixels * 3);
                cudaMemcpy(eyes[pass].data(), eye, eyes[pass].size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                float low = eyes[pass][0], high = eyes[pass][0];
                double sum = 0.0;
                for (float value : eyes[pass])
                {
                    sum += value;
                    low = value < low ? value : low;
                    high = value > high ? value : high;
                }
                const float mean = float(sum / double(eyes[pass].size()));
                if (!(high - low > 0.05f) || mean < 0.01f)
                {
                    std::printf("FAIL: %dx%d %s eye is flat (%.4f..%.4f, mean %.4f)\n",
                                c.width, c.height, rightEye ? "right" : "left", low, high, mean);
                    ok = false;
                }
            }

            if (ok)
            {
                // The two eyes must differ. Both are warped from the same frame
                // with the same network outputs here, so if the mirroring were a
                // no-op they would come back identical -- which in Resolve looks
                // like a working plugin producing no 3D at all.
                size_t same = 0;
                for (size_t i = 0; i < eyes[0].size(); ++i)
                {
                    if (eyes[0][i] == eyes[1][i]) ++same;
                }
                const double fraction = double(same) / double(eyes[0].size());
                if (fraction > 0.98)
                {
                    std::printf("FAIL: %dx%d the two eyes agree on %.1f%% of samples; "
                                "the mirroring is a no-op\n",
                                c.width, c.height, 100.0 * fraction);
                    ++failures;
                }
                else
                {
                    std::printf("    ok   %4dx%-4d depth %4dx%-4d maxWidth %4d  "
                                "fill at %dx%d, eyes differ on %.1f%%, fill %.1f ms\n",
                                c.width, c.height, c.depthWidth, c.depthHeight, c.maxWidth,
                                gpu.inpaintWidth(), gpu.inpaintHeight(),
                                100.0 * (1.0 - fraction), ort.lastRunMilliseconds());
                }
            }
            else
            {
                ++failures;
            }
            cudaFree(deviceImage);
            cudaFree(deviceX);
        }
    }

    // --- mlbw_l2_inpaint_video ----------------------------------------------
    //
    // The window build and the cache mapping, which is where the monobw work's
    // periodic glitch lived: the frame that builds a window was composited
    // against slot eleven's warp rather than its own, one frame in six.
    //
    // Each slot gets a distinct flat grey, so the cached output can be traced
    // back to the slot it came from. The alignment machinery is shared with
    // monobw and already covered above; what is new here is the mlbw window
    // loop feeding it.
    std::printf("\n-- mlbw_l2_inpaint_video --\n");
    {
        const int width = 384, height = 216;
        const int depthWidth = 192, depthHeight = 108;
        const size_t pixels = size_t(width) * size_t(height);
        const size_t depthPixels = size_t(depthWidth) * size_t(depthHeight);

        std::vector<float> x(depthPixels * 3);
        for (size_t i = 0; i < depthPixels; ++i)
        {
            x[i] = float((i * 37) % 1000) / 1000.0f;
            x[depthPixels + i] = 0.6f;
            x[2 * depthPixels + i] = -0.3f;
        }
        float* deviceX = upload(x);

        iw3::MlbwGpu gpu;
        iw3::MonoBwVideoGpu video;
        if (!deviceX || !gpu.prepare(width, height, depthWidth, depthHeight) ||
            !video.prepare(width, height))
        {
            std::printf("FAIL: setup: %s%s\n", gpu.error().c_str(), video.error().c_str());
            ++failures;
        }
        else
        {
            const int64_t xShape[4] = {1, 3, depthHeight, depthWidth};
            const int64_t shape[4] = {iw3::MonoBwVideoGpu::kSequence, 3, height, width};
            // The extra argument is not decoration: with one argument this is a
            // function declaration, not a vector. Third time in this codebase.
            std::vector<float> greys(size_t(iw3::MonoBwVideoGpu::kSequence), 0.0f);
            bool ok = true;

            for (int slot = 0; slot < iw3::MonoBwVideoGpu::kSequence && ok; ++slot)
            {
                // A flat frame per slot. The warp of a flat frame is still flat,
                // so whatever comes out of the cache can be read straight back
                // as "which slot did this come from".
                greys[size_t(slot)] = 0.2f + 0.05f * float(slot);
                std::vector<float> image(pixels * 3, greys[size_t(slot)]);
                float* deviceImage = upload(image);
                if (!deviceImage)
                {
                    std::printf("FAIL: slot %d upload\n", slot);
                    ok = false;
                    break;
                }

                const float* delta = nullptr;
                const float* weight = nullptr;
                const float* logits = nullptr;
                if (!ort.runMlbwDevice(4, deviceX, xShape, &delta, &weight, &logits) ||
                    !gpu.prepareEye(deviceImage, delta, weight, logits, false, 0, 0, nullptr))
                {
                    dump(ort, "mlbw video window");
                    std::printf("FAIL: slot %d warp: %s\n", slot, gpu.error().c_str());
                    ok = false;
                }
                else
                {
                    gpu.prepareInpaintInput(0, nullptr);
                    video.storeFrame(slot, gpu.inpaintEyeHalfDevice(),
                                     gpu.processedMaskHalfDevice(), nullptr);
                }
                cudaDeviceSynchronize();
                cudaFree(deviceImage);
            }

            const float* filled = nullptr;
            if (ok && !ort.runInpaintDevice(3, reinterpret_cast<const float*>(video.eyesDevice()),
                                            reinterpret_cast<const float*>(video.masksDevice()),
                                            shape, &filled))
            {
                dump(ort, "mlbw video fill");
                std::printf("FAIL: temporal fill inference\n");
                ok = false;
            }
            if (ok)
            {
                video.cacheOutput(false, filled, nullptr);
                cudaDeviceSynchronize();

                int wrong = 0;
                for (int offset = 0; offset < iw3::MonoBwVideoGpu::kStride; ++offset)
                {
                    const size_t middle = size_t(height / 2) * size_t(width) + size_t(width / 2);
                    unsigned short bits = 0;
                    cudaMemcpy(&bits,
                               static_cast<const char*>(video.cachedFrame(false, offset)) +
                                   middle * sizeof(unsigned short),
                               sizeof(bits), cudaMemcpyDeviceToHost);
                    const float got = halfToFloat(bits);
                    const int expectedSlot = iw3::MonoBwVideoGpu::kPad + offset;
                    const float want = greys[size_t(expectedSlot)];
                    const bool good = std::abs(got - want) < 0.02f;
                    std::printf("    %s cache[%d] = %.3f, slot %d went in as %.3f\n",
                                good ? "ok  " : "FAIL", offset, got, expectedSlot, want);
                    if (!good) ++wrong;
                }
                if (wrong)
                {
                    std::printf("FAIL: %d of %d cached frames came from the wrong slot\n",
                                wrong, iw3::MonoBwVideoGpu::kStride);
                    ++failures;
                }
            }
            else
            {
                ++failures;
            }
        }
        if (deviceX) cudaFree(deviceX);
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
