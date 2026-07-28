// Checks the C++ numeric core against the Python implementation.
//
// The Python is validated against stock iw3 at max absolute difference 0, so
// this is what carries that guarantee across the port. Reference data comes
// from tools/dump_pipeline_reference.py.
//
//     ofx\build\tests\Release\test_pipeline.exe tests\cpp\pipeline_reference.bin

#include "stereo_pipeline.h"

#include "reference_data.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

using iw3test::Case;
using iw3test::load;

int failures = 0;
int checks = 0;

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
    const std::string path = argc > 1 ? argv[1] : "tests/cpp/pipeline_reference.bin";
    std::vector<Case> cases;
    if (!load(path, cases))
    {
        return 2;
    }
    std::printf("%zu cases from %s\n\n", cases.size(), path.c_str());

    iw3::DepthResizer resizer;

    for (const Case& entry : cases)
    {
        if (entry.name.rfind("resize_", 0) == 0)
        {
            const int inW = entry.ints[0], inH = entry.ints[1];
            const int outW = entry.ints[2], outH = entry.ints[3];
            std::vector<float> got;
            resizer.resize(entry.inputs.data(), inW, inH, got, outW, outH);
            // Python accumulates in float64, this in float32; that is the whole
            // of the difference and it is well under a 16-bit depth step.
            expectClose(entry.name, got, entry.outputs, 2e-6);
        }
        else if (entry.name.rfind("mapper_", 0) == 0)
        {
            const iw3::MapperType type = entry.ints[0] == 1 ? iw3::MapperType::Shift
                                                            : iw3::MapperType::Multiply;
            const double scale = double(entry.ints[1]) / 1000.0;
            const iw3::Mapper mapper(scale, type);
            std::vector<float> got(entry.inputs.size());
            for (size_t i = 0; i < entry.inputs.size(); ++i)
            {
                got[i] = float(mapper(double(entry.inputs[i])));
            }
            expectClose(entry.name, got, entry.outputs, 1e-5);
        }
        else if (entry.name.rfind("input_tensor_", 0) == 0)
        {
            const int width = entry.ints[0], height = entry.ints[1];
            const bool border = entry.ints[2] != 0;
            // divergence and convergence are recoverable from the name, but the
            // tensor's own constant channels are the thing being checked, so
            // they are re-derived from the reference instead.
            const size_t pixels = size_t(width) * size_t(height);
            const double imageWidth = double(std::max(width, height));
            const double divergenceValue = double(entry.outputs[pixels]);
            const double convergenceValue = double(entry.outputs[2 * pixels]);
            // Undo the scaling the builder applies, to get back the parameters.
            const double divergence = border
                ? 0.0  // the border ramp zeroes column 0; recover from the middle
                : divergenceValue * 32.0 / (0.5 * 0.01 * imageWidth);
            double actualDivergence = divergence;
            if (border)
            {
                const size_t middle = pixels + size_t(height / 2) * size_t(width) + size_t(width / 2);
                actualDivergence = double(entry.outputs[middle]) * 32.0 / (0.5 * 0.01 * imageWidth);
            }
            const double divergencePix = actualDivergence * 0.5 * 0.01 * imageWidth;
            double convergence = 0.0;
            if (divergencePix != 0.0)
            {
                const size_t middle = 2 * pixels + size_t(height / 2) * size_t(width) + size_t(width / 2);
                convergence = -double(entry.outputs[middle]) * 32.0 / divergencePix;
            }

            std::vector<float> got;
            iw3::buildInputTensor(entry.inputs.data(), width, height,
                                  actualDivergence, convergence, border, got);
            expectClose(entry.name, got, entry.outputs, 1e-5);
        }
        else if (entry.name == "auto_depth_size")
        {
            ++checks;
            bool ok = true;
            for (size_t i = 0; i + 3 < entry.ints.size(); i += 4)
            {
                const int frameW = entry.ints[i], frameH = entry.ints[i + 1];
                const int wantW = entry.ints[i + 2], wantH = entry.ints[i + 3];
                const std::pair<int, int> got = iw3::autoDepthSize(frameW, frameH);
                if (got.first != wantW || got.second != wantH)
                {
                    std::printf("  FAIL auto_depth_size %dx%d -> %dx%d, want %dx%d\n",
                                frameW, frameH, got.first, got.second, wantW, wantH);
                    ok = false;
                }
            }
            if (ok)
            {
                std::printf("  ok   %-44s all %zu frame sizes\n",
                            entry.name.c_str(), entry.ints.size() / 4);
            }
            else
            {
                ++failures;
            }
        }
        else if (entry.name == "dubois")
        {
            const size_t pixels = size_t(entry.ints[0]);
            std::vector<float> got(pixels * 3);
            iw3::duboisAnaglyph(entry.inputs.data(), entry.inputs.data() + pixels * 3, pixels, got.data());
            expectClose(entry.name, got, entry.outputs, 1e-6);

            // A grey pair must come out the same grey: that is what catches a
            // mistyped coefficient, which the random comparison above would
            // also catch but far less legibly.
            ++checks;
            std::vector<float> grey(3 * 4, 0.42f), greyOut(3 * 4);
            iw3::duboisAnaglyph(grey.data(), grey.data(), 4, greyOut.data());
            double worst = 0.0;
            for (float value : greyOut)
            {
                worst = std::max(worst, double(std::abs(value - 0.42f)));
            }
            if (worst > 1e-5)
            {
                std::printf("  FAIL dubois grey preservation: off by %.3e\n", worst);
                ++failures;
            }
            else
            {
                std::printf("  ok   %-44s grey stays grey (%.1e)\n", "dubois_grey", worst);
            }
        }
        else if (entry.name.rfind("monobw_grid_", 0) == 0)
        {
            const int depthWidth = entry.ints[0], depthHeight = entry.ints[1];
            const int imageWidth = entry.ints[2];
            const bool border = entry.ints[3] != 0;
            const double divergence = double(entry.ints[4]) / 1000.0;
            const double convergence = double(entry.ints[5]) / 1000.0;

            std::vector<float> gridX, gridY;
            iw3::monobwGrid(entry.inputs.data(), depthWidth, depthHeight,
                            divergence, convergence, border, imageWidth, gridX, gridY);
            // 5e-5 in normalised coordinates is 2.3e-2 of a pixel at this
            // width, and the bar is set by one specific amplifier rather than
            // by general float32 drift.
            //
            // interpolate_1d divides by (d1 - d0 + 1e-5). Where the index map
            // is maximally stretched -- which is to say inside a hole -- d1 and
            // d0 are nearly equal and that epsilon is the whole denominator, so
            // a last-bit difference in d0 comes out multiplied by about 1e5.
            // Narrow rows stay at 3.6e-7; 938 wide at divergence 10, which is
            // the most stretching any sane setting produces, reaches 1.8e-5.
            //
            // It lands where the pixels are about to be replaced by the inpaint
            // model, and the mask check below is exact, which is the property
            // that would actually be visible if this drifted.
            expectClose(entry.name, gridX, entry.outputs, 5e-5);
        }
        else if (entry.name.rfind("monobw_forward_", 0) == 0)
        {
            const int width = entry.ints[0], height = entry.ints[1];
            const int depthWidth = entry.ints[2], depthHeight = entry.ints[3];
            const bool border = entry.ints[4] != 0;
            const int fixMask = entry.ints[5];
            const double divergence = double(entry.ints[6]) / 1000.0;
            const double convergence = double(entry.ints[7]) / 1000.0;

            const size_t pixels = size_t(width) * size_t(height);
            const float* image = entry.inputs.data();
            const float* depth = entry.inputs.data() + pixels * 3;

            std::vector<float> eye, mask;
            iw3::monobwForward(image, width, height, depth, depthWidth, depthHeight,
                               divergence, convergence, border, fixMask, eye, mask);

            std::vector<float> wantEye(entry.outputs.begin(),
                                       entry.outputs.begin() + ptrdiff_t(pixels * 3));
            std::vector<float> wantMask(entry.outputs.begin() + ptrdiff_t(pixels * 3),
                                        entry.outputs.end());
            expectClose(entry.name + "/eye", eye, wantEye, 1e-5);

            // The mask is a threshold on a difference, so it cannot be compared
            // with a tolerance: a pixel either matches or it is on the wrong
            // side of the line. Count disagreements instead.
            ++checks;
            size_t wrong = 0;
            for (size_t i = 0; i < mask.size() && i < wantMask.size(); ++i)
            {
                if (mask[i] != wantMask[i]) ++wrong;
            }
            if (mask.size() != wantMask.size() || wrong != 0)
            {
                std::printf("  FAIL %-44s %zu of %zu mask pixels differ\n",
                            (entry.name + "/mask").c_str(), wrong, wantMask.size());
                ++failures;
            }
            else
            {
                size_t set = 0;
                for (float value : wantMask) if (value != 0.0f) ++set;
                std::printf("  ok   %-44s %zu of %zu mask pixels, exact\n",
                            (entry.name + "/mask").c_str(), set, wantMask.size());
            }
        }
        else if (entry.name.rfind("mask_preprocess_", 0) == 0)
        {
            const int width = entry.ints[0], height = entry.ints[1];
            const int inner = entry.ints[2], outer = entry.ints[3];
            const int baseWidth = entry.ints[4];

            std::vector<float> got;
            iw3::maskPreprocess(entry.inputs.data(), width, height, inner, outer, baseWidth, got);

            // A mask is 0 or 1 and every operation on it is a max, a min or an
            // or. Nothing here rounds, so the bar is equality: a tolerance
            // would let a whole misplaced pixel through.
            ++checks;
            size_t wrong = 0;
            for (size_t i = 0; i < got.size() && i < entry.outputs.size(); ++i)
            {
                if (got[i] != entry.outputs[i]) ++wrong;
            }
            if (got.size() != entry.outputs.size() || wrong != 0)
            {
                std::printf("  FAIL %-44s %zu of %zu pixels differ\n",
                            entry.name.c_str(), wrong, entry.outputs.size());
                ++failures;
            }
            else
            {
                size_t set = 0;
                for (float value : entry.outputs) if (value != 0.0f) ++set;
                std::printf("  ok   %-44s %zu of %zu pixels, exact\n",
                            entry.name.c_str(), set, entry.outputs.size());
            }
        }
        else
        {
            std::printf("  ??   %s (no handler)\n", entry.name.c_str());
        }
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
