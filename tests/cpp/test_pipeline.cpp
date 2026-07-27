// Checks the C++ numeric core against the Python implementation.
//
// The Python is validated against stock iw3 at max absolute difference 0, so
// this is what carries that guarantee across the port. Reference data comes
// from tools/dump_pipeline_reference.py.
//
//     ofx\build\tests\Release\test_pipeline.exe tests\cpp\pipeline_reference.bin

#include "stereo_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{

struct Case
{
    std::string name;
    std::vector<int32_t> ints;
    std::vector<float> inputs;
    std::vector<float> outputs;
};

template <typename T>
bool readVector(std::ifstream& stream, std::vector<T>& out)
{
    int32_t count = 0;
    if (!stream.read(reinterpret_cast<char*>(&count), 4)) return false;
    out.resize(size_t(count));
    if (count > 0 && !stream.read(reinterpret_cast<char*>(out.data()), std::streamsize(count * sizeof(T))))
    {
        return false;
    }
    return true;
}

bool load(const std::string& path, std::vector<Case>& cases)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        std::printf("cannot open %s\n", path.c_str());
        return false;
    }
    char magic[4] = {};
    stream.read(magic, 4);
    if (std::memcmp(magic, "IW3P", 4) != 0)
    {
        std::printf("bad magic in %s\n", path.c_str());
        return false;
    }
    int32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), 4);
    for (int32_t i = 0; i < count; ++i)
    {
        Case entry;
        int32_t nameLength = 0;
        stream.read(reinterpret_cast<char*>(&nameLength), 4);
        entry.name.resize(size_t(nameLength));
        stream.read(entry.name.data(), nameLength);
        if (!readVector(stream, entry.ints)) return false;
        if (!readVector(stream, entry.inputs)) return false;
        if (!readVector(stream, entry.outputs)) return false;
        cases.push_back(std::move(entry));
    }
    return true;
}

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
        else
        {
            std::printf("  ??   %s (no handler)\n", entry.name.c_str());
        }
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
