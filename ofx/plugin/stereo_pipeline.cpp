#include "stereo_pipeline.h"

#include <algorithm>
#include <cmath>

namespace iw3
{

namespace
{

// --- mappers, ported from iw3/mapper.py -------------------------------------

double softplus01(double x, double bias, double scale)
{
    const double minV = std::log(1 + std::exp((0 - bias) * scale));
    const double maxV = std::log(1 + std::exp((1 - bias) * scale));
    const double v = std::log(1.0 + std::exp((x - bias) * scale));
    return (v - minV) / (maxV - minV);
}

double invSoftplus01(double x, double bias, double scale)
{
    const auto safeLog = [](double value) { return std::log(std::max(std::expm1(value), 1e-6)); };
    const double minV = safeLog((0 - bias) * scale);
    const double maxV = safeLog((1 - bias) * scale);
    return (safeLog((x - bias) * scale) - minV) / (maxV - minV);
}

double shiftRelativeDepth(double x, double minDistance)
{
    const double maxDistance = 16.0;
    const double provisionalMax = minDistance + maxDistance;
    const double A = 1.0 / provisionalMax;
    const double B = (1.0 / minDistance) - (1.0 / provisionalMax);
    double distance = 1.0 / (A + B * x);
    distance = (1.0 - minDistance) + distance;
    const double newX = 1.0 / distance;
    const double minValue = 1.0 / (maxDistance + 1.0);
    const double valueRange = 1.0 - 1.0 / (maxDistance + 1.0);
    return (newX - minValue) / valueRange;
}

// RELATIVE_MUL_MAPPER and RELATIVE_SHIFT_MAPPER from iw3/mapper.py, indexed by
// foreground_scale + 3. Index 3 is "none" in both.
double applyMultiplyMapper(int index, double x)
{
    switch (index)
    {
        case 0: return invSoftplus01(x, -0.0001, 3.4343);   // inv_mul_3
        case 1: return invSoftplus01(x, -0.0003, 6.2626);   // inv_mul_2
        case 2: return invSoftplus01(x, -0.002102, 7.8788); // inv_mul_1
        case 3: return x;                                   // none
        case 4: return softplus01(x, 0.343, 12);            // mul_1
        case 5: return softplus01(x, 0.515, 12);            // mul_2
        case 6: return softplus01(x, 0.687, 12);            // mul_3
        default: return x;
    }
}

double applyShiftMapper(int index, double x)
{
    switch (index)
    {
        case 0: return shiftRelativeDepth(x, 0.45);  // shift_045
        case 1: return shiftRelativeDepth(x, 0.6);   // shift_06
        case 2: return shiftRelativeDepth(x, 0.8);   // shift_08
        case 3: return x;                            // none
        case 4: return shiftRelativeDepth(x, 1.4);   // shift_14
        case 5: return shiftRelativeDepth(x, 2.0);   // shift_20
        case 6: return shiftRelativeDepth(x, 3.0);   // shift_30
        default: return x;
    }
}

// The sizing rule from iw3_ext/depth_file.py.
constexpr int kPrepMultiple = 14;
constexpr int kDefaultLowerBound = 392;

}  // namespace

Mapper::Mapper(double foregroundScale, MapperType type)
    : _type(type)
{
    _identity = std::abs(foregroundScale) < 1e-9;
    if (_identity)
    {
        return;
    }

    // resolve_mapper_name(): an integral scale picks one mapper, a fractional
    // one interpolates the two it lies between. Note the negative branch keeps
    // the sign, so -1.5 interpolates inv_mul_1 and inv_mul_2 rather than
    // walking the wrong way along the list.
    const double floored = std::floor(std::abs(foregroundScale));
    const double ceiled = std::ceil(std::abs(foregroundScale));
    _weight = std::abs(foregroundScale) - floored;

    int a = int(floored);
    int b = int(ceiled);
    if (foregroundScale < 0)
    {
        a = -a;
        b = -b;
    }
    _a = std::clamp(a + 3, 0, 6);
    _b = std::clamp(b + 3, 0, 6);
}

double Mapper::operator()(double x) const
{
    if (_identity)
    {
        return x;
    }
    const auto apply = [this](int index, double value)
    {
        return _type == MapperType::Multiply ? applyMultiplyMapper(index, value)
                                             : applyShiftMapper(index, value);
    };
    if (_a == _b)
    {
        return apply(_a, x);
    }
    return apply(_a, x) * (1.0 - _weight) + apply(_b, x) * _weight;
}

std::pair<int, int> autoDepthSize(int frameWidth, int frameHeight)
{
    int lowerBound = kDefaultLowerBound;
    if (lowerBound % kPrepMultiple != 0)
    {
        lowerBound += kPrepMultiple - lowerBound % kPrepMultiple;
    }

    const int shortest = std::min(frameWidth, frameHeight);
    const double scale = double(lowerBound) / double(shortest);

    const auto scaled = [&](int value)
    {
        int result = int(double(value) * scale);
        result -= result % kPrepMultiple;
        return std::max(result, lowerBound);
    };
    return {scaled(frameWidth), scaled(frameHeight)};
}

std::pair<int, int> depthTargetSize(int frameWidth, int frameHeight, int stereoWidth)
{
    if (stereoWidth <= 0)
    {
        const std::pair<int, int> automatic = autoDepthSize(frameWidth, frameHeight);
        // Only bring depth down, never up: a frame smaller than the model's
        // working size is already fine.
        if (automatic.first >= frameWidth)
        {
            return {frameWidth, frameHeight};
        }
        return automatic;
    }

    const int width = std::min(frameWidth, stereoWidth);
    // iw3 uses the *image's* aspect ratio here, not the depth's, deliberately.
    const int height = int(double(frameHeight) * (double(width) / double(frameWidth)));
    return {width, std::max(1, height)};
}

void DepthResizer::build(Axis& axis, int inSize, int outSize)
{
    if (axis.inSize == inSize && axis.outSize == outSize)
    {
        return;
    }
    axis.inSize = inSize;
    axis.outSize = outSize;
    axis.start.assign(size_t(outSize), 0);
    axis.count.assign(size_t(outSize), 0);
    axis.offset.assign(size_t(outSize), 0);
    axis.weights.clear();

    const double scale = outSize > 1 ? double(inSize - 1) / double(outSize - 1) : 0.0;
    const double support = scale >= 1.0 ? scale : 1.0;
    const double invScale = scale >= 1.0 ? 1.0 / scale : 1.0;

    for (int i = 0; i < outSize; ++i)
    {
        // scale * (i + 0.5) even though align_corners is true: torch's
        // antialias path uses this, unlike its plain path. Using scale * i here
        // is wrong by 0.2 -- see docs/phase2-onnx.md.
        const double center = scale * (double(i) + 0.5);
        const int xmin = std::max(int(center - support + 0.5), 0);
        const int xmax = std::min(int(center + support + 0.5), inSize);

        axis.start[size_t(i)] = xmin;
        axis.count[size_t(i)] = std::max(0, xmax - xmin);
        axis.offset[size_t(i)] = int(axis.weights.size());

        double total = 0.0;
        const size_t first = axis.weights.size();
        for (int j = xmin; j < xmax; ++j)
        {
            const double w = std::max(0.0, 1.0 - std::abs((double(j) - center + 0.5) * invScale));
            axis.weights.push_back(float(w));
            total += w;
        }
        if (total > 0.0)
        {
            for (size_t k = first; k < axis.weights.size(); ++k)
            {
                axis.weights[k] = float(double(axis.weights[k]) / total);
            }
        }
    }
}

void DepthResizer::resize(const float* source, int sourceWidth, int sourceHeight,
                          std::vector<float>& target, int targetWidth, int targetHeight)
{
    build(_horizontal, sourceWidth, targetWidth);
    build(_vertical, sourceHeight, targetHeight);

    // Horizontal first, into scratch at (sourceHeight x targetWidth).
    _scratch.resize(size_t(sourceHeight) * size_t(targetWidth));
    for (int y = 0; y < sourceHeight; ++y)
    {
        const float* row = source + size_t(y) * size_t(sourceWidth);
        float* out = _scratch.data() + size_t(y) * size_t(targetWidth);
        for (int x = 0; x < targetWidth; ++x)
        {
            const int start = _horizontal.start[size_t(x)];
            const int count = _horizontal.count[size_t(x)];
            const float* weights = _horizontal.weights.data() + _horizontal.offset[size_t(x)];
            float sum = 0.0f;
            for (int k = 0; k < count; ++k)
            {
                sum += row[start + k] * weights[k];
            }
            out[x] = sum;
        }
    }

    // Then vertical.
    target.resize(size_t(targetHeight) * size_t(targetWidth));
    for (int y = 0; y < targetHeight; ++y)
    {
        const int start = _vertical.start[size_t(y)];
        const int count = _vertical.count[size_t(y)];
        const float* weights = _vertical.weights.data() + _vertical.offset[size_t(y)];
        float* out = target.data() + size_t(y) * size_t(targetWidth);
        for (int x = 0; x < targetWidth; ++x)
        {
            float sum = 0.0f;
            for (int k = 0; k < count; ++k)
            {
                sum += _scratch[size_t(start + k) * size_t(targetWidth) + size_t(x)] * weights[k];
            }
            out[x] = std::clamp(sum, 0.0f, 1.0f);
        }
    }
}

void buildInputTensor(const float* depth, int width, int height,
                      double divergence, double convergence,
                      bool preserveScreenBorder,
                      std::vector<float>& out)
{
    const size_t pixels = size_t(width) * size_t(height);
    out.resize(pixels * 3);

    // image_width is the depth's longer side, as in iw3.
    const double imageWidth = double(std::max(width, height));
    const double divergencePix = divergence * 0.5 * 0.01 * imageWidth;
    const float divergenceValue = float(divergencePix / 32.0);
    const float convergenceValue = float((-divergencePix * convergence) / 32.0);

    std::copy(depth, depth + pixels, out.begin());
    std::fill(out.begin() + ptrdiff_t(pixels), out.begin() + ptrdiff_t(pixels * 2), divergenceValue);
    std::fill(out.begin() + ptrdiff_t(pixels * 2), out.end(), convergenceValue);

    if (!preserveScreenBorder)
    {
        return;
    }

    // Left exactly as iw3 writes it: imageWidth cancels algebraically, but
    // cancelling it by hand changes the rounding.
    const int borderPix = int(std::lround(divergence * 0.75 * 0.01 * imageWidth *
                                          (double(width) / imageWidth)));
    if (borderPix <= 0)
    {
        return;
    }
    const int span = std::min(borderPix, width);
    for (int y = 0; y < height; ++y)
    {
        for (int channel = 1; channel <= 2; ++channel)
        {
            float* row = out.data() + size_t(channel) * pixels + size_t(y) * size_t(width);
            for (int x = 0; x < span; ++x)
            {
                const float weight = span > 1 ? float(x) / float(span - 1) : 0.0f;
                row[x] *= weight;
                row[width - 1 - x] *= weight;
            }
        }
    }
}

float deltaScale(int depthWidth)
{
    return float(1.0 / double(depthWidth / 2 - 1));
}

void duboisAnaglyph(const float* left, const float* right, size_t pixels, float* out)
{
    // Red/cyan Dubois. Each row sums to 1.0 across all six coefficients, so a
    // grey pair comes out the same grey.
    static const double kLeft[3][3] = {
        {0.4561, 0.500484, 0.176381},
        {-0.0400822, -0.0378246, -0.0157589},
        {-0.0152161, -0.0205971, -0.00546856},
    };
    static const double kRight[3][3] = {
        {-0.0434706, -0.0879388, -0.00155529},
        {0.378476, 0.73364, -0.0184503},
        {-0.0721527, -0.112961, 1.2264},
    };

    for (size_t i = 0; i < pixels; ++i)
    {
        const double lr = left[i];
        const double lg = left[pixels + i];
        const double lb = left[2 * pixels + i];
        const double rr = right[i];
        const double rg = right[pixels + i];
        const double rb = right[2 * pixels + i];

        for (int o = 0; o < 3; ++o)
        {
            const double value =
                kLeft[o][0] * lr + kLeft[o][1] * lg + kLeft[o][2] * lb +
                kRight[o][0] * rr + kRight[o][1] * rg + kRight[o][2] * rb;
            out[size_t(o) * pixels + i] = float(std::clamp(value, 0.0, 1.0));
        }
    }
}

}  // namespace iw3
