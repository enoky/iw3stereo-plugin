#include "stereo_pipeline.h"

#include "monobw_math.h"
#include "numeric_math.h"

#include <algorithm>
#include <cmath>

namespace iw3
{

namespace
{

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

iw3::math::MapperParams Mapper::params() const
{
    iw3::math::MapperParams params;
    params.a = _a;
    params.b = _b;
    params.weight = _weight;
    params.shift = _type == MapperType::Shift ? 1 : 0;
    params.identity = _identity ? 1 : 0;
    return params;
}

double Mapper::operator()(double x) const
{
    return iw3::math::applyMapper(params(), x);
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

void buildResampleAxis(ResampleAxis& axis, int inSize, int outSize)
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
    buildResampleAxis(_horizontal, sourceWidth, targetWidth);
    buildResampleAxis(_vertical, sourceHeight, targetHeight);

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
    float divergenceValue = 0.0f, convergenceValue = 0.0f;
    iw3::math::featureValues(divergence, convergence, imageWidth, divergenceValue, convergenceValue);

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
    for (size_t i = 0; i < pixels; ++i)
    {
        double r = 0.0, g = 0.0, b = 0.0;
        iw3::math::duboisPixel(left[i], left[pixels + i], left[2 * pixels + i],
                               right[i], right[pixels + i], right[2 * pixels + i],
                               r, g, b);
        out[i] = float(iw3::math::clamp01(r));
        out[pixels + i] = float(iw3::math::clamp01(g));
        out[2 * pixels + i] = float(iw3::math::clamp01(b));
    }
}

void monobwGrid(const float* depth, int depthWidth, int depthHeight,
                double divergence, double convergence,
                bool preserveScreenBorder, int imageWidth,
                std::vector<float>& gridX, std::vector<float>& gridY)
{
    gridX.resize(size_t(depthWidth) * size_t(depthHeight));
    gridY.resize(size_t(depthHeight));

    const float shiftPx = float(iw3::math::monobwShiftPx(divergence, depthWidth, depthHeight));
    int borderPix = 0;
    if (preserveScreenBorder)
    {
        borderPix = iw3::math::monobwBorderPix(divergence, imageWidth, depthWidth);
        borderPix = std::min(borderPix, depthWidth);
    }

    // The same three stages the kernels run, in the same order and out of the
    // same header.
    std::vector<float> dest(size_t(depthWidth), 0.0f);
    std::vector<float> moved(size_t(depthWidth), 0.0f);
    std::vector<float> smoothed(size_t(depthWidth), 0.0f);
    for (int y = 0; y < depthHeight; ++y)
    {
        const float* row = depth + size_t(y) * size_t(depthWidth);
        iw3::math::monobwDestIndexRow(row, depthWidth, shiftPx, float(convergence), borderPix,
                                      dest.data(), moved.data());
        for (int x = 0; x < depthWidth; ++x)
        {
            smoothed[size_t(x)] = iw3::math::monobwSmoothAt(dest.data(), moved.data(),
                                                            depthWidth, x);
        }
        float* out = gridX.data() + size_t(y) * size_t(depthWidth);
        for (int x = 0; x < depthWidth; ++x)
        {
            out[x] = iw3::math::monobwInvertAt(smoothed.data(), depthWidth, x);
        }
    }

    for (int y = 0; y < depthHeight; ++y)
    {
        gridY[size_t(y)] = iw3::math::linspaceAt(-1.0f, 1.0f, depthHeight, y);
    }
}

void monobwForward(const float* image, int width, int height,
                   const float* depth, int depthWidth, int depthHeight,
                   double divergence, double convergence,
                   bool preserveScreenBorder, int fixScreenBorderMask,
                   std::vector<float>& eye, std::vector<float>& mask)
{
    std::vector<float> gridXSmall, gridYSmall;
    monobwGrid(depth, depthWidth, depthHeight, divergence, convergence,
               preserveScreenBorder, width, gridXSmall, gridYSmall);

    const size_t pixels = size_t(width) * size_t(height);
    eye.resize(pixels * 3);
    mask.assign(pixels, 0.0f);

    // The grid is lifted to the frame's resolution once. iw3 interpolates it
    // twice -- its guard compares an int to a torch.Size and is always true, so
    // the second call resizes to the size it already is. That was measured to
    // be an exact no-op before it was dropped here.
    std::vector<float> gridXFull(pixels);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            gridXFull[size_t(y) * size_t(width) + size_t(x)] =
                iw3::math::bilinearResizeAt(gridXSmall.data(), depthWidth, depthHeight,
                                            width, height, x, y);
        }
    }
    std::vector<float> gridYFull(size_t(height), 0.0f);
    for (int y = 0; y < height; ++y)
    {
        // One column: the y grid is constant along x, and bilinear
        // interpolation of equal taps returns the value exactly.
        gridYFull[size_t(y)] = iw3::math::bilinearResizeAt(gridYSmall.data(), 1, depthHeight,
                                                           1, height, 0, y);
    }

    for (int y = 0; y < height; ++y)
    {
        const float* gridRow = gridXFull.data() + size_t(y) * size_t(width);
        const float gy = gridYFull[size_t(y)];
        for (int x = 0; x < width; ++x)
        {
            const size_t index = size_t(y) * size_t(width) + size_t(x);
            const float gx = gridRow[x];
            for (int c = 0; c < 3; ++c)
            {
                eye[size_t(c) * pixels + index] =
                    iw3::math::bilinearSampleBorder(image + size_t(c) * pixels,
                                                    width, height, gx, gy);
            }
            mask[index] = iw3::math::monobwStretched(gridRow, width, x) ? 1.0f : 0.0f;
        }
    }

    if (!preserveScreenBorder && fixScreenBorderMask > 0)
    {
        // The screen border stretches for a reason that is not an occlusion,
        // and inpainting it makes it worse.
        const int borderPix = std::min(
            iw3::math::monobwMaskBorderPix(divergence, width, depthWidth), width);
        for (int y = 0; y < height; ++y)
        {
            float* row = mask.data() + size_t(y) * size_t(width);
            for (int x = 0; x < borderPix; ++x)
            {
                row[x] = 0.0f;
            }
            if (fixScreenBorderMask == 2)
            {
                for (int x = 0; x < borderPix; ++x)
                {
                    row[width - 1 - x] = 0.0f;
                }
            }
        }
    }
}

void maskPreprocess(const float* mask, int width, int height,
                    int innerDilation, int outerDilation, int baseWidth,
                    std::vector<float>& out)
{
    const size_t pixels = size_t(width) * size_t(height);
    const int outer = iw3::math::maskDilateIterations(outerDilation, width, baseWidth);
    const int inner = iw3::math::maskDilateIterations(innerDilation, width, baseWidth);

    // closing() is two dilates then two erodes, and the passes cannot run in
    // place: each reads its input's whole 3x3 neighbourhood.
    std::vector<float> a(pixels, 0.0f);
    std::vector<float> b(pixels, 0.0f);

    const float* source = mask;
    float* destination = a.data();
    for (int pass = 0; pass < 4; ++pass)
    {
        const bool dilating = pass < 2;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                destination[size_t(y) * size_t(width) + size_t(x)] =
                    dilating ? iw3::math::maskDilateAt(source, width, height, x, y)
                             : iw3::math::maskErodeAt(source, width, height, x, y);
            }
        }
        source = destination;
        destination = (destination == a.data()) ? b.data() : a.data();
    }

    // `source` now holds the closed mask. The original goes back in because
    // closing erases isolated pixels, and the two dilations run as one window.
    out.resize(pixels);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            out[size_t(y) * size_t(width) + size_t(x)] =
                iw3::math::maskFinishAt(source, mask, width, x, y, outer, inner);
        }
    }
}

}  // namespace iw3
