// MonoBW's arithmetic, shared verbatim between the CPU path and the CUDA
// kernels -- the same discipline as numeric_math.h, and for the same reason:
// tests/cpp/test_pipeline.cpp can only run this on the CPU, and having the GPU
// compile the same lines is what extends that coverage to the GPU path.
//
// This is iw3's sbs.monobw, ported from stereo_inpaint.py, which matches stock
// iw3 at difference 0. It is a forward warp expressed as a backward one:
//
//   1. push each pixel sideways by its depth              -> dest_index
//   2. make that monotone with a running maximum          -> invertible
//   3. blur the index map where step 2 moved it           -> no hard folds
//   4. invert it by binary search and interpolation       -> a sampling grid
//   5. mark where the inverse had to stretch              -> the hole mask
//
// Steps 1-5 are all *per row* and independent of every other row, which is why
// the CUDA side can run one thread per row and call straight into this header.
// A row is at the depth's width -- a few hundred to about a thousand -- so the
// serial scan inside a thread is short.
//
// It lives in its own header rather than in numeric_math.h because it is a row
// algorithm rather than a per-element one, and mixing the two would make the
// smaller file harder to read.

#pragma once

#include "numeric_math.h"

namespace iw3
{
namespace math
{

// MonoBW's smooth_kernel default. The 1x9 gaussian is applied along the width
// only, which is the axis the whole algorithm works on.
constexpr int kSmoothKernel = 9;
constexpr int kSmoothRadius = kSmoothKernel / 2;

// The 1x5 max_pool that widens "the monotonisation moved this pixel" before the
// blur is applied through it.
constexpr int kChangedRadius = 2;

// interpolate_1d's guard against a zero-width interval. Not a tolerance to
// tune: it is part of the arithmetic and changing it changes the picture.
constexpr float kInterpEpsilon = 1e-5f;

// compute_stretch_mask's threshold, as a fraction of one pixel's grid step.
constexpr float kStretchThreshold = 0.5f;

// nunif's get_gaussian_kernel1d, verbatim: sigma from the kernel size, samples
// on a symmetric integer grid, normalised to sum 1.
//
// Computed rather than tabulated so the formula is the thing under review. The
// intermediate is double because Python's is -- 9 * 0.15 + 0.35 is not 1.7 in
// binary, and rounding it early moves the last bit of every tap.
IW3_HD inline void smoothKernel(float* weights)
{
    const float sigma = float(double(kSmoothKernel) * 0.15 + 0.35);
    const float half = float(kSmoothKernel - 1) * 0.5f;
    float total = 0.0f;
    for (int i = 0; i < kSmoothKernel; ++i)
    {
        const float x = float(i) - half;
        const float value = expf(-0.5f * (x / sigma) * (x / sigma));
        weights[i] = value;
        total += value;
    }
    for (int i = 0; i < kSmoothKernel; ++i)
    {
        weights[i] /= total;
    }
}

// shift_size_px, the one scalar the row algorithm needs that depends on shape.
//
// base_size = max(H, W) is a Python max() over what become symbolic dimensions
// under torch.export, which is why this is host arithmetic here rather than
// something the kernel rederives -- see docs/monobw-inpaint.md.
inline double monobwShiftPx(double divergence, int depthWidth, int depthHeight)
{
    const double baseSize = double(depthWidth > depthHeight ? depthWidth : depthHeight);
    const double deltaScale = baseSize / double(depthWidth);
    return divergence * (0.01 * deltaScale * double(depthWidth - 1) * 0.5);
}

// preserve_screen_border's ramp width, and the width of the border region whose
// stretch mask is suppressed when it is off. Both left exactly as iw3 writes
// them: image_width cancels algebraically, but cancelling it changes the
// rounding, and these feed a round().
inline int monobwBorderPix(double divergence, int imageWidth, int depthWidth)
{
    const double scaled = double(depthWidth) / double(imageWidth);
    return int(lround(divergence * 0.75 * 0.01 * double(imageWidth) * scaled));
}

inline int monobwMaskBorderPix(double divergence, int imageWidth, int depthWidth)
{
    const double scaled = double(depthWidth) / double(imageWidth);
    return int(lround(divergence * 0.01 * double(imageWidth) * scaled)) + 1;
}

// torch.linspace(start, end, n) evaluated at i. Worth spelling out because the
// n == 1 case is not the average of the endpoints -- torch returns `start` --
// and the two border ramps run in opposite directions, so a single-pixel ramp
// is 0 on the left and 1 on the right rather than the same on both.
IW3_HD inline float linspaceAt(float start, float end, int n, int i)
{
    if (n <= 1)
    {
        return start;
    }
    return start + float(i) * ((end - start) / float(n - 1));
}

// torch.searchsorted(sorted, value, right=False): the first index whose entry
// is >= value.
//
// A binary search rather than a scan, and that is not an optimisation. The blur
// in step 3 replaces some entries and not others, and a mix of two monotone
// sequences need not itself be monotone, so the sequence handed to this is only
// *almost* sorted. Where it is not, a binary search and a linear scan disagree
// -- and torch does a binary search, so this does too.
IW3_HD inline int lowerBound(const float* sorted, int count, float value)
{
    int low = 0;
    int high = count;
    while (low < high)
    {
        const int mid = low + ((high - low) >> 1);
        if (sorted[mid] < value)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

IW3_HD inline int clampInt(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

// compute_backward_grid, split into the three stages the GPU needs it in.
//
// Only the first is genuinely sequential -- a running maximum along the row --
// and the other two are per element. Keeping them separate is not a GPU detail
// leaking into shared code: it is also how the arithmetic reads most clearly,
// because each stage reads the previous stage's output and never its own.
//
// Splitting matters because the sequential stage was the whole cost when all
// three ran in one thread per row: measured 1.85 ms of a 1.92 ms total at HD,
// scaling with row *length* rather than with pixels. A few hundred threads
// cannot hide a scan of a thousand dependent steps. As three stages, only the
// scan runs one-thread-per-row and the other two run one thread per element.
//
// The y coordinate is in none of them: it is the same linear ramp for every row
// and the caller builds it once.

// 1 and 2. Push each pixel sideways by its depth, and carry a running maximum
// so the result is monotone and therefore invertible.
//
// The two are fused because cummax *is* the running maximum of step 1's output,
// so one pass does both. `moved` records where the maximum overrode the raw
// value, which is what stage 2 needs to know and is otherwise recomputed.
//
// `borderPix` is preserve_screen_border's ramp, 0 when off.
IW3_HD inline void monobwDestIndexRow(const float* depth, int width,
                                      float shiftPx, float convergence, int borderPix,
                                      float* dest, float* moved)
{
    float running = -INFINITY;
    for (int x = 0; x < width; ++x)
    {
        float shift = (depth[x] - convergence) * shiftPx;
        if (borderPix > 0)
        {
            // Two sequential multiplies, not one combined weight: when the two
            // ramps overlap on a narrow frame, iw3 applies both.
            if (x < borderPix)
            {
                shift *= linspaceAt(0.0f, 1.0f, borderPix, x);
            }
            if (x >= width - borderPix)
            {
                shift *= linspaceAt(1.0f, 0.0f, borderPix, x - (width - borderPix));
            }
        }
        const float raw = float(x) + shift;
        running = raw > running ? raw : running;
        dest[x] = running;
        moved[x] = (raw != running) ? 1.0f : 0.0f;
    }
}

// 3. Blur the index map, but only where the monotonisation moved it, and widen
//    that region by a 1x5 max first so the blur is not applied to a single
//    pixel in the middle of an untouched run.
//
// Returns the value for one x. It reads the *unblurred* dest at every tap and
// the *undilated* flags at every neighbour, which is why the result goes to a
// separate array rather than back into dest.
IW3_HD inline float monobwSmoothAt(const float* dest, const float* moved, int width, int x)
{
    bool widened = false;
    for (int k = -kChangedRadius; k <= kChangedRadius && !widened; ++k)
    {
        const int sx = x + k;
        // max_pool2d pads with -inf, so out-of-range neighbours never win.
        if (sx >= 0 && sx < width && moved[sx] != 0.0f)
        {
            widened = true;
        }
    }
    if (!widened)
    {
        return dest[x];
    }

    float kernel[kSmoothKernel];
    smoothKernel(kernel);
    float sum = 0.0f;
    for (int k = 0; k < kSmoothKernel; ++k)
    {
        // Replicate padding, as GaussianFilter2d is built with.
        const int sx = clampInt(x + k - kSmoothRadius, 0, width - 1);
        sum += dest[sx] * kernel[k];
    }
    return sum;
}

// 4. Invert the mapping at one x, and normalise to grid coordinates.
//
// src_index is arange, so gathering it by an index just gives that index back;
// the two gathers iw3 writes are folded away here, which is exact for any width
// a frame can have.
IW3_HD inline float monobwInvertAt(const float* smoothed, int width, int x)
{
    const int found = lowerBound(smoothed, width, float(x));
    const int i0 = clampInt(found - 1, 0, width - 1);
    const int i1 = clampInt(found, 0, width - 1);

    const float d0 = smoothed[i0];
    const float d1 = smoothed[i1];
    const float s0 = float(i0);
    const float s1 = float(i1);

    const float t = (float(x) - d0) / ((d1 - d0) + kInterpEpsilon);
    const float back = s0 + t * (s1 - s0);
    return (back / float(width - 1)) * 2.0f - 1.0f;
}

// 5. compute_stretch_mask, for one pixel of a row at the *image's* width.
//
// A step of less than half a grid cell between neighbouring entries means the
// inverse is reading the same source pixel twice or more: a hole. The mask is
// set on both sides of the step, so this ors the two steps a pixel touches.
IW3_HD inline bool monobwStretched(const float* gridXRow, int width, int x)
{
    const float threshold = (2.0f / float(width - 1)) * kStretchThreshold;
    if (x > 0 && (gridXRow[x] - gridXRow[x - 1]) < threshold)
    {
        return true;
    }
    if (x + 1 < width && (gridXRow[x + 1] - gridXRow[x]) < threshold)
    {
        return true;
    }
    return false;
}

// F.grid_sample(mode="bilinear", padding_mode="border", align_corners=True) for
// one pixel of one plane.
//
// The grid is in normalised -1..1 coordinates; border padding means the sample
// position is clamped into range rather than the value being zeroed, so an edge
// pixel repeats outwards.
IW3_HD inline float bilinearSampleBorder(const float* plane, int width, int height,
                                         float gx, float gy)
{
    // align_corners=True: -1 maps to 0 and +1 maps to size - 1.
    float sx = (gx + 1.0f) * 0.5f * float(width - 1);
    float sy = (gy + 1.0f) * 0.5f * float(height - 1);

    const float x0f = floorf(sx);
    const float y0f = floorf(sy);
    const float fx = sx - x0f;
    const float fy = sy - y0f;

    const int x0 = clampInt(int(x0f), 0, width - 1);
    const int x1 = clampInt(int(x0f) + 1, 0, width - 1);
    const int y0 = clampInt(int(y0f), 0, height - 1);
    const int y1 = clampInt(int(y0f) + 1, 0, height - 1);

    const float v00 = plane[size_t(y0) * size_t(width) + size_t(x0)];
    const float v01 = plane[size_t(y0) * size_t(width) + size_t(x1)];
    const float v10 = plane[size_t(y1) * size_t(width) + size_t(x0)];
    const float v11 = plane[size_t(y1) * size_t(width) + size_t(x1)];

    // The weight order is grid_sample's own: the four corners are accumulated
    // as (1-fx)(1-fy), fx(1-fy), (1-fx)fy, fx fy.
    return v00 * ((1.0f - fx) * (1.0f - fy))
         + v01 * (fx * (1.0f - fy))
         + v10 * ((1.0f - fx) * fy)
         + v11 * (fx * fy);
}

// F.interpolate(mode="bilinear", align_corners=True) for one pixel of one
// plane. Used to lift the grid from the depth's resolution to the frame's.
IW3_HD inline float bilinearResizeAt(const float* plane, int inWidth, int inHeight,
                                     int outWidth, int outHeight, int x, int y)
{
    const float scaleX = outWidth > 1 ? float(inWidth - 1) / float(outWidth - 1) : 0.0f;
    const float scaleY = outHeight > 1 ? float(inHeight - 1) / float(outHeight - 1) : 0.0f;

    const float sx = float(x) * scaleX;
    const float sy = float(y) * scaleY;

    const int x0 = clampInt(int(floorf(sx)), 0, inWidth - 1);
    const int y0 = clampInt(int(floorf(sy)), 0, inHeight - 1);
    const int x1 = clampInt(x0 + 1, 0, inWidth - 1);
    const int y1 = clampInt(y0 + 1, 0, inHeight - 1);
    const float fx = sx - float(x0);
    const float fy = sy - float(y0);

    const float v00 = plane[size_t(y0) * size_t(inWidth) + size_t(x0)];
    const float v01 = plane[size_t(y0) * size_t(inWidth) + size_t(x1)];
    const float v10 = plane[size_t(y1) * size_t(inWidth) + size_t(x0)];
    const float v11 = plane[size_t(y1) * size_t(inWidth) + size_t(x1)];

    const float top = v00 + (v01 - v00) * fx;
    const float bottom = v10 + (v11 - v10) * fx;
    return top + (bottom - top) * fy;
}

}  // namespace math
}  // namespace iw3
