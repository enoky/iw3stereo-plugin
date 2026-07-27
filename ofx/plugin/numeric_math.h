// The pure arithmetic, shared verbatim between the CPU path and the CUDA
// kernels.
//
// One source of truth matters here more than usual: tests/cpp/test_pipeline.cpp
// validates these against Python that matches stock iw3 at difference 0, and it
// can only run them on the CPU. Having the GPU compile the same lines is what
// extends that guarantee to the GPU path -- a second transcription into .cu
// would not be covered by anything.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define IW3_HD __host__ __device__
#else
#define IW3_HD
#endif

namespace iw3
{
namespace math
{

// --- mappers, from iw3/mapper.py --------------------------------------------

IW3_HD inline double softplus01(double x, double bias, double scale)
{
    const double minV = log(1 + exp((0 - bias) * scale));
    const double maxV = log(1 + exp((1 - bias) * scale));
    const double v = log(1.0 + exp((x - bias) * scale));
    return (v - minV) / (maxV - minV);
}

IW3_HD inline double safeExpm1Log(double value)
{
    const double e = expm1(value);
    return log(e > 1e-6 ? e : 1e-6);
}

IW3_HD inline double invSoftplus01(double x, double bias, double scale)
{
    const double minV = safeExpm1Log((0 - bias) * scale);
    const double maxV = safeExpm1Log((1 - bias) * scale);
    return (safeExpm1Log((x - bias) * scale) - minV) / (maxV - minV);
}

IW3_HD inline double shiftRelativeDepth(double x, double minDistance)
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

// RELATIVE_MUL_MAPPER and RELATIVE_SHIFT_MAPPER, indexed by foreground_scale + 3.
// Index 3 is "none" in both.
IW3_HD inline double applyMultiplyMapper(int index, double x)
{
    switch (index)
    {
        case 0: return invSoftplus01(x, -0.0001, 3.4343);    // inv_mul_3
        case 1: return invSoftplus01(x, -0.0003, 6.2626);    // inv_mul_2
        case 2: return invSoftplus01(x, -0.002102, 7.8788);  // inv_mul_1
        case 3: return x;                                    // none
        case 4: return softplus01(x, 0.343, 12);             // mul_1
        case 5: return softplus01(x, 0.515, 12);             // mul_2
        case 6: return softplus01(x, 0.687, 12);             // mul_3
        default: return x;
    }
}

IW3_HD inline double applyShiftMapper(int index, double x)
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

// A resolved mapper, small enough to pass to a kernel by value.
struct MapperParams
{
    int a = 3;
    int b = 3;
    double weight = 0.0;
    int shift = 0;  // 0 = multiply family, 1 = shift family
    int identity = 1;
};

IW3_HD inline double applyMapper(const MapperParams& params, double x)
{
    if (params.identity)
    {
        return x;
    }
    const double first = params.shift ? applyShiftMapper(params.a, x) : applyMultiplyMapper(params.a, x);
    if (params.a == params.b)
    {
        return first;
    }
    const double second = params.shift ? applyShiftMapper(params.b, x) : applyMultiplyMapper(params.b, x);
    return first * (1.0 - params.weight) + second * params.weight;
}

// --- Dubois red/cyan ---------------------------------------------------------
//
// Each row sums to 1.0 across all six coefficients, so a grey pair stays grey.
// That is the property tests/cpp/test_pipeline.cpp checks, because it catches a
// single mistyped digit that a random comparison reports far less legibly.

IW3_HD inline void duboisPixel(double lr, double lg, double lb,
                               double rr, double rg, double rb,
                               double& outR, double& outG, double& outB)
{
    outR = 0.4561 * lr + 0.500484 * lg + 0.176381 * lb
         - 0.0434706 * rr - 0.0879388 * rg - 0.00155529 * rb;
    outG = -0.0400822 * lr - 0.0378246 * lg - 0.0157589 * lb
         + 0.378476 * rr + 0.73364 * rg - 0.0184503 * rb;
    outB = -0.0152161 * lr - 0.0205971 * lg - 0.00546856 * lb
         - 0.0721527 * rr - 0.112961 * rg + 1.2264 * rb;
}

IW3_HD inline double clamp01(double v)
{
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

// --- the divergence / convergence feature values ----------------------------

IW3_HD inline void featureValues(double divergence, double convergence, double imageWidth,
                                 float& divergenceValue, float& convergenceValue)
{
    const double divergencePix = divergence * 0.5 * 0.01 * imageWidth;
    divergenceValue = float(divergencePix / 32.0);
    convergenceValue = float((-divergencePix * convergence) / 32.0);
}

}  // namespace math
}  // namespace iw3
