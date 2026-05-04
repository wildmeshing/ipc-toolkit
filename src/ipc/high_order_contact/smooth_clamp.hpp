#pragma once

#include <type_traits>

namespace ipc {

/// Width of the cubic-blend region at each end of [0, 1] for smooth_clamp01.
constexpr double kSmoothClampEps = 0.1;

namespace detail {
template <typename T>
double smooth_clamp_scalar(const T& x)
{
    if constexpr (std::is_same_v<T, double>) return x;
    else return x.val;
}
} // namespace detail

/// C^1 smooth saturation onto [0, 1].
///   f(x) = 0                       for x <= 0
///   f(x) = -x^3/eps^2 + 2 x^2/eps  for 0 <= x <= eps      (cubic blend)
///   f(x) = x                       for eps <= x <= 1-eps
///   reflected cubic blend          for 1-eps <= x <= 1
///   f(x) = 1                       for x >= 1
/// Continuous and C^1 globally; monotone on (0, eps) since
///   f'(x) = (x/eps) * (4 - 3 x/eps) > 0 for x in (0, eps).
template <typename T>
T smooth_clamp01(const T& x)
{
    constexpr double eps = kSmoothClampEps;
    const double xv = detail::smooth_clamp_scalar(x);
    if (xv <= 0.0) return T(0.0);
    if (xv >= 1.0) return T(1.0);
    if (xv < eps) {
        return -x * x * x / (eps * eps) + 2.0 * x * x / eps;
    }
    if (xv > 1.0 - eps) {
        const T s = 1.0 - x;
        return 1.0 + s * s * s / (eps * eps) - 2.0 * s * s / eps;
    }
    return x;
}

} // namespace ipc
