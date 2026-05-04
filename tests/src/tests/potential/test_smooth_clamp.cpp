#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <ipc/high_order_contact/smooth_clamp.hpp>

using ipc::smooth_clamp01;
using ipc::kSmoothClampEps;
using Catch::Approx;

namespace {

// Central finite-difference derivative of smooth_clamp01 at x.
double fd_derivative(double x, double h)
{
    return (smooth_clamp01(x + h) - smooth_clamp01(x - h)) / (2.0 * h);
}

} // namespace

TEST_CASE("smooth_clamp01 anchor values", "[smooth_clamp]")
{
    const double eps = kSmoothClampEps;
    CHECK(smooth_clamp01(0.0) == Approx(0.0));
    CHECK(smooth_clamp01(eps) == Approx(eps));
    CHECK(smooth_clamp01(0.5) == Approx(0.5));
    CHECK(smooth_clamp01(1.0 - eps) == Approx(1.0 - eps));
    CHECK(smooth_clamp01(1.0) == Approx(1.0));
}

TEST_CASE("smooth_clamp01 saturates outside [0,1]", "[smooth_clamp]")
{
    for (const double x : { -10.0, -1.0, -0.001 })
        CHECK(smooth_clamp01(x) == Approx(0.0));
    for (const double x : { 1.001, 2.0, 100.0 })
        CHECK(smooth_clamp01(x) == Approx(1.0));
}

TEST_CASE("smooth_clamp01 is identity in [eps, 1-eps]", "[smooth_clamp]")
{
    const double eps = kSmoothClampEps;
    const int N = 100;
    for (int i = 0; i <= N; i++) {
        const double x = eps + (1.0 - 2.0 * eps) * i / N;
        CHECK(smooth_clamp01(x) == Approx(x));
    }
}

TEST_CASE("smooth_clamp01 is monotonically increasing", "[smooth_clamp]")
{
    const int N = 1000;
    double prev = smooth_clamp01(-0.2);
    for (int i = 1; i <= N; i++) {
        const double x = -0.2 + 1.4 * i / N; // sweep [-0.2, 1.2]
        const double cur = smooth_clamp01(x);
        CHECK(cur >= prev - 1e-15);
        prev = cur;
    }
}

TEST_CASE("smooth_clamp01 is continuous (C0) at all knots", "[smooth_clamp]")
{
    const double eps = kSmoothClampEps;
    const double knots[] = { 0.0, eps, 1.0 - eps, 1.0 };
    const double h = 1e-8;
    for (const double k : knots) {
        const double left  = smooth_clamp01(k - h);
        const double right = smooth_clamp01(k + h);
        CHECK(std::abs(left - right) < 1e-7);
    }
}

TEST_CASE("smooth_clamp01 is C1 (derivative matches across pieces at knots)", "[smooth_clamp]")
{
    const double eps = kSmoothClampEps;
    // Analytical derivatives per piece (defined for x in their respective range):
    auto d_sat_lo  = [](double)         { return 0.0; };                                    // x <= 0
    auto d_blend_l = [&](double x)      { return -3 * x * x / (eps * eps) + 4 * x / eps; }; // 0 <= x <= eps
    auto d_id      = [](double)         { return 1.0; };                                    // eps <= x <= 1-eps
    auto d_blend_r = [&](double x)      {
        const double s = 1.0 - x;
        // d/dx [1 + s^3/eps^2 - 2 s^2/eps] with s = 1-x
        return -3 * s * s / (eps * eps) + 4 * s / eps;
    };
    auto d_sat_hi  = [](double)         { return 0.0; };                                    // x >= 1

    // At each knot the two adjacent piecewise derivative formulas must agree.
    CHECK(d_sat_lo(0.0)         == Approx(d_blend_l(0.0)));
    CHECK(d_blend_l(eps)        == Approx(d_id(eps)));
    CHECK(d_id(1.0 - eps)       == Approx(d_blend_r(1.0 - eps)));
    CHECK(d_blend_r(1.0)        == Approx(d_sat_hi(1.0)));
}

TEST_CASE("smooth_clamp01 derivative matches FD across the whole range", "[smooth_clamp]")
{
    const double eps = kSmoothClampEps;
    const double h = 1e-6;
    const int N = 200;

    // Probe densely inside each smooth piece (avoid straddling knots).
    auto check_segment = [&](double a, double b) {
        for (int i = 1; i < N; i++) {
            const double x = a + (b - a) * i / N;
            // Stay strictly inside the piece by margin > h.
            if (x - h < a || x + h > b) continue;
            const double fd = fd_derivative(x, h);
            // Analytical derivative per piece:
            //   x in [0, eps]:    f'(x) = -3 x^2 / eps^2 + 4 x / eps
            //   x in [eps, 1-eps]: f'(x) = 1
            //   x in [1-eps, 1]:  f'(x) = 3 (1-x)^2 / eps^2 - 4 (1-x) / eps + ... ; via reflection same form as above on s=1-x
            double analytic;
            if (x < 0.0 || x > 1.0) {
                analytic = 0.0;
            } else if (x < eps) {
                analytic = -3 * x * x / (eps * eps) + 4 * x / eps;
            } else if (x > 1.0 - eps) {
                const double s = 1.0 - x;
                analytic = -3 * s * s / (eps * eps) + 4 * s / eps;
            } else {
                analytic = 1.0;
            }
            CHECK(std::abs(fd - analytic) < 1e-5);
        }
    };

    check_segment(-0.2, 0.0);          // saturated below: derivative 0
    check_segment(0.0, eps);           // entering blend
    check_segment(eps, 1.0 - eps);     // identity
    check_segment(1.0 - eps, 1.0);     // exiting blend
    check_segment(1.0, 1.2);           // saturated above: derivative 0
}
