#pragma once

#include <array>
#include <cmath>
#include <tuple>

namespace contact_potential_integration {

template <typename F>
struct SubstitutionWindow {
    F psi_lower;
    F psi_upper;
    F q0;
    F q1;
    F length;
    std::array<F, 2> rotated_normal;
};

template <typename F>
struct LineSegment {
    std::array<F, 2> p0;
    std::array<F, 2> p1;
    std::array<F, 2> delta;

    LineSegment();
    LineSegment(const std::array<F, 2>& p0_in, const std::array<F, 2>& p1_in);

    std::array<F, 2> point(F u) const;
};

template <typename F> F cubic_bspline(F v);

template <typename F> F H_kernel(F z);

template <typename F>
F directional_factor(const std::array<F, 2>& delta, const std::array<F, 2>& normal, double alpha);

template <typename F>
F point_contact_potential(
    const std::array<F, 2>& p0,
    const std::array<F, 2>& n0,
    const std::array<F, 2>& p1,
    const std::array<F, 2>& n1,
    F epsilon,
    double alpha,
    double power);

template <typename F>
std::tuple<F, F, std::array<F, 2>, F> rotate_point_and_normal(
    const LineSegment<F>& segment,
    const std::array<F, 2>& point,
    const std::array<F, 2>& normal);

template <typename F>
SubstitutionWindow<F> compute_substitution_window(
    const LineSegment<F>& segment,
    const std::array<F, 2>& point,
    const std::array<F, 2>& normal,
    double alpha);

template <typename F>
F integrate_potential_line_segment_substitution(
    const LineSegment<F>& segment,
    const std::array<F, 2>& point,
    const std::array<F, 2>& normal,
    double epsilon,
    double alpha,
    double power,
    int quad_order);

void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights);

}  // namespace contact_potential_integration

extern "C" double integrate_potential_line_segment_substitution_double(
    double p0x,
    double p0y,
    double p1x,
    double p1y,
    double pointx,
    double pointy,
    double normalx,
    double normaly,
    double epsilon,
    double alpha,
    double power,
    int quad_order);

extern "C" double integrate_potential_line_segment_substitution_ad_grad_double(
    double p0x,
    double p0y,
    double p1x,
    double p1y,
    double pointx,
    double pointy,
    double normalx,
    double normaly,
    double epsilon,
    double alpha,
    double power,
    int quad_order,
    double* grad_pointx,
    double* grad_pointy,
    double* grad_normalx,
    double* grad_normaly);

extern "C" double integrate_potential_line_segment_substitution_fd_grad_double(
    double p0x,
    double p0y,
    double p1x,
    double p1y,
    double pointx,
    double pointy,
    double normalx,
    double normaly,
    double epsilon,
    double alpha,
    double power,
    int quad_order,
    double h,
    double* grad_pointx,
    double* grad_pointy,
    double* grad_normalx,
    double* grad_normaly);
