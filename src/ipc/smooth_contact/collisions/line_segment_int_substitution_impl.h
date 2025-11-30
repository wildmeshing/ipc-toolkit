#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "line_segment_int_substitution.h"

using namespace std;

namespace contact_potential_integration {

constexpr double PI = 3.14159265358979323846;

template <typename F>
LineSegment<F>::LineSegment() : p0{{0.0, 0.0}}, p1{{0.0, 0.0}}, delta{{0.0, 0.0}} {}

template <typename F>
LineSegment<F>::LineSegment(const std::array<F, 2>& p0_in, const std::array<F, 2>& p1_in)
    : p0(p0_in), p1(p1_in), delta{{p1_in[0] - p0_in[0], p1_in[1] - p0_in[1]}} {}

template <typename F>
std::array<F, 2> LineSegment<F>::point(F u) const {
    return {{p0[0] + u * delta[0], p0[1] + u * delta[1]}};
}

template <typename F>
F cubic_bspline(F v) {
    F abs_v = abs(v);
    if (abs_v < 1.0) {
        return (2.0 / 3.0) - abs_v * abs_v + 0.5 * abs_v * abs_v * abs_v;
    } else if (abs_v < 2.0) {
        F t = 2.0 - abs_v;
        return (1.0 / 6.0) * t * t * t;
    }
    return 0.0;
}

template <typename F>
F H_kernel(F z) {
    if (z < -3.0) {
        return 0.0;
    } else if (z < -2.0) {
        F t = 3.0 + z;
        return (1.0 / 6.0) * t * t * t;
    } else if (z < -1.0) {
        return (1.0 / 6.0) * (3.0 - 9.0 * z - 9.0 * z * z - 2.0 * z * z * z);
    } else if (z < 0.0) {
        return 1.0 + (z * z * z) / 6.0;
    }
    return 1.0;
}

template <typename F>
F directional_factor(const std::array<F, 2>& delta, const std::array<F, 2>& normal, double alpha) {
    F denom = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
    if (denom == 0.0) {
        return 0.0;
    }
    F phi_m = abs(delta[0] * normal[1] - delta[1] * normal[0]) / denom;
    F phi_e = -(delta[0] * normal[0] + delta[1] * normal[1]) / denom;
    double alpha_inv = 2.0 / alpha;
    return alpha_inv * cubic_bspline(alpha_inv * phi_m) * H_kernel(3.0 * phi_e / alpha);
}

template <typename F>
F point_contact_potential(
    const std::array<F, 2>& p0,
    const std::array<F, 2>& n0,
    const std::array<F, 2>& p1,
    const std::array<F, 2>& n1,
    F epsilon,
    double alpha,
    double power) {
    F dx = p1[0] - p0[0];
    F dy = p1[1] - p0[1];
    F distance2 = dx * dx + dy * dy;
    F distance = sqrt(distance2);
    F g_xy = directional_factor(std::array<F, 2>{{dx, dy}}, n1, alpha);
    F g_yx = directional_factor(std::array<F, 2>{{-dx, -dy}}, n0, alpha);
    F gamma = g_xy * g_yx;
    auto eps_scale = 2.0 / epsilon;
    F weight = 1.5 * cubic_bspline(eps_scale * distance);
    F numerator = gamma * weight;
    F potential = numerator / pow(distance, power);
    return potential;
}

template <typename F>
std::tuple<F, F, std::array<F, 2>, F> rotate_point_and_normal(
    const LineSegment<F>& segment,
    const std::array<F, 2>& point,
    const std::array<F, 2>& normal) {
    F length = sqrt(segment.delta[0] * segment.delta[0] + segment.delta[1] * segment.delta[1]);
    if (length == 0.0) {
        throw std::runtime_error("Line segment must have non-zero length.");
    }
    std::array<F, 2> ex{{segment.delta[0] / length, segment.delta[1] / length}};
    std::array<F, 2> ey{{-ex[1], ex[0]}};
    std::array<F, 2> rel{{point[0] - segment.p0[0], point[1] - segment.p0[1]}};
    F q0 = rel[0] * ex[0] + rel[1] * ex[1];
    F q1 = rel[0] * ey[0] + rel[1] * ey[1];

    F nx = normal[0] * ex[0] + normal[1] * ex[1];
    F ny = normal[0] * ey[0] + normal[1] * ey[1];

    F norm_len = sqrt(nx * nx + ny * ny);
    if (norm_len == 0.0) {
        throw std::runtime_error("Normal vector must have non-zero length.");
    }
    nx /= norm_len;
    ny /= norm_len;

    return std::make_tuple(q0, q1, std::array<F, 2>{{nx, ny}}, length);
}

template <typename F>
SubstitutionWindow<F> compute_substitution_window(
    const LineSegment<F>& segment,
    const std::array<F, 2>& point,
    const std::array<F, 2>& normal,
    double alpha) {
    auto [q0, q1, rotated_normal, length] = rotate_point_and_normal(segment, point, normal);

    if (!(0.0 < alpha && alpha < 1.0)) {
        throw std::runtime_error("Substitution integral requires 0 < alpha < 1.");
    }

    double phi = asin(max(1e-12, min(1.0 - 1e-12, alpha)));
    F theta = atan2(rotated_normal[1], rotated_normal[0]);

    F psi_lower_geom = max(-phi, -theta - (PI / 2.0) - phi);
    F psi_upper_geom = min(phi, -theta - (PI / 2.0) + phi);
    if (psi_lower_geom >= psi_upper_geom) {
        return SubstitutionWindow<F>{0.0, 0.0, q0, q1, length, {{0.0, 0.0}}};
    }

    F psi_lower_x = atan((q0 - length) / abs(q1));
    F psi_upper_x = atan(q0 / abs(q1));

    F psi_lower = max(psi_lower_geom, psi_lower_x);
    F psi_upper = min(psi_upper_geom, psi_upper_x);

    if (psi_lower >= psi_upper) {
        return SubstitutionWindow<F>{0.0, 0.0, q0, q1, length, {{0.0, 0.0}}};
    }

    return SubstitutionWindow<F>{psi_lower, psi_upper, q0, q1, length, rotated_normal};
}

void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);
    int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(PI * (i + 0.75) / (n + 0.5));
        double z1;
        double p1 = 0.0;
        double p2 = 0.0;
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j <= n; ++j) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            double pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (std::abs(z - z1) > 1e-14);

        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        double pp = n * (z * p1 - p2) / (z * z - 1.0);
        double w = 2.0 / ((1.0 - z * z) * pp * pp);
        weights[i] = weights[n - 1 - i] = w;
    }
}

template <typename F>
F integrate_potential_line_segment_substitution(
    const LineSegment<F>& segment,
    const std::array<F, 2>& point,
    const std::array<F, 2>& normal,
    double epsilon,
    double alpha,
    double power,
    int quad_order) {
    auto window = compute_substitution_window(segment, point, normal, alpha);
    if (window.psi_lower >= window.psi_upper) {
        return static_cast<F>(0.0);
    }

    F psi_lower = window.psi_lower;
    F psi_upper = window.psi_upper;
    F q1_abs = abs(window.q1);

    std::vector<double> nodes;
    std::vector<double> weights;
    gauss_legendre(quad_order, nodes, weights);

    F half = 0.5 * (psi_upper - psi_lower);
    F center = 0.5 * (psi_upper + psi_lower);

    F scaled_sum = static_cast<F>(0.0);
    for (int i = 0; i < quad_order; ++i) {
        F psi = center + half * nodes[i];
        F cos_psi = cos(psi);
        F potential = point_contact_potential(
            std::array<F, 2>{{-tan(psi), 0.0}},
            std::array<F, 2>{{0.0, 1.0}},
            std::array<F, 2>{{0.0, 1.0}},
            window.rotated_normal,
            epsilon / q1_abs,
            alpha,
            power);
        scaled_sum += weights[i] * potential / (cos_psi * cos_psi);
    }

    F scale_factor = half * pow(q1_abs, 1.0 - power) / window.length;
    return scale_factor * scaled_sum;
}

inline double integrate_potential_line_segment_substitution_double(
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
    int quad_order) {
    LineSegment<double> seg({{p0x, p0y}}, {{p1x, p1y}});
    std::array<double, 2> pt{{pointx, pointy}};
    std::array<double, 2> n{{normalx, normaly}};
    return integrate_potential_line_segment_substitution(seg, pt, n, epsilon, alpha, power, quad_order);
}

template double point_contact_potential(
    const std::array<double, 2>&,
    const std::array<double, 2>&,
    const std::array<double, 2>&,
    const std::array<double, 2>&,
    double,
    double,
    double);
template SubstitutionWindow<double> compute_substitution_window(
    const LineSegment<double>&,
    const std::array<double, 2>&,
    const std::array<double, 2>&,
    double);
template double integrate_potential_line_segment_substitution(
    const LineSegment<double>&,
    const std::array<double, 2>&,
    const std::array<double, 2>&,
    double,
    double,
    double,
    int);

}  // namespace contact_potential_integration
