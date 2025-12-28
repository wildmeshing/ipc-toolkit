#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace offset_potential {

/**
 * @brief Heaviside function, with 0 -> 1 transition on -1 to 1.
 *
 * @tparam F The floating point type.
 * @param t The input value.
 * @return The Heaviside value.
 */
template <typename F>
F H0(F t) {
    if (t < 0.0) return 0.0;
    else return 1.0;
}

template <typename F>
inline F sqr(F x) { return x * x; }

template <typename F>
F activation_function(F d, const double r) {
    // quadratic-logrithmic 2 stage function taken from the GAIA implementation
    constexpr double CORRECTION = 1.; // 1 = as defined in GAIA, 2 = correct C2 implementation?
    constexpr double k = 1; // Stiffness, fixed to 1 as we already have stiffness implemented.
	F pd = r - d;
    const double tau = r * 0.5;
    if (d < tau && d > 0)
    {
        const double k2 = 0.5 * sqr(tau) * k;
        // Here I add two factors to make the function C1 (double typo? check) 
        const double b = k2 / (CORRECTION*r) + k2 * log(tau);
        return CORRECTION*(-log(d) * k2 + b);
    }
    else {
        return 0.5 * k * sqr(pd);
    }
}

/**
 * @brief Calculates the cosine of the angle between the segment's tangent and
 * the vector from a point on the segment's line to the query point.
 *
 * @tparam F The floating point type.
 * @param r_value Perpendicular distance from the query point to the line.
 * @param y_q Projected distance of the query point along the tangent.
 * @param y Position along the segment's tangent.
 * @return The phi value (a cosine).
 */
template <typename F>
F phi_value(F r_value, F y_q, F y) {
    using namespace std;
    using namespace TinyAD;
    F diff = y_q - y;
    F denom = hypot(diff, r_value);
    // Avoid division by zero if the point is on the vertex/endpoint
    return (denom > 1e-12) ? diff / denom : 0.0;
}

/**
 * @brief Calculates the potential contribution from a single edge of a polyline.
 *
 * @tparam F The floating point type.
 * @param point The 2D query point.
 * @param p0 The start point of the segment.
 * @param tangent The unit tangent vector of the segment.
 * @param normal The unit normal vector of the segment.
 * @param length The length of the segment.
 * @param power Decay rate of the potential.
 * @param epsilon The smoothing radius for the potential.
 * @param phi_start Output parameter for the phi value at the start.
 * @param phi_end Output parameter for the phi value at the end.
 * @return The edge potential contribution.
 */
template <typename F>
F polyline_edge_potential(
    const std::array<F, 2>& point,
    const std::array<F, 2>& p0,
    const std::array<F, 2>& tangent,
    const std::array<F, 2>& normal,
    F length,
    double power,
    double epsilon,
    F& phi_start,
    F& phi_end) {
    using namespace std;
    using namespace TinyAD;
    std::array<F, 2> rel = {{point[0] - p0[0], point[1] - p0[1]}};
    F r_q = rel[0] * normal[0] + rel[1] * normal[1];
    F y_q = rel[0] * tangent[0] + rel[1] * tangent[1];

    phi_start = phi_value(r_q, y_q, F(0));
    phi_end = phi_value(r_q, y_q, length);

    F denom = pow(abs(r_q), power);
    if (denom > 1e-12) {
        //F r = abs(r_q);
        F r = max(0.0, r_q); // Only offset in the normal direction? check
        return activation_function(r, epsilon) * H0(phi_start) * H0(-phi_end) / denom;
    }
    else return 0.0;
}

/**
 * @brief Calculates the potential contribution from a single vertex of a polyline.
 *
 * This function handles start, end, and interior vertices.
 *
 * @tparam F The floating point type.
 * @param point The 2D query point.
 * @param vertex_pt The location of the polyline vertex.
 * @param phi_start_next For an interior or start vertex, the phi value at the
 *   start of the *next* edge. Pass nullptr for the end vertex.
 * @param phi_end_prev For an interior or end vertex, the phi value at the
 *   end of the *previous* edge. Pass nullptr for the start vertex.
 * @param power Decay rate of the potential with distance.
 * @param epsilon The smoothing radius for the potential.
 * @return The calculated potential contribution from the vertex.
 */
template <typename F>
F polyline_vertex_potential(
    const std::array<F, 2>& point,
    const std::array<F, 2>& vertex_pt,
    const F* phi_start_next,
    const F* phi_end_prev,
    double power,
    double epsilon) {
    using namespace std;
    using namespace TinyAD;
    F term = 1.0;
    if (phi_start_next) {  // Start or interior vertex
        term -= H0(*phi_start_next);
    }
    if (phi_end_prev) {  // End or interior vertex
        term -= H0(-*phi_end_prev);
    }
    term = max(0.0, term); // Added because offset potential has no negative terms.

    F dist_to_vertex = hypot(point[0] - vertex_pt[0], point[1] - vertex_pt[1]);
    if (abs(dist_to_vertex) > 1e-12) {
        return activation_function(dist_to_vertex, epsilon) * term / pow(dist_to_vertex, power);
    }
    else return 0.0;
}

}  // ed_offset_potential