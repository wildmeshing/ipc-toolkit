#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace smoothed_offset_potential {

template <typename F>
struct PolylineGeometry {
    const std::vector<std::array<F, 2>>& pts;
    const std::vector<std::array<F, 2>>& tangents;
    const std::vector<std::array<F, 2>>& normals;
    const std::vector<F>& lengths;
};

template <typename F>
F my_abs(const F &x) { return x < 0 ? -x : x; }

/**
 * @brief Smoothed p.w. cubic Heaviside, with 0 -> 1 transition on -1 to 1.
 *
 * @tparam F The floating point type.
 * @param t The input value.
 * @return The smoothed Heaviside value.
 */
template <typename F>
F H(F t) {
    if (t < -1.0) {
        return 0.0;
    }
    if (t > 1.0) {
        return 1.0;
    }
    return ((2.0 - t) * (t + 1.0) * (t + 1.0)) / 4.0;
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
 * @param alpha Smoothness parameter.
 * @param power Decay rate of the potential.
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
    F alpha,
    double power,
    F& phi_start,
    F& phi_end) {
    std::array<F, 2> rel = {{point[0] - p0[0], point[1] - p0[1]}};
    F r_q = rel[0] * normal[0] + rel[1] * normal[1];
    F y_q = rel[0] * tangent[0] + rel[1] * tangent[1];

    phi_start = phi_value(r_q, y_q, F(0));
    phi_end = phi_value(r_q, y_q, length);

    F denom = pow(my_abs(r_q), power);
    if (denom > 1e-12) {
        return H(phi_start / alpha) * H(-phi_end / alpha) / denom;
    }
    return 0.0;
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
 * @param alpha Smoothness parameter for angular transitions.
 * @param power Decay rate of the potential with distance.
 * @return The calculated potential contribution from the vertex.
 */
template <typename F>
F polyline_vertex_potential(
    const std::array<F, 2>& point,
    const std::array<F, 2>& vertex_pt,
    const F* phi_start_next,
    const F* phi_end_prev,
    F alpha,
    double power) {
    F term = 1.0;
    if (phi_start_next) {  // Start or interior vertex
        term -= H(*phi_start_next / alpha);
    }
    if (phi_end_prev) {  // End or interior vertex
        term -= H(-*phi_end_prev / alpha);
    }

    F dist_to_vertex = hypot(point[0] - vertex_pt[0], point[1] - vertex_pt[1]);
    if (my_abs(dist_to_vertex) > 1e-12) {
        return term / pow(dist_to_vertex, power);
    }
    return 0.0;
}

/**
 * @brief Calculates the smoothed potential at a 'point' due to a polyline.
 *
 * @tparam F The floating point type.
 * @param point The 2D query point.
 * @param geometry The pre-calculated geometry of the polyline.
 * @param alpha Smoothness parameter for angular transitions.
 * @param power Decay rate of the potential with distance.
 * @return The calculated potential value.
 */
template <typename F>
F polyline_potential(
    const std::array<F, 2>& point,
    const PolylineGeometry<F>& geometry,
    F alpha,
    double power) {
    const auto& pts = geometry.pts;
    const auto& tangents = geometry.tangents;
    const auto& normals = geometry.normals;
    const auto& lengths = geometry.lengths;
    size_t num_segments = lengths.size();

    std::vector<F> phi_start(num_segments);
    std::vector<F> phi_end(num_segments);

    F total = 0.0;

    // Edge contributions
    for (size_t idx = 0; idx < num_segments; ++idx) {
        total += polyline_edge_potential(
            point,
            pts[idx],
            tangents[idx],
            normals[idx],
            lengths[idx],
            alpha,
            power,
            phi_start[idx],
            phi_end[idx]);
    }

    // Vertex contributions
    for (size_t idx = 0; idx < num_segments + 1; ++idx) {
        if (idx == num_segments) {
            // End vertex
            total += polyline_vertex_potential(point, pts[idx], static_cast<const F*>(nullptr),
                                               &phi_end[idx - 1], alpha, power);
        } else if (idx == 0) {
            // Start vertex
            total += polyline_vertex_potential(point, pts[idx], &phi_start[idx],
                                               static_cast<const F*>(nullptr), alpha, power);
        } else {
            // Interior vertex
            total += polyline_vertex_potential(
                point, pts[idx], &phi_start[idx], &phi_end[idx - 1], alpha, power);
        }
    }

    return total;
}

}  // namespace smoothed_offset_potential

extern "C" double polyline_potential_double(
    const double* point,
    int num_vertices,
    const double* pts,
    const double* tangents,
    const double* normals,
    const double* lengths,
    double alpha,
    double power);
