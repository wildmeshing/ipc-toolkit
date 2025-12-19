#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace smoothed_offset_potential {

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
    using namespace std;
    using namespace TinyAD;
    
    std::array<F, 2> rel = {{point[0] - p0[0], point[1] - p0[1]}};
    F r_q = rel[0] * normal[0] + rel[1] * normal[1];
    F y_q = rel[0] * tangent[0] + rel[1] * tangent[1];

    phi_start = phi_value(r_q, y_q, F(0));
    phi_end = phi_value(r_q, y_q, length);

    F denom = pow(abs(r_q), power);
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
    using namespace std;
    using namespace TinyAD;

    F term = 1.0;
    if (phi_start_next) {  // Start or interior vertex
        term -= H(*phi_start_next / alpha);
    }
    if (phi_end_prev) {  // End or interior vertex
        term -= H(-*phi_end_prev / alpha);
    }

    F dist_to_vertex = hypot(point[0] - vertex_pt[0], point[1] - vertex_pt[1]);
    if (abs(dist_to_vertex) > 1e-12) {
        return term / pow(dist_to_vertex, power);
    }
    return 0.0;
}

/**
 * @brief Intersects a line segment with a single half-plane.
 *
 * Returns the interval [u_min, u_max] of the segment's parameter `u`
 * (in [0,1]) that lies within the half-plane.
 *
 * @tparam F The floating point type.
 * @param p0 The start point of the line segment.
 * @param p1 The end point of the line segment.
 * @param vertex The point on the boundary of the half-plane.
 * @param n The normal vector of the half-plane, pointing inwards.
 * @return A tuple (u_min, u_max) for the updated interval.
 */
template <typename F>
std::array<F, 2> intersect_segment_with_halfplane(
    const std::array<F, 2>& p0,
    const std::array<F, 2>& p1,
    const std::array<F, 2>& vertex,
    const std::array<F, 2>& n) {
    using namespace std;
    using namespace TinyAD;

    F u_min = 0.0, u_max = 1.0;
    std::array<F, 2> delta = {{p1[0] - p0[0], p1[1] - p0[1]}};
    F dot_delta_n = delta[0] * n[0] + delta[1] * n[1];
    std::array<F, 2> p0_minus_vertex = {{p0[0] - vertex[0], p0[1] - vertex[1]}};
    F dot_p0_minus_vertex_n = p0_minus_vertex[0] * n[0] + p0_minus_vertex[1] * n[1];

    if (abs(dot_delta_n) < 1e-12) {  // Segment parallel to plane boundary
        return (dot_p0_minus_vertex_n < 0.0) ? std::array<F, 2>{{1.0, 0.0}} : std::array<F, 2>{{u_min, u_max}};
    } else {
        F u = -dot_p0_minus_vertex_n / dot_delta_n;
        if (dot_delta_n > 0.0) {  // Entering half-plane
            return {{max(u_min, u), u_max}};
        } else {  // Exiting half-plane
            return {{u_min, min(u_max, u)}};
        }
    }
}

template <typename F>
std::array<F, 2> compute_vertex_window(
    const std::array<F, 2>& p0,
    const std::array<F, 2>& p1,
    const std::array<F, 2>& vertex,
    const std::array<F, 2>* v1,
    const std::array<F, 2>* v2,
    F alpha) {
    using namespace std;
    using namespace TinyAD;
    if (!v1 && !v2) {
        return {{0.0, 1.0}};
    }

    F phi = asin(alpha);
    F cos_angle = cos(phi);
    F sin_angle = sin(phi);

    bool has_n1 = false;
    std::array<F, 2> n1;
    bool has_n2 = false;
    std::array<F, 2> n2;

    if (v1) {
        has_n1 = true;
        // Rotate by angle
        n1 = {{(*v1)[0] * cos_angle - (*v1)[1] * sin_angle,
              (*v1)[0] * sin_angle + (*v1)[1] * cos_angle}};
    }

    if (v2) {
        has_n2 = true;
        // Rotate by -angle
        n2 = {{(*v2)[0] * cos_angle + (*v2)[1] * sin_angle,
             -(*v2)[0] * sin_angle + (*v2)[1] * cos_angle}};
    }

    if (has_n1 && has_n2) {
        F cross_product_n = n1[0] * n2[1] - n1[1] * n2[0];
        F cross_product_v = (*v1)[0] * (*v2)[1] - (*v1)[1] * (*v2)[0];
        if (cross_product_n < 0.0 && cross_product_v > 0.0) {
            // Union of two intersections
            auto res1 = intersect_segment_with_halfplane(p0, p1, vertex, n1);
            auto res2 = intersect_segment_with_halfplane(p0, p1, vertex, n2);
            bool empty1 = res1[0] > res1[1];
            bool empty2 = res2[0] > res2[1];

            if (empty1 && empty2) return {{1.0, 0.0}};
            if (empty1) return res2;
            if (empty2) return res1;

            return {{min(res1[0], res2[0]), max(res1[1], res2[1])}};
        }
    }

    // Default behavior: intersection of half-planes
    F u_min = 0.0, u_max = 1.0;
    if (has_n1) {
        auto res1 = intersect_segment_with_halfplane(p0, p1, vertex, n1);
        if (res1[0] > res1[1]) return {{1.0, 0.0}};
        u_min = max(u_min, res1[0]);
        u_max = min(u_max, res1[1]);
    }
    if (has_n2) {
        auto res2 = intersect_segment_with_halfplane(p0, p1, vertex, n2);
        if (res2[0] > res2[1]) return {{1.0, 0.0}};
        u_min = max(u_min, res2[0]);
        u_max = min(u_max, res2[1]);
    }

    if (u_min > u_max) {
        return {{1.0, 0.0}};
    }
    return {{u_min, u_max}};
}

/**
 * @brief Computes the intersection of a segment with the edge window defined by
 * three half-planes: the edge itself, and two endpoint cuts.
 */
template <typename F>
std::array<F, 2> compute_edge_window(
    const std::array<F, 2>& p0,
    const std::array<F, 2>& p1,
    const std::array<F, 2>& edge_p0,
    const std::array<F, 2>& edge_p1,
    F alpha) {
    using namespace std;
    using namespace TinyAD;
    
    F ex = edge_p1[0] - edge_p0[0];
    F ey = edge_p1[1] - edge_p0[1];
    F length = hypot(ex, ey);
    
    if (length < 1e-12) {
        return {{1.0, 0.0}};
    }

    // 1. Edge Half-plane (aligned with edge, normal up)
    std::array<F, 2> n_edge = {{-ey / length, ex / length}};
    
    // 2. Left and Right Endpoint Half-planes
    F angle = asin(alpha);
    F cos_angle = cos(angle);
    F sin_angle = sin(angle);

    // Left (at edge_p0): v1 = edge vector (ex, ey), rotate by angle
    std::array<F, 2> n_left = {{
        ex * cos_angle - ey * sin_angle,
        ex * sin_angle + ey * cos_angle
    }};

    // Right (at edge_p1): v2 = -edge vector (-ex, -ey), rotate by -angle
    std::array<F, 2> n_right = {{
        (-ex) * cos_angle + (-ey) * sin_angle,
        -(-ex) * sin_angle + (-ey) * cos_angle
    }};

    auto res_edge = intersect_segment_with_halfplane(p0, p1, edge_p0, n_edge);
    auto res_left = intersect_segment_with_halfplane(p0, p1, edge_p0, n_left);
    auto res_right = intersect_segment_with_halfplane(p0, p1, edge_p1, n_right);

    if (res_edge[0] > res_edge[1] || res_left[0] > res_left[1] || res_right[0] > res_right[1]) {
        return {{1.0, 0.0}};
    }

    F u_min = max({res_edge[0], res_left[0], res_right[0]});
    F u_max = min({res_edge[1], res_left[1], res_right[1]});

    if (u_min > u_max) {
        return {{1.0, 0.0}};
    }
    return {{u_min, u_max}};
}

}  // namespace smoothed_offset_potential