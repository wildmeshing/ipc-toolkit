#include "distance_type.hpp"

#include <ipc/utils/eigen_ext.hpp>
#include <ipc/utils/logger.hpp>

#include <Eigen/Geometry>
#include <geogram/numerics/exact_geometry.h>
#include "fp_filters.h"

namespace ipc {

using ExReal = GEO::expansion_nt; // exact scalar type
using ExVec3 = GEO::vec3E; // exact vector type

constexpr double PARALLEL_THRESHOLD {1.0e-20};
// constexpr double PARALLEL_THRESHOLD {0}; TODO set to zero eventually

inline void init_pck() { // TODO init once in main
    static bool initialized = false;
    if (!initialized) {
        GEO::PCK::initialize();
        initialized = true;
    }
}

inline ExVec3 make_exact(Eigen::ConstRef<VectorMax3d> v) {
    ExReal x{v.x()};
    ExReal y{v.y()};
    ExReal z{v.size() < 3 ? 0 : v.z()}; // compatibility with 2D vectors
    return ExVec3(std::move(x), std::move(y), std::move(z));
}

inline int dot_3(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_
) {
    // Evaluates the sign of dot(p1-p0, p2-p0)
    const int s = dot_3d_filter(p0_.data(), p1_.data(), p2_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().debug("dot_3 filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExReal ss = dot(p1 - p0, p2 - p0);
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

inline int dot_2(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_
) {
    // Evaluates the sign of dot(p1-p0, p2-p0)
    //const int s = dot_3d_filter(p0_.data(), p1_.data(), p2_.data());
    //if (s != FPG_UNCERTAIN_VALUE) return s;
    //logger().debug("dot_3 filter uncertain - fallback to exact arithmetic");
    // TODO implement 2d filter
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExReal ss = dot(p1 - p0, p2 - p0);
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

inline int dot_cross_diff(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_,
    Eigen::ConstRef<VectorMax3d> p3_
) {
    /*
    Evaluates the sign of dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p0)) =
    = dot(p1-p0, p3-p0) * dot(p1-p0, p2-p0) - dot(p1-p0, p1-p0) * dot(p2-p0, p3-p0)
    */
    const int s = dot_cross_diff_3d_filter(p0_.data(), p1_.data(), p2_.data(), p3_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().debug("dot_cross_diff filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExVec3 p3 = make_exact(p3_);
    const ExReal ss = dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p0));
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}


PointEdgeDistanceType point_edge_distance_type(
    Eigen::ConstRef<VectorMax3d> p,
    Eigen::ConstRef<VectorMax3d> e0,
    Eigen::ConstRef<VectorMax3d> e1)
{
    init_pck();
    assert(p.size() == e0.size() && p.size() == e1.size());
    if (p.size() == 2) {
        if (dot_2(e0, p, e1) <= 0) return PointEdgeDistanceType::P_E0;
        else if (dot_2(e1, p, e0) <= 0) return PointEdgeDistanceType::P_E1;
        else return PointEdgeDistanceType::P_E;
    }
    else {
        if (dot_3(e0, p, e1) <= 0) return PointEdgeDistanceType::P_E0;
        else if (dot_3(e1, p, e0) <= 0) return PointEdgeDistanceType::P_E1;
        else return PointEdgeDistanceType::P_E;
    }
}


PointTriangleDistanceType point_triangle_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> p,
    Eigen::ConstRef<Eigen::Vector3d> t0,
    Eigen::ConstRef<Eigen::Vector3d> t1,
    Eigen::ConstRef<Eigen::Vector3d> t2)
{
    init_pck();
    const int dot01 = dot_3(t0, p, t1);
    const int dot02 = dot_3(t0, p, t2);
    if (dot01 <= 0 && dot02 <= 0) {
        return PointTriangleDistanceType::P_T0;
    }
    const int dot12 = dot_3(t1, p, t2);
    const int dot10 = dot_3(t1, p, t0);
    if (dot12 <= 0 && dot10 <= 0) {
        return PointTriangleDistanceType::P_T1;
    }
    const int dot20 = dot_3(t2, p, t0);
    const int dot21 = dot_3(t2, p, t1);
    if (dot20 <= 0 && dot21 <= 0) {
        return PointTriangleDistanceType::P_T2;
    }

    if (dot_cross_diff(t0, t1, t2, p) >= 0 && dot01 > 0 && dot10 > 0) {
        return PointTriangleDistanceType::P_E0;
    }
    if (dot_cross_diff(t1, t2, t0, p) >= 0 && dot12 > 0 && dot21 > 0) {
        return PointTriangleDistanceType::P_E1;
    }
    if (dot_cross_diff(t2, t0, t1, p) >= 0 && dot20 > 0 && dot02 > 0) {
        return PointTriangleDistanceType::P_E2;
    }

    return PointTriangleDistanceType::P_T;
}


bool is_parallel_edge_edge(
    Eigen::ConstRef<Eigen::Vector3d> ea0_,
    Eigen::ConstRef<Eigen::Vector3d> ea1_,
    Eigen::ConstRef<Eigen::Vector3d> eb0_,
    Eigen::ConstRef<Eigen::Vector3d> eb1_)
{
    // TODO use a zero filter maybe
    init_pck();
    const ExVec3 ea0 = make_exact(ea0_);
    const ExVec3 ea1 = make_exact(ea1_);
    const ExVec3 eb0 = make_exact(eb0_);
    const ExVec3 eb1 = make_exact(eb1_);

    const ExVec3 u = ea1 - ea0;
    const ExVec3 v = eb1 - eb0;

    const ExReal cross_norm_sqr = cross(u, v).length2();
    if constexpr (PARALLEL_THRESHOLD == 0.0) return cross_norm_sqr == 0;
    const ExReal a = u.length2();
    const ExReal c = v.length2();
    const ExReal z = (a*c > 1.0) ? a*c : ExReal(1.0);
    return cross_norm_sqr < z * PARALLEL_THRESHOLD;
}

// A more robust implementation of http://geomalgorithms.com/a07-_distance.html
EdgeEdgeDistanceType edge_edge_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> ea0_,
    Eigen::ConstRef<Eigen::Vector3d> ea1_,
    Eigen::ConstRef<Eigen::Vector3d> eb0_,
    Eigen::ConstRef<Eigen::Vector3d> eb1_)
{
    init_pck();
    const ExVec3 ea0 = make_exact(ea0_);
    const ExVec3 ea1 = make_exact(ea1_);
    const ExVec3 eb0 = make_exact(eb0_);
    const ExVec3 eb1 = make_exact(eb1_);

    const ExVec3 u = ea1 - ea0;
    const ExVec3 v = eb1 - eb0;
    const ExVec3 w = ea0 - eb0;

    const ExReal a = u.length2();   // always ≥ 0
    const ExReal b = dot(u, v);
    const ExReal c = v.length2();   // always ≥ 0
    const ExReal d = dot(u, w);
    const ExReal e = dot(v, w);
    const ExReal D = a * c - b * b; // always ≥ 0

    // Degenerate cases should not happen in practice, but we handle them
    if (a == 0 && c == 0) {
        return EdgeEdgeDistanceType::EA0_EB0;
    } else if (a == 0) {
        return EdgeEdgeDistanceType::EA0_EB;
    } else if (c == 0) {
        return EdgeEdgeDistanceType::EA_EB0;
    }

    // Special handling for parallel edges
    const ExReal cross_norm_sqr = cross(u, v).length2();
    bool is_parallel;
    if constexpr (PARALLEL_THRESHOLD == 0.0) {
        is_parallel = (cross_norm_sqr == 0);
    } else {
        const ExReal z = (a*c > 1.0) ? a*c : ExReal(1.0);
        is_parallel = cross_norm_sqr < z * PARALLEL_THRESHOLD;
    }
    if (is_parallel) {
        return edge_edge_parallel_distance_type(ea0_, ea1_, eb0_, eb1_);
    }

    EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::EA_EB;

    // compute the line parameters of the two closest points
    const ExReal sN = (b * e - c * d);
    ExReal tN, tD;   // tc = tN / tD
    if (sN <= 0) { // sc < 0 ⟹ the s=0 edge is visible
        tN = e;
        tD = c;
        default_case = EdgeEdgeDistanceType::EA0_EB;
    } else if (sN >= D) { // sc > 1 ⟹ the s=1 edge is visible
        tN = e + b;
        tD = c;
        default_case = EdgeEdgeDistanceType::EA1_EB;
    } else {
        tN = (a * e - b * d);
        tD = D; // default tD = D ≥ 0
    }

    if (tN <= 0) { // tc < 0 ⟹ the t=0 edge is visible
        // recompute sc for this edge
        if (-d <= 0) {
            return EdgeEdgeDistanceType::EA0_EB0;
        } else if (-d >= a) {
            return EdgeEdgeDistanceType::EA1_EB0;
        } else {
            return EdgeEdgeDistanceType::EA_EB0;
        }
    } else if (tN >= tD) { // tc > 1 ⟹ the t=1 edge is visible
        // recompute sc for this edge
        if ((-d + b) <= 0) {
            return EdgeEdgeDistanceType::EA0_EB1;
        } else if ((-d + b) >= a) {
            return EdgeEdgeDistanceType::EA1_EB1;
        } else {
            return EdgeEdgeDistanceType::EA_EB1;
        }
    }

    return default_case;
}

EdgeEdgeDistanceType edge_edge_parallel_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> ea0_,
    Eigen::ConstRef<Eigen::Vector3d> ea1_,
    Eigen::ConstRef<Eigen::Vector3d> eb0_,
    Eigen::ConstRef<Eigen::Vector3d> eb1_)
{
    init_pck();
    const ExVec3 ea0 = make_exact(ea0_);
    const ExVec3 ea1 = make_exact(ea1_);
    const ExVec3 eb0 = make_exact(eb0_);
    const ExVec3 eb1 = make_exact(eb1_);

    const ExVec3 ea = ea1 - ea0;
    const ExReal ea_len_sqr = ea.length2();
    const ExReal alpha_N = dot(eb0 - ea0, ea);
    const ExReal beta_N = dot(eb1 - ea0, ea);

    uint8_t eac; // 0: EA0, 1: EA1, 2: EA
    uint8_t ebc; // 0: EB0, 1: EB1, 2: EB
    if (alpha_N < 0) {
        eac = (beta_N >= 0 && beta_N <= ea_len_sqr) ? 2 : 0;
        ebc = (beta_N <= alpha_N) ? 0 : (beta_N <= ea_len_sqr ? 1 : 2);
    } else if (alpha_N > ea_len_sqr) {
        eac = (beta_N >= 0 && beta_N <= ea_len_sqr) ? 2 : 1;
        ebc = (beta_N >= alpha_N) ? 0 : (beta_N >= 0 ? 1 : 2);
    } else {
        eac = 2;
        ebc = 0;
    }

    // f(0, 0) = 0000 = 0 -> EA0_EB0
    // f(0, 1) = 0001 = 1 -> EA0_EB1
    // f(1, 0) = 0010 = 2 -> EA1_EB0
    // f(1, 1) = 0011 = 3 -> EA1_EB1
    // f(2, 0) = 0100 = 4 -> EA_EB0
    // f(2, 1) = 0101 = 5 -> EA_EB1
    // f(0, 2) = 0110 = 6 -> EA0_EB
    // f(1, 2) = 0111 = 7 -> EA1_EB
    // f(2, 2) = 1000 = 8 -> EA_EB

    assert(eac != 2 || ebc != 2); // This case results in a degenerate line-line
    return EdgeEdgeDistanceType(ebc < 2 ? (eac << 1 | ebc) : (6 + eac));
}

} // namespace ipc
