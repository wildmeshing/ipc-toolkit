#include "distance_type.hpp"

#include <ipc/utils/eigen_ext.hpp>
#include <ipc/utils/logger.hpp>

#include <Eigen/Geometry>
#include <geogram/numerics/exact_geometry.h>
#include "fp_filters.h"

#ifdef IPC_TOOLKIT_WITH_GEOGRAM
#include <geogram/numerics/exact_geometry.h>
#include "fp_filters.h"
#endif

#ifdef IPC_TOOLKIT_WITH_GEOGRAM
namespace ipc {
using ExReal = GEO::expansion_nt; // exact scalar type
using ExVec3 = GEO::vec3E; // exact vector type

inline void init_pck() {
    struct PckInit { PckInit() { GEO::PCK::initialize(); } };
    static PckInit _;
}

inline ExVec3 make_exact(Eigen::ConstRef<VectorMax3d> v) {
    ExReal x{v.x()};
    ExReal y{v.y()};
    ExReal z{v.size() < 3 ? 0 : v.z()}; // compatibility with 2D vectors
    return ExVec3(std::move(x), std::move(y), std::move(z));
}

int dot3_3d(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_
) {
    // Evaluates the sign of dot(p1-p0, p2-p0)
    const int s = dot3_3d_filter(p0_.data(), p1_.data(), p2_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().trace("dot3_3d filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExReal ss = dot(p1 - p0, p2 - p0);
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

int dot3_2d(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_
) {
    // Evaluates the sign of dot(p1-p0, p2-p0)
    const int s = dot3_2d_filter(p0_.data(), p1_.data(), p2_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().trace("dot3_2d filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExReal ss = dot(p1 - p0, p2 - p0);
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

int cross_dot_cross_1(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_,
    Eigen::ConstRef<VectorMax3d> p3_
) {
    /*
    Evaluates the sign of dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p0)) =
    = dot(p1-p0, p3-p0) * dot(p1-p0, p2-p0) - dot(p1-p0, p1-p0) * dot(p2-p0, p3-p0)
    */
    const int s = cross_dot_cross_1_3d_filter(p0_.data(), p1_.data(), p2_.data(), p3_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().trace("cross_dot_cross_1 filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExVec3 p3 = make_exact(p3_);
    const ExReal ss = dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p0));
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

int cross_dot_cross_2(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_,
    Eigen::ConstRef<VectorMax3d> p3_
) {
    /*
    Evaluates the sign of dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p2)) =
    = dot(p1-p0, p3-p0) * dot(p1-p2, p2-p0) - dot(p1-p0, p1-p2) * dot(p2-p0, p3-p0)
    */
    const int s = cross_dot_cross_2_3d_filter(p0_.data(), p1_.data(), p2_.data(), p3_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().trace("cross_dot_cross_1 filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExVec3 p3 = make_exact(p3_);
    const ExReal ss = dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p2));
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}


static PointEdgeDistanceType point_edge_distance_type_predicate(
    Eigen::ConstRef<VectorMax3d> p,
    Eigen::ConstRef<VectorMax3d> e0,
    Eigen::ConstRef<VectorMax3d> e1)
{
    init_pck();
    assert(p.size() == e0.size() && p.size() == e1.size());
    if (p.size() == 2) {
        if (dot3_2d(e0, p, e1) <= 0) return PointEdgeDistanceType::P_E0;
        else if (dot3_2d(e1, p, e0) <= 0) return PointEdgeDistanceType::P_E1;
        else return PointEdgeDistanceType::P_E;
    }
    else {
        if (dot3_3d(e0, p, e1) <= 0) return PointEdgeDistanceType::P_E0;
        else if (dot3_3d(e1, p, e0) <= 0) return PointEdgeDistanceType::P_E1;
        else return PointEdgeDistanceType::P_E;
    }
}

// Standard analytic implementation.
static PointEdgeDistanceType point_edge_distance_type_standard(
    Eigen::ConstRef<VectorMax3d> p,
    Eigen::ConstRef<VectorMax3d> e0,
    Eigen::ConstRef<VectorMax3d> e1)
{
    assert(p.size() == 2 || p.size() == 3);
    assert(e0.size() == 2 || e0.size() == 3);
    assert(e1.size() == 2 || e1.size() == 3);

    const VectorMax3d e = e1 - e0;
    const double e_length_sqr = e.squaredNorm();
    if (e_length_sqr == 0) {
        logger().warn("Degenerate edge in point_edge_distance_type!");
        return PointEdgeDistanceType::P_E0; // WARNING: use arbitrary end-point
    }

    const double ratio = e.dot(p - e0) / e_length_sqr;
    if (ratio < 0) {
        return PointEdgeDistanceType::P_E0; // PP (p-e0)
    } else if (ratio > 1) {
        return PointEdgeDistanceType::P_E1; // PP (p-e1)
    } else {
        return PointEdgeDistanceType::P_E; // PE
    }
}

PointEdgeDistanceType point_edge_distance_type(
    Eigen::ConstRef<VectorMax3d> p,
    Eigen::ConstRef<VectorMax3d> e0,
    Eigen::ConstRef<VectorMax3d> e1)
{
    return DistanceTypeConfig::instance().use_standard()
        ? point_edge_distance_type_standard(p, e0, e1)
        : point_edge_distance_type_predicate(p, e0, e1);
}


static PointTriangleDistanceType point_triangle_distance_type_predicate(
    Eigen::ConstRef<Eigen::Vector3d> p,
    Eigen::ConstRef<Eigen::Vector3d> t0,
    Eigen::ConstRef<Eigen::Vector3d> t1,
    Eigen::ConstRef<Eigen::Vector3d> t2)
{
    init_pck();
    const int dot01 = dot3_3d(t0, p, t1);
    const int dot02 = dot3_3d(t0, p, t2);
    if (dot01 <= 0 && dot02 <= 0) {
        return PointTriangleDistanceType::P_T0;
    }
    const int dot12 = dot3_3d(t1, p, t2);
    const int dot10 = dot3_3d(t1, p, t0);
    if (dot12 <= 0 && dot10 <= 0) {
        return PointTriangleDistanceType::P_T1;
    }
    const int dot20 = dot3_3d(t2, p, t0);
    const int dot21 = dot3_3d(t2, p, t1);
    if (dot20 <= 0 && dot21 <= 0) {
        return PointTriangleDistanceType::P_T2;
    }

    if (cross_dot_cross_1(t0, t1, t2, p) >= 0 && dot01 > 0 && dot10 > 0)
        return PointTriangleDistanceType::P_E0;
    if (cross_dot_cross_1(t1, t2, t0, p) >= 0 && dot12 > 0 && dot21 > 0)
        return PointTriangleDistanceType::P_E1;
    if (cross_dot_cross_1(t2, t0, t1, p) >= 0 && dot20 > 0 && dot02 > 0)
        return PointTriangleDistanceType::P_E2;

    return PointTriangleDistanceType::P_T;
}

// Standard analytic implementation.
static PointTriangleDistanceType point_triangle_distance_type_standard(
    Eigen::ConstRef<Eigen::Vector3d> p,
    Eigen::ConstRef<Eigen::Vector3d> t0,
    Eigen::ConstRef<Eigen::Vector3d> t1,
    Eigen::ConstRef<Eigen::Vector3d> t2)
{
    const Eigen::Vector3d normal = (t1 - t0).cross(t2 - t0);

    Eigen::Matrix<double, 2, 3> basis, param;

    basis.row(0) = t1 - t0;
    basis.row(1) = basis.row(0).cross(normal);
    param.col(0) = (basis * basis.transpose()).ldlt().solve(basis * (p - t0));
    if (param(0, 0) > 0.0 && param(0, 0) < 1.0 && param(1, 0) >= 0.0) {
        return PointTriangleDistanceType::P_E0; // edge 0 is the closest
    }

    basis.row(0) = t2 - t1;
    basis.row(1) = basis.row(0).cross(normal);
    param.col(1) = (basis * basis.transpose()).ldlt().solve(basis * (p - t1));
    if (param(0, 1) > 0.0 && param(0, 1) < 1.0 && param(1, 1) >= 0.0) {
        return PointTriangleDistanceType::P_E1; // edge 1 is the closest
    }

    basis.row(0) = t0 - t2;
    basis.row(1) = basis.row(0).cross(normal);
    param.col(2) = (basis * basis.transpose()).ldlt().solve(basis * (p - t2));
    if (param(0, 2) > 0.0 && param(0, 2) < 1.0 && param(1, 2) >= 0.0) {
        return PointTriangleDistanceType::P_E2; // edge 2 is the closest
    }

    if (param(0, 0) <= 0.0 && param(0, 2) >= 1.0) {
        return PointTriangleDistanceType::P_T0;
    } else if (param(0, 1) <= 0.0 && param(0, 0) >= 1.0) {
        return PointTriangleDistanceType::P_T1;
    } else if (param(0, 2) <= 0.0 && param(0, 1) >= 1.0) {
        return PointTriangleDistanceType::P_T2;
    } else {
        return PointTriangleDistanceType::P_T;
    }
}

PointTriangleDistanceType point_triangle_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> p,
    Eigen::ConstRef<Eigen::Vector3d> t0,
    Eigen::ConstRef<Eigen::Vector3d> t1,
    Eigen::ConstRef<Eigen::Vector3d> t2)
{
    return DistanceTypeConfig::instance().use_standard()
        ? point_triangle_distance_type_standard(p, t0, t1, t2)
        : point_triangle_distance_type_predicate(p, t0, t1, t2);
}


bool is_almost_parallel_edge_edge(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    const Eigen::Vector3d u = ea1 - ea0;
    const Eigen::Vector3d v = eb1 - eb0;
    const double cross_norm_sqr = u.cross(v).squaredNorm();
    const double a = u.squaredNorm();
    const double c = v.squaredNorm();
    const double z = (a*c > 1.0) ? a*c : 1.0;
    return cross_norm_sqr < z * PARALLEL_THRESHOLD;
}

bool is_parallel_edge_edge(
    Eigen::ConstRef<Eigen::Vector3d> ea0_,
    Eigen::ConstRef<Eigen::Vector3d> ea1_,
    Eigen::ConstRef<Eigen::Vector3d> eb0_,
    Eigen::ConstRef<Eigen::Vector3d> eb1_)
{
    if constexpr (PARALLEL_THRESHOLD == 0.0) {
        init_pck();
        // TODO use a zero filter?
        const int s = cross_null_3d_filter(ea0_.data(), ea1_.data(), eb0_.data(), eb1_.data());
        if (s != FPG_UNCERTAIN_VALUE) return false;
        const ExVec3 ea0 = make_exact(ea0_);
        const ExVec3 ea1 = make_exact(ea1_);
        const ExVec3 eb0 = make_exact(eb0_);
        const ExVec3 eb1 = make_exact(eb1_);
        const ExReal cross_norm_sqr = cross(ea1-ea0, eb1-eb0).length2();
        return cross_norm_sqr == 0;
    }
    else return is_almost_parallel_edge_edge(ea0_, ea1_, eb0_, eb1_);
}


static EdgeEdgeDistanceType edge_edge_distance_type_predicate(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    init_pck();

    const PointEdgeDistanceType dt_ea0 = point_edge_distance_type(ea0, eb0, eb1);
    const PointEdgeDistanceType dt_ea1 = point_edge_distance_type(ea1, eb0, eb1);

    if (dt_ea0 == PointEdgeDistanceType::P_E0 && dot3_3d(ea0, eb0, ea1) <= 0)
        return EdgeEdgeDistanceType::EA0_EB0;
    if (dt_ea0 == PointEdgeDistanceType::P_E1 && dot3_3d(ea0, eb1, ea1) <= 0)
        return EdgeEdgeDistanceType::EA0_EB1;
    if (dt_ea1 == PointEdgeDistanceType::P_E0 && dot3_3d(ea1, eb0, ea0) <= 0)
        return EdgeEdgeDistanceType::EA1_EB0;
    if (dt_ea1 == PointEdgeDistanceType::P_E1 && dot3_3d(ea1, eb1, ea0) <= 0)
        return EdgeEdgeDistanceType::EA1_EB1;

    const PointEdgeDistanceType dt_eb0 = point_edge_distance_type(eb0, ea0, ea1);
    const PointEdgeDistanceType dt_eb1 = point_edge_distance_type(eb1, ea0, ea1);

    if (dt_eb0 == PointEdgeDistanceType::P_E && cross_dot_cross_2(eb0, ea0, ea1, eb1) >= 0)
        return EdgeEdgeDistanceType::EA_EB0;
    if (dt_eb1 == PointEdgeDistanceType::P_E && cross_dot_cross_2(eb1, ea0, ea1, eb0) >= 0)
        return EdgeEdgeDistanceType::EA_EB1;
    if (dt_ea0 == PointEdgeDistanceType::P_E && cross_dot_cross_2(ea0, eb0, eb1, ea1) >= 0)
        return EdgeEdgeDistanceType::EA0_EB;
    if (dt_ea1 == PointEdgeDistanceType::P_E && cross_dot_cross_2(ea1, eb0, eb1, ea0) >= 0)
        return EdgeEdgeDistanceType::EA1_EB;

    return EdgeEdgeDistanceType::EA_EB;
}

// Standard analytic implementation.
// A more robust implementation of http://geomalgorithms.com/a07-_distance.html
static EdgeEdgeDistanceType edge_edge_distance_type_standard(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    // Relative sin² threshold for parallelism: treat edges as parallel when
    // sin²(θ) < LEGACY_PARALLEL_THRESHOLD, scaled by a*c. This avoids
    // misclassifying short nearly-collinear coplanar segments (which the old
    // absolute threshold could not handle) and routes such cases to
    // edge_edge_parallel_distance_type.
    constexpr double LEGACY_PARALLEL_THRESHOLD = 2.5e-16;

    const Eigen::Vector3d u = ea1 - ea0;
    const Eigen::Vector3d v = eb1 - eb0;
    const Eigen::Vector3d w = ea0 - eb0;

    const double a = u.squaredNorm();
    const double b = u.dot(v);
    const double c = v.squaredNorm();
    const double d = u.dot(w);
    const double e = v.dot(w);
    const double D = a * c - b * b;

    if (a == 0.0 && c == 0.0) {
        return EdgeEdgeDistanceType::EA0_EB0;
    } else if (a == 0.0) {
        return EdgeEdgeDistanceType::EA0_EB;
    } else if (c == 0.0) {
        return EdgeEdgeDistanceType::EA_EB0;
    }

    // Special handling for parallel edges
    const double parallel_tolerance = LEGACY_PARALLEL_THRESHOLD * a * c;
    if (u.cross(v).squaredNorm() < parallel_tolerance) {
        return edge_edge_parallel_distance_type(ea0, ea1, eb0, eb1);
    }

    EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::EA_EB;

    const double sN = (b * e - c * d);
    double tN, tD;
    if (sN <= 0.0) {
        tN = e;
        tD = c;
        default_case = EdgeEdgeDistanceType::EA0_EB;
    } else if (sN >= D) {
        tN = e + b;
        tD = c;
        default_case = EdgeEdgeDistanceType::EA1_EB;
    } else {
        tN = (a * e - b * d);
        tD = D;
        if (tN > 0.0 && tN < tD
            && u.cross(v).squaredNorm() < parallel_tolerance) {
            if (sN < D / 2) {
                tN = e;
                tD = c;
                default_case = EdgeEdgeDistanceType::EA0_EB;
            } else {
                tN = e + b;
                tD = c;
                default_case = EdgeEdgeDistanceType::EA1_EB;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0) {
            return EdgeEdgeDistanceType::EA0_EB0;
        } else if (-d >= a) {
            return EdgeEdgeDistanceType::EA1_EB0;
        } else {
            return EdgeEdgeDistanceType::EA_EB0;
        }
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0) {
            return EdgeEdgeDistanceType::EA0_EB1;
        } else if ((-d + b) >= a) {
            return EdgeEdgeDistanceType::EA1_EB1;
        } else {
            return EdgeEdgeDistanceType::EA_EB1;
        }
    }

    return default_case;
}

EdgeEdgeDistanceType edge_edge_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    return DistanceTypeConfig::instance().use_standard()
        ? edge_edge_distance_type_standard(ea0, ea1, eb0, eb1)
        : edge_edge_distance_type_predicate(ea0, ea1, eb0, eb1);
}

EdgeEdgeDistanceType edge_edge_parallel_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    const Eigen::Vector3d ea = ea1 - ea0;
    const double alpha = (eb0 - ea0).dot(ea) / ea.squaredNorm();
    const double beta = (eb1 - ea0).dot(ea) / ea.squaredNorm();

    uint8_t eac; // 0: EA0, 1: EA1, 2: EA
    uint8_t ebc; // 0: EB0, 1: EB1, 2: EB
    if (alpha < 0) {
        eac = (0 <= beta && beta <= 1) ? 2 : 0;
        ebc = (beta <= alpha) ? 0 : (beta <= 1 ? 1 : 2);
    } else if (alpha > 1) {
        eac = (0 <= beta && beta <= 1) ? 2 : 1;
        ebc = (beta >= alpha) ? 0 : (0 <= beta ? 1 : 2);
    } else {
        eac = 2;
        ebc = 0;
    }

    assert(eac != 2 || ebc != 2);
    return EdgeEdgeDistanceType(ebc < 2 ? (eac << 1 | ebc) : (6 + eac));
}

#else

#error "NOT IMPLEMENTED!"

#endif

} // namespace ipc
