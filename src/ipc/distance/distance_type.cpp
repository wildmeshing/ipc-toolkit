#include "distance_type.hpp"

#include <ipc/utils/eigen_ext.hpp>
#include <ipc/utils/logger.hpp>

#include <Eigen/Geometry>
#include <geogram/numerics/exact_geometry.h>
#include "fp_filters.h"

namespace ipc {

using ExReal = GEO::expansion_nt; // exact scalar type
using ExVec3 = GEO::vec3E; // exact vector type

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

int dot3_3d(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_
) {
    // Evaluates the sign of dot(p1-p0, p2-p0)
    const int s = dot3_3d_filter(p0_.data(), p1_.data(), p2_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().debug("dot3_3d filter uncertain - fallback to exact arithmetic");
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
    logger().debug("dot3_2d filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExReal ss = dot(p1 - p0, p2 - p0);
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

int dot4_3d(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_,
    Eigen::ConstRef<VectorMax3d> p3_
) {
    // Evaluates the sign of dot(p1-p0, p3-p2)
    const int s = dot4_3d_filter(p0_.data(), p1_.data(), p2_.data(), p3_.data());
    if (s != FPG_UNCERTAIN_VALUE) return s;
    logger().debug("dot4_3d filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExVec3 p3 = make_exact(p3_);
    const ExReal ss = dot(p1 - p0, p3 - p2);
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
    logger().debug("cross_dot_cross_1 filter uncertain - fallback to exact arithmetic");
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
    logger().debug("cross_dot_cross_1 filter uncertain - fallback to exact arithmetic");
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExVec3 p3 = make_exact(p3_);
    const ExReal ss = dot(cross(p1-p0, p2-p0), cross(p3-p0, p1-p2));
    return (ss > 0) ? 1 : ((ss < 0) ? -1 : 0);
}

int dot_4(
    Eigen::ConstRef<VectorMax3d> p0_,
    Eigen::ConstRef<VectorMax3d> p1_,
    Eigen::ConstRef<VectorMax3d> p2_,
    Eigen::ConstRef<VectorMax3d> p3_)
{
    init_pck();
    const ExVec3 p0 = make_exact(p0_);
    const ExVec3 p1 = make_exact(p1_);
    const ExVec3 p2 = make_exact(p2_);
    const ExVec3 p3 = make_exact(p3_);
    const ExReal ss = dot(p1 - p0, p3 - p2);
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


PointTriangleDistanceType point_triangle_distance_type(
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


bool is_parallel_edge_edge(
    Eigen::ConstRef<Eigen::Vector3d> ea0_,
    Eigen::ConstRef<Eigen::Vector3d> ea1_,
    Eigen::ConstRef<Eigen::Vector3d> eb0_,
    Eigen::ConstRef<Eigen::Vector3d> eb1_)
{
    // TODO use a proper zero filter?
    init_pck();
    const int s = crossnull_3d_filter(ea0_.data(), ea1_.data(), eb0_.data(), eb1_.data());
    if (s != FPG_UNCERTAIN_VALUE) return false;
    const ExVec3 ea0 = make_exact(ea0_);
    const ExVec3 ea1 = make_exact(ea1_);
    const ExVec3 eb0 = make_exact(eb0_);
    const ExVec3 eb1 = make_exact(eb1_);
    return cross(ea1 - ea0, eb1 - eb0).length2() == 0;
}


EdgeEdgeDistanceType edge_edge_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    init_pck();
    if (is_parallel_edge_edge(ea0, ea1, eb0, eb1))
        return edge_edge_parallel_distance_type(ea0, ea1, eb0, eb1);

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

    if (dt_ea0 == PointEdgeDistanceType::P_E && cross_dot_cross_2(ea0, eb0, eb1, ea1) >= 0)
        return EdgeEdgeDistanceType::EA0_EB;
    if (dt_ea1 == PointEdgeDistanceType::P_E && cross_dot_cross_2(ea1, eb0, eb1, ea0) >= 0)
        return EdgeEdgeDistanceType::EA1_EB;

    const PointEdgeDistanceType dt_eb0 = point_edge_distance_type(eb0, ea0, ea1);
    const PointEdgeDistanceType dt_eb1 = point_edge_distance_type(eb1, ea0, ea1);

    if (dt_eb0 == PointEdgeDistanceType::P_E && cross_dot_cross_2(eb0, ea0, ea1, eb1) >= 0)
        return EdgeEdgeDistanceType::EA_EB0;
    if (dt_eb1 == PointEdgeDistanceType::P_E && cross_dot_cross_2(eb1, ea0, ea1, eb0) >= 0)
        return EdgeEdgeDistanceType::EA_EB1;

    return EdgeEdgeDistanceType::EA_EB;
}


EdgeEdgeDistanceType edge_edge_parallel_distance_type(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1)
{
    init_pck();

    const int sa0 = dot3_3d(ea0, eb0, ea1);
    const int sa1 = dot3_3d(ea1, eb0, ea0);
    const int sb0 = dot3_3d(ea0, eb1, ea1);
    const int sb1 = dot3_3d(ea1, eb1, ea0);
    const int sab = dot4_3d(ea0, ea1, eb0, eb1);

    if (sa0 < 0) {
        if (sab <= 0) {
            return EdgeEdgeDistanceType::EA0_EB0;
        }
        if (sb1 >= 0) {
            if (sb0 >= 0) return EdgeEdgeDistanceType::EA_EB1;
            else return EdgeEdgeDistanceType::EA0_EB1;
        }
        return EdgeEdgeDistanceType::EA0_EB;
    } 
    
    if (sa1 < 0) {
        if (sab >= 0) {
            return EdgeEdgeDistanceType::EA1_EB0;
        }
        if (sb0 >= 0) {
            if (sb1 >= 0) return EdgeEdgeDistanceType::EA_EB1;
            else return EdgeEdgeDistanceType::EA1_EB1;
        }
        return EdgeEdgeDistanceType::EA1_EB;
    }

    return EdgeEdgeDistanceType::EA_EB0;
}

} // namespace ipc
