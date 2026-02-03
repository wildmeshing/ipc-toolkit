#pragma once

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/smooth_contact/common.hpp>
#include <ipc/utils/autodiff_types.hpp>
#include <ipc/utils/math.hpp>

#include <iostream>

namespace ipc {
template <typename scalar, int dim> class PointEdgeDistance {
public:
    PointEdgeDistance() = delete;
    PointEdgeDistance(const PointEdgeDistance&) = delete;
    PointEdgeDistance& operator=(const PointEdgeDistance&) = delete;

    static scalar point_point_sqr_distance(
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> a,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> b)
    {
        return (a - b).squaredNorm();
    }

    static scalar point_line_sqr_distance(
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> p,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e1)
    {
        if constexpr (dim == 2) {
            return Math<scalar>::sqr(Math<scalar>::cross2(e0 - p, e1 - p))
                / (e1 - e0).squaredNorm();
        } else {
            return (e0 - p).cross(e1 - p).squaredNorm() / (e1 - e0).squaredNorm();
        }
    }

    static scalar point_edge_sqr_distance(
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> p,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e1,
        const PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO)
    {
        switch (dtype) {
        case PointEdgeDistanceType::P_E:
            return point_line_sqr_distance(p, e0, e1);
        case PointEdgeDistanceType::P_E0:
            return point_point_sqr_distance(p, e0);
        case PointEdgeDistanceType::P_E1:
            return point_point_sqr_distance(p, e1);
        case PointEdgeDistanceType::AUTO:
        default:
            const Eigen::Vector<scalar, dim> t = e1 - e0;
            const Eigen::Vector<scalar, dim> pos = p - e0;
            const scalar s = pos.dot(t) / t.squaredNorm();
            return (pos - Math<scalar>::l_ns(s) * t).squaredNorm();
        }
    }

    static Eigen::Vector<scalar, dim> point_line_closest_point_direction(
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> p,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e1);

    static Eigen::Vector<scalar, dim> point_edge_closest_point_direction(
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> p,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<scalar, dim>> e1,
        const PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);
};

template <int dim> class PointEdgeDistanceDerivatives {
public:
    PointEdgeDistanceDerivatives() = delete;
    PointEdgeDistanceDerivatives(const PointEdgeDistanceDerivatives&) = delete;
    PointEdgeDistanceDerivatives&
    operator=(const PointEdgeDistanceDerivatives&) = delete;

    static std::tuple<Eigen::Vector<double, dim>, Eigen::Matrix<double, dim, 3 * dim>>
    point_line_closest_point_direction_grad(
        Eigen::ConstRef<Eigen::Vector<double, dim>> p,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e1);

    static std::tuple<
        Eigen::Vector<double, dim>,
        Eigen::Matrix<double, dim, 3 * dim>,
        std::array<Eigen::Matrix<double, 3 * dim, 3 * dim>, dim>>
    point_line_closest_point_direction_hessian(
        Eigen::ConstRef<Eigen::Vector<double, dim>> p,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e1);

    static std::tuple<Eigen::Vector<double, dim>, Eigen::Matrix<double, dim, 3 * dim>>
    point_edge_closest_point_direction_grad(
        Eigen::ConstRef<Eigen::Vector<double, dim>> p,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e1,
        const PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);

    static std::tuple<
        Eigen::Vector<double, dim>,
        Eigen::Matrix<double, dim, 3 * dim>,
        std::array<Eigen::Matrix<double, 3 * dim, 3 * dim>, dim>>
    point_edge_closest_point_direction_hessian(
        Eigen::ConstRef<Eigen::Vector<double, dim>> p,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e0,
        Eigen::ConstRef<Eigen::Vector<double, dim>> e1,
        const PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);
};
} // namespace ipc
