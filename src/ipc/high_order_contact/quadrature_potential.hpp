#pragma once
#include "ipc/collision_mesh.hpp"
#include "ipc/candidates/edge_edge.hpp"
#include "ipc/high_order_contact/high_order_collisions.hpp"
#include "ipc/distance/point_point.hpp"
#include "ipc/distance/point_triangle.hpp"
#include "ipc/smooth_contact/distance/edge_edge.hpp"

namespace ipc
{
    namespace PointPotentialHelper {
        double evaluate_potential_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const HighOrderCollisionDict<PointType::VERTEX>& collisions,
            const HighOrderContactParameters& params);

        Eigen::VectorXd evaluate_potential_gradient_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const HighOrderCollisionDict<PointType::VERTEX>& collisions,
            const HighOrderContactParameters& params);

        Eigen::MatrixXd evaluate_potential_hessian_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const HighOrderCollisionDict<PointType::VERTEX>& collisions,
            const HighOrderContactParameters& params,
            PSDProjectionMethod project_to_psd);

        double evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::EDGE>& collisions,
            const HighOrderContactParameters& params,
            EdgeEdgeDistanceType dtype);

        /// @brief Compute the gradient of P(q) for a point q
        /// @return The gradient vector with respect to collisions.m_vertex_ids
        /// @param V_extended Extended vertices matrix, with the point q appended to the last row
        /// @param collisions Primitives that are close in distance to point q
        /// @param q Closest point between two edges, together with the derivatives of q with respect to vids
        template <typename ADType>
        std::enable_if_t<IsADGrad<ADType>::value || IsADHessian<ADType>::value, Eigen::VectorXd>
        evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::EDGE>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector3<ADType>> q);

        Eigen::MatrixXd evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::EDGE>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q);

        double evaluate_potential_at_face_center_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params);

        Eigen::VectorXd evaluate_potential_gradient_at_face_center_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params);

        Eigen::MatrixXd evaluate_potential_hessian_at_face_center_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params,
            PSDProjectionMethod project_to_psd);
    }

    class PointPotential
    {
    public:

        constexpr static int r = 2;

        PointPotential(
            const CollisionMesh& mesh_,
            const Candidates& candidates_,
            const HighOrderContactParameters params_)
                : mesh(mesh_),
                  candidates(candidates_),
                  params(params_)
        {
        }

        HighOrderCollisionDict<PointType::VERTEX>
        build_collisions_at_vertex(
            const Eigen::MatrixXd& V,
            index_t vid) const;

        HighOrderCollisionDict<PointType::EDGE>
        build_collisions_at_edge_edge_closest_point_advanced(
        const Eigen::MatrixXd& V,
            index_t e0,
            index_t e1) const;

        HighOrderCollisionDict<PointType::FACE>
        build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
            index_t fid) const;

        const CollisionMesh& mesh;
        const Candidates& candidates;
        const HighOrderContactParameters params;
    };
}
