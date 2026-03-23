#pragma once
#include <array>
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
            VertexMatrixView<3> V_extended,
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
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::EDGE>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector3<ADType>> q);

        Eigen::MatrixXd evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::EDGE>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q);

        double evaluate_potential_at_face_center_with_cached_collisions(
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params);

        Eigen::VectorXd evaluate_potential_gradient_at_face_center_with_cached_collisions(
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params);

        Eigen::MatrixXd evaluate_potential_hessian_at_face_center_with_cached_collisions(
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params,
            PSDProjectionMethod project_to_psd);

        /// @brief Gradient of the face-interior potential for an arbitrary
        ///   interior quadrature point q = λ0·v0 + λ1·v1 + λ2·v2.
        /// @param lambda Barycentric coordinates of the interior point.
        ///   The chain-rule factors λk replace the 1/3 used for the centroid.
        Eigen::VectorXd evaluate_potential_gradient_at_face_interior_point_with_cached_collisions(
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params,
            const std::array<double, 3>& lambda);

        /// @brief Hessian of the face-interior potential for an arbitrary
        ///   interior quadrature point q = λ0·v0 + λ1·v1 + λ2·v2.
        Eigen::MatrixXd evaluate_potential_hessian_at_face_interior_point_with_cached_collisions(
            VertexMatrixView<3> V_extended,
            const HighOrderCollisionDict<PointType::FACE>& collisions,
            const HighOrderContactParameters& params,
            const std::array<double, 3>& lambda,
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
            : mesh(mesh_)
            , candidates(candidates_)
            , params(params_)
        {
        }

        std::unique_ptr<HighOrderCollisionDict<PointType::VERTEX>>
        build_collisions_at_vertex(const Eigen::MatrixXd& V, index_t vid, size_t& num_collision_pairs) const;

        std::unique_ptr<HighOrderCollisionDict<PointType::EDGE>>
        build_collisions_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
            index_t e0,
            index_t e1,
            EdgeEdgeDistanceType dtype,
            size_t& num_collision_pairs) const;

        std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>
        build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
            index_t fid,
            size_t& num_collision_pairs) const;

        std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>
        build_collisions_at_face_interior_point(
            const Eigen::MatrixXd& V,
            index_t fid,
            const std::array<double, 3>& lambda,
            size_t& num_collision_pairs) const;

        const CollisionMesh& mesh;
        const Candidates& candidates;
        const HighOrderContactParameters params;
    };
}
