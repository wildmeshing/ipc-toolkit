#pragma once
#include "ipc/collision_mesh.hpp"
#include "ipc/candidates/edge_edge.hpp"
#include "ipc/high_order_contact/high_order_collisions.hpp"
#include "ipc/distance/point_point.hpp"
#include "ipc/distance/point_triangle.hpp"
#include "ipc/smooth_contact/distance/edge_edge.hpp"

namespace ipc
{
    enum class PointType
    {
        Vertex,
        Edge,
        Face
    };

    namespace PointPotentialHelper {
        double evaluate_potential_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params);

        double evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params,
            EdgeEdgeDistanceType dtype);

        template <typename ADType>
        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector4i> vids,
            Eigen::ConstRef<Eigen::Vector3<ADType>> q,
            EdgeEdgeDistanceType dtype);

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector4i> vids,
            Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q,
            EdgeEdgeDistanceType dtype);

        double evaluate_potential_at_face_center_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_face_center_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_face_center_with_cached_collisions(
            ConcatMatrixView<3> V_extended,
            Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
            const HighOrderCollisionDict<3>& collisions,
            const HighOrderContactParameters& params);
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

        HighOrderCollisionDict<3>
        build_collisions_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const;

        HighOrderCollisionDict<3>
        build_collisions_at_edge_edge_closest_point_advanced(
        const Eigen::MatrixXd& V,
            const index_t e0,
            const index_t e1) const;

        HighOrderCollisionDict<3>
        build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
            const index_t fid) const;

        const CollisionMesh& mesh;
        const Candidates& candidates;
        const HighOrderContactParameters params;
    };
}
