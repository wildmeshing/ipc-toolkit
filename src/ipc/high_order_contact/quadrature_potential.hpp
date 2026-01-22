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
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_vertex_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params);

        double evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>& collisions,
            const HighOrderContactParameters& params);

        double evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V_extended,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V_extended,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector4i> vids,
            const Eigen::Vector3<ADGrad<12>>& q);

        // Eigen::SparseMatrix<double> evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
        //     const Eigen::MatrixXd& V_extended,
        //     const unordered_map<std::array<index_t, 4>, std::shared_ptr<HighOrderCollision>>& collisions,
        //     const HighOrderContactParameters& params,
        //     Eigen::ConstRef<Eigen::Vector4i> vids,
        //     const Eigen::Vector3<ADHessian<12>>& q);

        double evaluate_potential_at_face_center_with_cached_collisions(
            const Eigen::MatrixXd& V_extended,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_face_center_with_cached_collisions(
            const Eigen::MatrixXd& V_extended,
            Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params);

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_face_center_with_cached_collisions(
            const Eigen::MatrixXd& V_extended,
            Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
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

        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
        build_collisions_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const;

        /// @brief Evaluate P(q) at a vertex vid
        double evaluate_potential_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const;

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const;

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const;

        unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>
        build_collisions_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
            const index_t e0,
            const index_t e1) const;

        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
        build_collisions_at_edge_edge_closest_point_advanced(
        const Eigen::MatrixXd& V,
            const index_t e0,
            const index_t e1) const;

        /// @brief Evaluate P(q) at the point on edge e0 that is closest to edge e1
        double evaluate_potential_at_edge_edge_closest_point(
            const Eigen::MatrixXd& V,
            const index_t e0,
            const index_t e1) const;

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_edge_edge_closest_point(
            const Eigen::MatrixXd& V,
            const index_t e0,
            const index_t e1) const;

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_edge_edge_closest_point(
            const Eigen::MatrixXd& V,
            const index_t e0,
            const index_t e1) const;

        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
        build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
            const index_t fid) const;

        double evaluate_potential_at_face_center(
            const Eigen::MatrixXd& V,
            const index_t fid) const;

        Eigen::SparseMatrix<double> evaluate_potential_gradient_at_face_center(
            const Eigen::MatrixXd& V,
            const index_t fid) const;

        Eigen::SparseMatrix<double> evaluate_potential_hessian_at_face_center(
            const Eigen::MatrixXd& V,
            const index_t fid) const;

        const CollisionMesh& mesh;
        const Candidates& candidates;
        const HighOrderContactParameters params;
    };

    class QuadraturePotential
    {
    public:
        // mollifier for uv
        template <typename T>
        static T mollified_identity(T x)
        {
            const double m = 1e-2;
            if (x <= 0)
                return T(0);
            if (x < m)
                return x * x * (2 * m - x) / (m * m);
            if (x <= 1 - m)
                return x;
            if (x < 1)
                return 1 - (1 - x) * (1 - x) * ((2 * m - 1) + x) / (m * m);

            return T(1);
        }

        QuadraturePotential(
            const CollisionMesh& mesh,
            const Eigen::MatrixXd& V,
            const double dhat);

        double evaluate_per_face(
            const Eigen::MatrixXd& V,
            const index_t face_id) const;

        Eigen::SparseMatrix<double> evaluate_per_face_gradient(
            const Eigen::MatrixXd& V,
            const index_t face_id) const;

        const CollisionMesh mesh;
        const double dhat;

        Candidates candidates;
        std::unique_ptr<PointPotential> point_potential;

        template <typename T = double>
        struct EdgePairClosestPoint
        {
            EdgePairClosestPoint(T uv0_, index_t e1_, T mollifier_)
            {
                uv0 = uv0_;
                e1 = e1_;

                mollifier = mollifier_;
                beta = mollified_identity(uv0);
            }

            EdgePairClosestPoint(T uv0_)
            {
                uv0 = uv0_;
                e1 = -1;

                mollifier = 1.;
                beta = mollified_identity(uv0);
            }

            T uv0;
            index_t e1;
            T mollifier;
            T beta;
        };
    };
}
