#include "quadrature_potential.hpp"

#include <absl/strings/str_format.h>

#include "ipc/candidates/candidates.hpp"
#include "ipc/distance/edge_edge.hpp"
#include "ipc/high_order_contact/high_order_collisions_builder.hpp"
#include "ipc/geometry/area.hpp"
#include "ipc/smooth_contact/distance/point_face.hpp"
#include "ipc/smooth_contact/distance/mollifier.hpp"

namespace ipc
{
    namespace
    {
        template <typename THash, typename TCollision>
        void insert_pair(
            unordered_map<THash, std::shared_ptr<TCollision>>& collisions,
            std::shared_ptr<TCollision> collision)
        {
            if (auto iter = collisions.find(collision->get_typed_hash()); iter != collisions.end()) {
                iter->second->weight += collision->weight;
                if (iter->second->weight == 0) {
                    collisions.erase(iter);
                }
            }
            else {
                collisions[collision->get_typed_hash()] = collision;
            }
        }
    }

    HighOrderCollisionDict<3>
    PointPotential::build_collisions_at_vertex(
        const Eigen::MatrixXd& V,
        const index_t vid) const
    {
        HighOrderCollisionDict<3> pairs;

        const auto& v_set = candidates.vv_set(vid);
        const auto& e_set = candidates.ve_set(vid);
        const auto& f_set = candidates.vf_set(vid);

        for (const auto& other_f : f_set) {
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<
                3>::reduce_point_triangle_collision(
                FaceVertexCandidate(other_f, vid),
                params, mesh, V)) {
                if (pair->is_active())
                    insert_pair(pairs, pair);
            }
        }

        for (const auto& other_e : e_set) {
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                EdgeVertexCandidate(other_e, vid),
                params, mesh, V)) {
                pair->weight = -1;
                if (pair->is_active())
                    insert_pair(pairs, pair);
            }
        }

        for (const auto& other_v : v_set) {
            std::shared_ptr<HighOrderCollision> pair = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                std::min(vid, other_v), std::max(vid, other_v),
                mesh, params, params.dhat, V);
            if (pair->is_active()) {
                insert_pair(pairs, pair);
            }
        }

        return pairs;
    }

    double PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params)
    {
        double potential = 0;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            potential += cc->weight * (*cc)(cc->dof(V), params);
        }

        return potential;
    }

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::VectorXd g = cc->weight * cc->gradient(cc->dof(V), params);
            assert(g.size() == cc->vertex_ids().size() * 3);
            for (index_t i = 0; i < cc->vertex_ids().size(); i++) {
                for (index_t d = 0; d < 3; d++) {
                    triplets.emplace_back(3 * cc->vertex_ids()[i] + d, 0, g(3 * i + d));
                }
            }
        }

        Eigen::SparseMatrix<double> grad(V.size(), 1);
        grad.setFromTriplets(triplets.begin(), triplets.end());

        return grad;
    }

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::MatrixXd h = cc->weight * cc->hessian(cc->dof(V), params);
            assert(h.rows() == cc->vertex_ids().size() * 3);
            assert(h.cols() == cc->vertex_ids().size() * 3);
            for (index_t i = 0; i < cc->vertex_ids().size(); i++) {
                for (index_t di = 0; di < 3; di++) {
                    for (index_t j = 0; j < cc->vertex_ids().size(); j++) {
                        for (index_t dj = 0; dj < 3; dj++) {
                            triplets.emplace_back(
                                3 * cc->vertex_ids()[i] + di,
                                3 * cc->vertex_ids()[j] + dj,
                                h(3 * i + di, 3 * j + dj));
                        }
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> hess(V.size(), V.size());
        hess.setFromTriplets(triplets.begin(), triplets.end());

        return hess;
    }

    HighOrderCollisionDict<3>
    PointPotential::build_collisions_at_edge_edge_closest_point_advanced(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        HighOrderCollisionDict<3> pairs;

        const auto& v_set = candidates.ev_set(e0);
        const auto& e_set = candidates.ee_set(e0);
        const auto& f_set = candidates.ef_set(e0);

        // Compute closest point
        const index_t e00 = mesh.edges()(e0, 0);
        const index_t e01 = mesh.edges()(e0, 1);
        const index_t e10 = mesh.edges()(e1, 0);
        const index_t e11 = mesh.edges()(e1, 1);
        const EdgeEdgeDistanceType dtype = edge_edge_distance_type(
            V.row(e00), V.row(e01),
            V.row(e10), V.row(e11)
        );
        if (dtype != EdgeEdgeDistanceType::EA_EB && dtype != EdgeEdgeDistanceType::EA_EB0 && dtype !=
            EdgeEdgeDistanceType::EA_EB1) {
            std::cout << "positions at error\n";
            std::cout << std::fixed << std::setprecision(15) << V({e00, e01, e10, e11}, Eigen::all) << std::endl;
            std::cout << "dtype " << static_cast<int>(dtype) << std::endl;
            log_and_throw_error("Can only handle EA_EB* distance type!");
        }

        if (is_parallel_edge_edge(V.row(e00), V.row(e01),
                                  V.row(e10), V.row(e11))) {
            log_and_throw_error("Cannot handle parallel edge!");
        }

        if (edge_edge_distance(V.row(e00), V.row(e01),
                               V.row(e10), V.row(e11), dtype) >= params.dhat * params.dhat) {
            return pairs;
        }

        double closest_uv = 0;
        if (dtype == EdgeEdgeDistanceType::EA_EB) {
            closest_uv = line_line_closest_point_pairs_uv<double>(
                V.row(e00), V.row(e01),
                V.row(e10), V.row(e11))(0);
        }
        else if (dtype == EdgeEdgeDistanceType::EA_EB0) {
            Eigen::RowVector3d p = V.row(e10);
            Eigen::RowVector3d d = p - V.row(e00);
            Eigen::RowVector3d t = V.row(e01) - V.row(e00);
            closest_uv = d.dot(t) / t.squaredNorm();
        }
        else if (dtype == EdgeEdgeDistanceType::EA_EB1) {
            Eigen::RowVector3d p = V.row(e11);
            Eigen::RowVector3d d = p - V.row(e00);
            Eigen::RowVector3d t = V.row(e01) - V.row(e00);
            closest_uv = d.dot(t) / t.squaredNorm();
        }
        else
            log_and_throw_error("Invalid dtype!");

        if (!std::isfinite(closest_uv)) {
            log_and_throw_error("Potentially parallel edges!");
        }

        const index_t vid = V.rows();

        Eigen::MatrixXd V_(V.rows() + 1, 3);
        V_.topRows(V.rows()) = V;
        V_.row(vid) = closest_uv * (V.row(e01) - V.row(e00)) + V.row(e00);

        // for (const auto& other_v : v_set) {
        for (index_t other_v = 0; other_v < mesh.num_vertices(); ++other_v) {
            std::shared_ptr<HighOrderCollision> pair = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                vid, other_v, mesh, params, params.dhat, V_);

            if (pair->is_active()) {
                insert_pair(pairs, pair);
            }
        }

        // for (const auto& other_e : e_set) {
        for (index_t other_e = 0; other_e < mesh.num_edges(); ++other_e) {
            if (other_e == e0)
                continue;

            auto pair = std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                other_e, vid, mesh, params, params.dhat, V_);

            if (!pair->is_active()) {
                continue;
            }

            auto dtype2 = point_edge_distance_type(V_.row(vid), V_.row(mesh.edges()(other_e, 0)),
                                                   V_.row(mesh.edges()(other_e, 1)));

            switch (dtype2) {
            case PointEdgeDistanceType::P_E0:
                {
                    std::shared_ptr<HighOrderCollision> pair2 = std::make_shared<HighOrderCollisionTemplate<
                        Vertex3, Vertex3>>(
                        vid, mesh.edges()(other_e, 0), mesh, params, params.dhat, V_);
                    pair2->weight = -1;
                    insert_pair(pairs, pair2);
                    break;
                }
            case PointEdgeDistanceType::P_E1:
                {
                    std::shared_ptr<HighOrderCollision> pair2 = std::make_shared<HighOrderCollisionTemplate<
                        Vertex3, Vertex3>>(
                        vid, mesh.edges()(other_e, 1), mesh, params, params.dhat, V_);
                    pair2->weight = -1;
                    insert_pair(pairs, pair2);
                    break;
                }
            case PointEdgeDistanceType::P_E:
                {
                    pair->weight = -1;
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                    break;
                }
            default:
                assert(false);
                break;
            }
        }

        // for (const auto& other_f : f_set) {
        for (index_t other_f = 0; other_f < mesh.num_faces(); ++other_f) {
            if (mesh.edges_to_faces()(e0, 0) == other_f || mesh.edges_to_faces()(e0, 1) == other_f)
                continue;

            auto pair = std::make_shared<HighOrderCollisionTemplate<Face3P1, Vertex3>>(
                other_f, vid, mesh, params, params.dhat, V_);

            if (!pair->is_active()) {
                continue;
            }

            auto dtype2 = point_triangle_distance_type(V_.row(vid), V_.row(mesh.faces()(other_f, 0)),
                                                       V_.row(mesh.faces()(other_f, 1)),
                                                       V_.row(mesh.faces()(other_f, 2)));

            switch (dtype2) {
            case PointTriangleDistanceType::P_T0:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(
                                    std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                                        vid, mesh.faces()(other_f, 0), mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_T1:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(
                                    std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                                        vid, mesh.faces()(other_f, 1), mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_T2:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(
                                    std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                                        vid, mesh.faces()(other_f, 2), mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_E0:
                {
                    insert_pair(pairs,
                                std::shared_ptr<HighOrderCollision>(
                                    std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                                        mesh.faces_to_edges()(other_f, 0), vid, mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_E1:
                {
                    insert_pair(pairs,
                                std::shared_ptr<HighOrderCollision>(
                                    std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                                        mesh.faces_to_edges()(other_f, 1), vid, mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_E2:
                {
                    insert_pair(pairs,
                                std::shared_ptr<HighOrderCollision>(
                                    std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                                        mesh.faces_to_edges()(other_f, 2), vid, mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_T:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                    break;
                }
            default:
                assert(false);
                break;
            }
        }

        return pairs;
    }

    double PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
        ConcatMatrixView<3> V_extended,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params,
        EdgeEdgeDistanceType dtype)
    {
        double potential = 0;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            double term = (*cc)(cc->dof(V_extended), params);
            assert(std::isfinite(term));
            potential += cc->weight * term;
        }

        return potential;
    }

    template <typename ADType>
    Eigen::SparseMatrix<double>
    PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
        ConcatMatrixView<3> V_extended,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector4i> vids,
        Eigen::ConstRef<Eigen::Vector3<ADType>> q,
        EdgeEdgeDistanceType dtype)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::VectorXd g = cc->weight * cc->gradient(cc->dof(V_extended), params);
            for (index_t i = 0; i < cc->vertex_ids().size(); i++) {
                const index_t global_id = cc->vertex_ids()[i];
                if (global_id == n_real_vertices) {
                    const Vector12d local_grad = (q(0) * g(3 * i + 0) + q(1) * g(3 * i + 1) + q(2) * g(3 * i + 2)).grad;
                    // distribute grad wrt virtual vertex to real edge vertices
                    for (index_t lv = 0; lv < 4; lv++) {
                        for (index_t d = 0; d < 3; d++) {
                            triplets.emplace_back(vids[lv] * 3 + d, 0, local_grad(lv * 3 + d));
                        }
                    }
                }
                else {
                    assert(global_id < n_real_vertices);
                    for (index_t d = 0; d < 3; d++) {
                        triplets.emplace_back(3 * cc->vertex_ids()[i] + d, 0, g(3 * i + d));
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> grad(n_real_vertices * 3, 1);
        grad.setFromTriplets(triplets.begin(), triplets.end());

        return grad;
    }

    template
    Eigen::SparseMatrix<double>
    PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<ADGrad<12>>(
        ConcatMatrixView<3> V_extended,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector4i> vids,
        Eigen::ConstRef<Eigen::Vector3<ADGrad<12>>> q,
        EdgeEdgeDistanceType dtype);

    template
    Eigen::SparseMatrix<double>
    PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<ADHessian<12>>(
        ConcatMatrixView<3> V_extended,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector4i> vids,
        Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q,
        EdgeEdgeDistanceType dtype);

    Eigen::SparseMatrix<double>
    PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
        ConcatMatrixView<3> V_extended,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector4i> vids,
        Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q,
        EdgeEdgeDistanceType dtype)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::MatrixXd h = cc->weight * cc->hessian(cc->dof(V_extended), params);
            Eigen::VectorXd g = cc->weight * cc->gradient(cc->dof(V_extended), params);

            for (index_t i = 0; i < cc->vertex_ids().size(); i++) {
                const index_t gi = cc->vertex_ids()[i];
                for (index_t j = 0; j < cc->vertex_ids().size(); j++) {
                    const index_t gj = cc->vertex_ids()[j];
                    if (gi == n_real_vertices && gj == n_real_vertices) {
                        assert(i == j);
                        // distribute derivatives wrt virtual vertex to real edge vertices
                        Matrix12d local_hess = Matrix12d::Zero();
                        {
                            Eigen::Matrix<double, 3, 12> tmp_g;
                            tmp_g << q(0).grad.transpose(), q(1).grad.transpose(), q(2).grad.transpose();
                            local_hess += tmp_g.transpose() * h.block<3, 3>(3 * i, 3 * j) * tmp_g;

                            for (int d = 0; d < 3; d++) {
                                local_hess += q(d).Hess * g(3 * i + d);
                            }
                        }

                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t li = 0; li < 4; li++) {
                                    for (index_t lj = 0; lj < 4; lj++) {
                                        triplets.emplace_back(vids[li] * 3 + di, vids[lj] * 3 + dj,
                                                              local_hess(3 * li + di, 3 * lj + dj));
                                    }
                                }
                            }
                        }
                    }
                    else if (gi == n_real_vertices) {
                        Eigen::Matrix<double, 12, 3> local_hess;
                        local_hess.setZero();
                        {
                            Eigen::Matrix<double, 3, 12> tmp_g;
                            tmp_g << q(0).grad.transpose(), q(1).grad.transpose(), q(2).grad.transpose();
                            local_hess += tmp_g.transpose() * h.block<3, 3>(3 * i, 3 * j);
                        }
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t li = 0; li < 4; li++) {
                                    triplets.emplace_back(vids[li] * 3 + di, gj * 3 + dj, local_hess(3 * li + di, dj));
                                    triplets.emplace_back(gj * 3 + dj, vids[li] * 3 + di, local_hess(3 * li + di, dj));
                                }
                            }
                        }
                    }
                    else if (gj == n_real_vertices) {
                        // Already handled in (gi == n_real_vertices) case
                    }
                    else {
                        assert(gi < n_real_vertices);
                        assert(gj < n_real_vertices);
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                triplets.emplace_back(3 * gi + di, 3 * gj + dj, h(3 * i + di, 3 * j + dj));
                            }
                        }
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> hess(n_real_vertices * 3, n_real_vertices * 3);
        hess.setFromTriplets(triplets.begin(), triplets.end());

        return hess;
    }

    HighOrderCollisionDict<3>
    PointPotential::build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
        const index_t fid) const
    {
        // the fake vertex id
        const index_t vid = V.rows();

        Eigen::MatrixXd V_(V.rows() + 1, 3);
        V_.topRows(V.rows()) = V;
        V_.row(vid) = (V.row(mesh.faces()(fid, 0)) + V.row(mesh.faces()(fid, 1)) + V.row(mesh.faces()(fid, 2))) / 3.;

        HighOrderCollisionDict<3> pairs;

        const auto& v_set = candidates.fv_set(fid);
        const auto& e_set = candidates.fe_set(fid);
        const auto& f_set = candidates.ff_set(fid);

        for (const auto& other_f : f_set) {
            assert(other_f != fid);
            if (auto pair = HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
                FaceVertexCandidate(other_f, vid),
                params, mesh, V_); pair->is_active()) {
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        for (const auto& other_e : e_set) {
            if (auto pair = HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                EdgeVertexCandidate(other_e, vid),
                params, mesh, V_); pair->is_active()) {
                pair->weight = -1;
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        for (const auto& other_v : v_set) {
            auto pair = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                std::min(vid, other_v), std::max(vid, other_v),
                mesh, params, params.dhat, V_);
            if (pair->is_active()) {
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        return pairs;
    }

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
        ConcatMatrixView<3> V_extended,
        Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::VectorXd g = cc->weight * cc->gradient(cc->dof(V_extended), params);
            for (index_t i = 0; i < cc->vertex_ids().size(); i++) {
                const index_t global_id = cc->vertex_ids()[i];
                if (global_id == n_real_vertices) {
                    // distribute grad wrt virtual vertex to real face vertices
                    for (index_t d = 0; d < 3; d++) {
                        for (index_t lv = 0; lv < 3; lv++) {
                            triplets.emplace_back(vids[lv] * 3 + d, 0, g(3 * i + d) / 3.);
                        }
                    }
                }
                else {
                    assert(global_id < n_real_vertices);
                    for (index_t d = 0; d < 3; d++) {
                        triplets.emplace_back(3 * global_id + d, 0, g(3 * i + d));
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> grad(n_real_vertices * 3, 1);
        grad.setFromTriplets(triplets.begin(), triplets.end());

        return grad;
    }

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(
        ConcatMatrixView<3> V_extended,
        Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::MatrixXd h = cc->weight * cc->hessian(cc->dof(V_extended), params);

            for (index_t i = 0; i < cc->vertex_ids().size(); i++) {
                const index_t gi = cc->vertex_ids()[i];
                for (index_t j = 0; j < cc->vertex_ids().size(); j++) {
                    const index_t gj = cc->vertex_ids()[j];
                    if (gi == n_real_vertices && gj == n_real_vertices) {
                        // distribute grad wrt virtual vertex to real face vertices
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t li = 0; li < 3; li++) {
                                    for (index_t lj = 0; lj < 3; lj++) {
                                        triplets.emplace_back(vids[li] * 3 + di, vids[lj] * 3 + dj,
                                                              h(3 * i + di, 3 * j + dj) / 9.);
                                    }
                                }
                            }
                        }
                    }
                    else if (gi == n_real_vertices) {
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t li = 0; li < 3; li++) {
                                    triplets.emplace_back(vids[li] * 3 + di, gj * 3 + dj,
                                                          h(3 * i + di, 3 * j + dj) / 3.);
                                }
                            }
                        }
                    }
                    else if (gj == n_real_vertices) {
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t lj = 0; lj < 3; lj++) {
                                    triplets.emplace_back(gi * 3 + di, vids[lj] * 3 + dj,
                                                          h(3 * i + di, 3 * j + dj) / 3.);
                                }
                            }
                        }
                    }
                    else {
                        assert(gi < n_real_vertices);
                        assert(gj < n_real_vertices);
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                triplets.emplace_back(3 * gi + di, 3 * gj + dj, h(3 * i + di, 3 * j + dj));
                            }
                        }
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> hess(n_real_vertices * 3, n_real_vertices * 3);
        hess.setFromTriplets(triplets.begin(), triplets.end());

        return hess;
    }

    double PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
        ConcatMatrixView<3> V_extended,
        const HighOrderCollisionDict<3>& collisions,
        const HighOrderContactParameters& params)
    {
        double potential = 0;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            potential += cc->weight * (*cc)(cc->dof(V_extended), params);
        }

        return potential;
    }
}
