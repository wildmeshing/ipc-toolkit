#include "quadrature_potential.hpp"

#include <array>
#include "absl/strings/internal/str_format/extension.h"
#include "ipc/candidates/candidates.hpp"
#include "ipc/distance/edge_edge.hpp"
#include "ipc/distance/point_edge.hpp"
#include "ipc/distance/point_triangle.hpp"
#include "ipc/distance/distance_type.hpp"
#include "ipc/high_order_contact/high_order_collisions_builder.hpp"
#include "ipc/smooth_contact/distance/mollifier.hpp"

namespace ipc {
    namespace {
        template <typename KeyType, typename ValueType>
        void
        insert_pair(unordered_map<KeyType, ValueType>& map, ValueType&& collision)
        {
            if (auto iter = map.find(collision->get_typed_hash());
                iter != map.end()) {
                iter->second->weight += collision->weight;
                if (iter->second->weight == 0) {
                    map.erase(iter);
                }
            } else {
                map[collision->get_typed_hash()] = std::move(collision);
            }
        }
    } // namespace

    std::unique_ptr<HighOrderCollisionDict<PointType::VERTEX>>
    PointPotential::build_collisions_at_vertex(
        const Eigen::MatrixXd& V, const index_t vid, size_t& num_collision_pairs) const
    {
        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>> pairs;
        num_collision_pairs = 0;

        const auto& v_set = candidates.vv_set(vid);
        const auto& e_set = candidates.ve_set(vid);
        const auto& f_set = candidates.vf_set(vid);

        const VertexMatrixView<3> V_view(V);

        const bool src_is_obstacle = mesh.is_obstacle_vertex(vid);
        const bool filter_obstacles = src_is_obstacle
            && params.integration_type != HighOrderContactParameters::IntegrationType::BRUTE_FORCE;

        for (const auto& other_f : f_set) {
            if (filter_obstacles && mesh.is_obstacle_face(other_f)) continue;
            ++num_collision_pairs;
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<
                3>::reduce_point_triangle_collision(
                FaceVertexCandidate(other_f, vid),
                params, mesh, V_view)) {
                insert_pair(pairs, std::move(pair));
            }
        }

        for (const auto& other_e : e_set) {
            if (filter_obstacles && mesh.is_obstacle_edge(other_e)) continue;
            ++num_collision_pairs;
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                EdgeVertexCandidate(other_e, vid),
                params, mesh, V_view)) {
                pair->weight = -1;
                insert_pair(pairs, std::move(pair));
            }
        }

        for (const auto& other_v : v_set) {
            if (filter_obstacles && mesh.is_obstacle_vertex(other_v)) continue;
            if ((V.row(vid) - V.row(other_v)).squaredNorm() >= params.dhat * params.dhat) {
                continue;
            }
            std::shared_ptr<HighOrderCollision> pair = std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                vid, other_v, mesh);
            ++num_collision_pairs;
            insert_pair(pairs, std::move(pair));
        }
        std::unique_ptr<HighOrderCollisionDict<PointType::VERTEX>> collisions = std::make_unique<HighOrderCollisionDict<PointType::VERTEX>>();
        collisions->initialize(std::vector<index_t>{vid}, std::vector<index_t>{vid}, pairs);
        return collisions;
    }

    double PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const HighOrderCollisionDict<PointType::VERTEX>& collisions,
        const HighOrderContactParameters& params)
    {
        double potential = 0;
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            potential += cc.weight * cc(cc.dof(V), params);
        }

        return potential;
    }

    Eigen::VectorXd PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const HighOrderCollisionDict<PointType::VERTEX>& collisions,
        const HighOrderContactParameters& params)
    {
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::VectorXd g = cc.weight * cc.gradient(cc.dof(V), params);
            assert(g.size() == cc.num_vertices() * 3);
            for (index_t j = 0; j < cc.num_vertices(); j++) {
                grad.segment<3>(3 * collisions.vertex_ids_inverse(cc.vertex_id(j))) += g.segment<3>(3 * j);
            }
        }

        return grad;
    }

    Eigen::MatrixXd PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const HighOrderCollisionDict<PointType::VERTEX>& collisions,
        const HighOrderContactParameters& params,
        PSDProjectionMethod project_to_psd)
    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(collisions.vertex_ids().size() * 3, collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::MatrixXd h = cc.hessian(cc.dof(V), params);
            // The following code can be used only if all weights are positive
            // if (project_to_psd != PSDProjectionMethod::NONE) {
            //     h = ipc::project_to_psd(h, project_to_psd);
            // }
            h *= cc.weight;

            assert(h.rows() == cc.num_vertices() * 3);
            assert(h.cols() == cc.num_vertices() * 3);
            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t li = collisions.vertex_ids_inverse(cc.vertex_id(i));
                for (index_t j = 0; j < cc.num_vertices(); j++) {
                    const index_t lj = collisions.vertex_ids_inverse(cc.vertex_id(j));
                    H.block<3, 3>(3 * li, 3 * lj) += h.block<3, 3>(3 * i, 3 * j);
                }
            }
        }

        if (project_to_psd != PSDProjectionMethod::NONE) {
            H = ipc::project_to_psd(H, project_to_psd);
        }
        return H;
    }

    std::unique_ptr<HighOrderCollisionDict<PointType::EDGE>>
    PointPotential::build_collisions_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1,
        EdgeEdgeDistanceType dtype,
        size_t& num_collision_pairs) const
    {
        const auto& v_set = candidates.ev_set(e0);
        const auto& e_set = candidates.ee_set(e0);
        const auto& f_set = candidates.ef_set(e0);

        // Compute closest point
        const index_t e00 = mesh.edges()(e0, 0);
        const index_t e01 = mesh.edges()(e0, 1);
        const index_t e10 = mesh.edges()(e1, 0);
        const index_t e11 = mesh.edges()(e1, 1);

#ifndef NDEBUG
        if (dtype != EdgeEdgeDistanceType::EA_EB && dtype != EdgeEdgeDistanceType::EA_EB0 && dtype !=
            EdgeEdgeDistanceType::EA_EB1) {
            log_and_throw_error("Can only handle EA_EB* distance type!");
        }
#endif

        unordered_map<std::array<int, 3>, std::shared_ptr<HighOrderCollision>> pairs;
        num_collision_pairs = 0;

        if (edge_edge_distance(V.row(e00), V.row(e01),
                               V.row(e10), V.row(e11), dtype) < params.dbar * params.dbar) {
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

            const index_t vid = V.rows(); // virtual vertex

            // Eigen::MatrixXd V_(V.rows() + 1, 3);
            // V_.topRows(V.rows()) = V;
            // V_.row(vid) = closest_uv * (V.row(e01) - V.row(e00)) + V.row(e00);
            const Eigen::RowVector3d ee_closest_point = closest_uv * (V.row(e01) - V.row(e00)) + V.row(e00);
            VertexMatrixView<3> V_(V, ee_closest_point);

            const bool src_is_obstacle_e = mesh.is_obstacle_edge(e0);
            const bool filter_obstacles_e = src_is_obstacle_e
                && params.integration_type != HighOrderContactParameters::IntegrationType::BRUTE_FORCE;

            for (const auto& other_v : v_set) {
                if (filter_obstacles_e && mesh.is_obstacle_vertex(other_v)) continue;
                if ((V_(vid) - V_(other_v)).squaredNorm() >= params.dbar * params.dbar) {
                    continue;
                }
                auto pair = std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                    vid, other_v, mesh);
                ++num_collision_pairs;
                pair->flag_as_safety();
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }

            for (const auto& other_e : e_set) {
                if (other_e == e0) continue;
                if (filter_obstacles_e && mesh.is_obstacle_edge(other_e)) continue;

                auto dtype2 = point_edge_distance_type(V_(vid), V_(mesh.edges()(other_e, 0)),
                                                       V_(mesh.edges()(other_e, 1)));

                const double dist_sqr = point_edge_distance(V_(vid), V_(mesh.edges()(other_e, 0)),
                                                       V_(mesh.edges()(other_e, 1)), dtype2);

                if (dist_sqr >= params.dbar * params.dbar) {
                    continue;
                }

                switch (dtype2) {
                case PointEdgeDistanceType::P_E0:
                    {
                        auto pair = std::make_shared<HighOrderCollision3DTemplate<
                            Vertex3, Vertex3>>(
                            vid, mesh.edges()(other_e, 0), mesh);
                        ++num_collision_pairs;
                        pair->weight = -1;
                        pair->flag_as_safety();
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointEdgeDistanceType::P_E1:
                    {
                        auto pair = std::make_shared<HighOrderCollision3DTemplate<
                            Vertex3, Vertex3>>(
                            vid, mesh.edges()(other_e, 1), mesh);
                        ++num_collision_pairs;
                        pair->weight = -1;
                        pair->flag_as_safety();
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointEdgeDistanceType::P_E:
                    {
                        auto pair = std::make_shared<HighOrderCollision3DTemplate<
                            Edge3P1, Vertex3>>(
                            other_e, vid, mesh);
                        ++num_collision_pairs;
                        pair->weight = -1;
                        pair->flag_as_safety();
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                default:
                    assert(false);
                    break;
                }
            }

            for (const auto& other_f : f_set) {
                if (mesh.edges_to_faces()(e0, 0) == other_f || mesh.edges_to_faces()(e0, 1) == other_f) continue;
                if (filter_obstacles_e && mesh.is_obstacle_face(other_f)) continue;

                auto dtype2 = point_triangle_distance_type(V_(vid), V_(mesh.faces()(other_f, 0)),
                                                           V_(mesh.faces()(other_f, 1)),
                                                           V_(mesh.faces()(other_f, 2)));

                const double dist_sqr = point_triangle_distance(V_(vid), V_(mesh.faces()(other_f, 0)),
                                                           V_(mesh.faces()(other_f, 1)),
                                                           V_(mesh.faces()(other_f, 2)), dtype2);

                if (dist_sqr >= params.dbar * params.dbar) {
                    continue;
                }

                switch (dtype2) {
                case PointTriangleDistanceType::P_T0:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                            vid, mesh.faces()(other_f, 0), mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointTriangleDistanceType::P_T1:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                            vid, mesh.faces()(other_f, 1), mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointTriangleDistanceType::P_T2:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                            vid, mesh.faces()(other_f, 2), mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointTriangleDistanceType::P_E0:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
                            mesh.faces_to_edges()(other_f, 0), vid, mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointTriangleDistanceType::P_E1:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
                            mesh.faces_to_edges()(other_f, 1), vid, mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointTriangleDistanceType::P_E2:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
                            mesh.faces_to_edges()(other_f, 2), vid, mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                case PointTriangleDistanceType::P_T:
                    {
                        ++num_collision_pairs;
                        auto pair =
                            std::make_shared<HighOrderCollision3DTemplate<Face3P1, Vertex3>>(
                            other_f, vid, mesh);
                        pair->flag_as_safety();    
                        insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
                        break;
                    }
                default:
                    assert(false);
                    break;
                }
            }
        }

        std::unique_ptr<HighOrderCollisionDict<PointType::EDGE>> collisions = std::make_unique<HighOrderCollisionDict<PointType::EDGE>>();
        collisions->initialize(std::vector<index_t>{e0, e1}, std::vector{e00, e01, e10, e11}, pairs);
        collisions->set_ee_dtype(dtype);
        return collisions;
    }

    double PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::EDGE>& collisions,
        const HighOrderContactParameters& params,
        EdgeEdgeDistanceType dtype)
    {
        double potential = 0;
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            double term = cc(cc.dof(V_extended), params);
            assert(std::isfinite(term));
            potential += cc.weight * term;
        }

        return potential;
    }

    template <typename ADType>
    std::enable_if_t<IsADGrad<ADType>::value || IsADHessian<ADType>::value, Eigen::VectorXd>
    PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::EDGE>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector3<ADType>> q)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::VectorXd g = cc.weight * cc.gradient(cc.dof(V_extended), params);
            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t global_id = cc.vertex_id(i);
                if (global_id == n_real_vertices) {
                    const Vector12d local_grad = (q(0) * g(3 * i + 0) + q(1) * g(3 * i + 1) + q(2) * g(3 * i + 2)).grad;
                    // distribute grad wrt virtual vertex to real edge vertices
                    for (index_t lv = 0; lv < 4; lv++) {
                        grad.segment<3>(3 * collisions.primary_local_ids()[lv]) += local_grad.segment<3>(lv * 3);
                    }
                }
                else {
                    assert(global_id < n_real_vertices);
                    grad.segment<3>(3 * collisions.vertex_ids_inverse(global_id)) += g.segment<3>(i * 3);
                }
            }
        }

        return grad;
    }

    template
    Eigen::VectorXd
    PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<ADGrad<12>>(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::EDGE>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector3<ADGrad<12>>> q);

    template
    Eigen::VectorXd
    PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<ADHessian<12>>(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::EDGE>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q);

    Eigen::MatrixXd
    PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::EDGE>& collisions,
        const HighOrderContactParameters& params,
        Eigen::ConstRef<Eigen::Vector3<ADHessian<12>>> q)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(collisions.vertex_ids().size() * 3, collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            const Eigen::VectorXd cc_dof = cc.dof(V_extended);
            Eigen::MatrixXd h = cc.weight * cc.hessian(cc_dof, params);
            Eigen::VectorXd g = cc.weight * cc.gradient(cc_dof, params);

            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t gi = cc.vertex_id(i);
                for (index_t j = 0; j < cc.num_vertices(); j++) {
                    const index_t gj = cc.vertex_id(j);
                    if (gi == n_real_vertices && gj == n_real_vertices) {
                        assert(i == j);
                        // distribute derivatives wrt virtual vertex to real edge vertices
                        Matrix12d local_hess;
                        {
                            Eigen::Matrix<double, 3, 12> tmp_g;
                            tmp_g << q(0).grad.transpose(), q(1).grad.transpose(), q(2).grad.transpose();
                            local_hess = tmp_g.transpose() * h.block<3, 3>(3 * i, 3 * j) * tmp_g;

                            for (int d = 0; d < 3; d++) {
                                local_hess += q(d).Hess * g(3 * i + d);
                            }
                        }

                        for (index_t li = 0; li < 4; li++) {
                            for (index_t lj = 0; lj < 4; lj++) {
                                H.block<3, 3>(collisions.primary_local_ids()[li] * 3,
                                    collisions.primary_local_ids()[lj] * 3) +=
                                        local_hess.block<3, 3>(3 * li, 3 * lj);
                            }
                        }
                    }
                    else if (gi == n_real_vertices) {
                        Eigen::Matrix<double, 12, 3> local_hess;
                        {
                            Eigen::Matrix<double, 3, 12> tmp_g;
                            tmp_g << q(0).grad.transpose(), q(1).grad.transpose(), q(2).grad.transpose();
                            local_hess = tmp_g.transpose() * h.block<3, 3>(3 * i, 3 * j);
                        }
                        for (index_t li = 0; li < 4; li++) {
                            const index_t lli = collisions.primary_local_ids()[li];
                            H.block<3, 3>(lli * 3, collisions.vertex_ids_inverse(gj) * 3) += local_hess.block<3, 3>(3 * li, 0);
                            H.block<3, 3>(collisions.vertex_ids_inverse(gj) * 3, lli * 3) += local_hess.block<3, 3>(3 * li, 0).transpose();
                        }
                    }
                    else if (gj == n_real_vertices) {
                        // Already handled in (gi == n_real_vertices) case
                    }
                    else {
                        assert(gi < n_real_vertices);
                        assert(gj < n_real_vertices);
                        H.block<3, 3>(3 * collisions.vertex_ids_inverse(gi), 3 * collisions.vertex_ids_inverse(gj)) += h.block<3, 3>(3 * i, 3 * j);
                    }
                }
            }
        }

        return H;
    }

    std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>
    PointPotential::build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
        const index_t fid,
        size_t& num_collision_pairs) const
    {
        // the fake vertex id
        const index_t vid = V.rows();

        // Use VertexMatrixView to avoid deep-copying the entire vertex matrix
        const Eigen::RowVector3d face_center =
            (V.row(mesh.faces()(fid, 0)) + V.row(mesh.faces()(fid, 1)) + V.row(mesh.faces()(fid, 2))) / 3.;
        VertexMatrixView<3> V_(V, face_center);

        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>> pairs;
        num_collision_pairs = 0;

        const auto& v_set = candidates.fv_set(fid);
        const auto& e_set = candidates.fe_set(fid);
        const auto& f_set = candidates.ff_set(fid);

        for (const auto& other_f : f_set) {
            assert(other_f != fid);
            ++num_collision_pairs;
            if (auto pair = HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
                FaceVertexCandidate(other_f, vid),
                params, mesh, V_)) {
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        for (const auto& other_e : e_set) {
            ++num_collision_pairs;
            if (auto pair = HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                EdgeVertexCandidate(other_e, vid),
                params, mesh, V_)) {
                pair->weight = -1;
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        for (const auto& other_v : v_set) {
            if ((V_(vid) - V_(other_v)).squaredNorm() >= params.dhat * params.dhat) {
                continue;
            }
            std::shared_ptr<HighOrderCollision> pair = std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                vid, other_v, mesh);
            ++num_collision_pairs;
            insert_pair(pairs, std::move(pair));
        }
        std::unique_ptr<HighOrderCollisionDict<PointType::FACE>> collisions = std::make_unique<HighOrderCollisionDict<PointType::FACE>>();
        collisions->initialize(std::vector<index_t>{fid}, std::vector<index_t>{mesh.faces()(fid, 0), mesh.faces()(fid, 1), mesh.faces()(fid, 2)}, pairs);
        return collisions;
    }

    std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>
    PointPotential::build_collisions_at_face_interior_point(
        const Eigen::MatrixXd& V,
        const index_t fid,
        const std::array<double, 3>& lambda,
        size_t& num_collision_pairs) const
    {
        const index_t vid = V.rows();

        const Eigen::RowVector3d q_pos =
            lambda[0] * V.row(mesh.faces()(fid, 0))
            + lambda[1] * V.row(mesh.faces()(fid, 1))
            + lambda[2] * V.row(mesh.faces()(fid, 2));
        VertexMatrixView<3> V_(V, q_pos);

        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>> pairs;
        num_collision_pairs = 0;

        const auto& v_set = candidates.fv_set(fid);
        const auto& e_set = candidates.fe_set(fid);
        const auto& f_set = candidates.ff_set(fid);

        // Detect boundary quadrature points (for rules that include boundary points).
        // lambda[k] == 0.0 means the point lies on the edge opposite vertex k,
        // i.e. edge faces_to_edges(fid, (k+1)%3).
        const bool lam_zero[3] = { lambda[0] == 0.0, lambda[1] == 0.0, lambda[2] == 0.0 };
        const int num_zero = lam_zero[0] + lam_zero[1] + lam_zero[2];

        index_t skip_edge_id = -1; // edge the quad point lies on (1 null lambda)
        index_t corner_vertex = -1; // vertex the quad point coincides with (2 null lambdas)

        if (num_zero == 1) {
            for (int k = 0; k < 3; k++) {
                if (lam_zero[k]) {
                    skip_edge_id = mesh.faces_to_edges()(fid, (k + 1) % 3);
                    break;
                }
            }
        } else if (num_zero == 2) {
            for (int k = 0; k < 3; k++) {
                if (!lam_zero[k]) {
                    corner_vertex = mesh.faces()(fid, k);
                    break;
                }
            }
        }

        const bool src_is_obstacle_f = mesh.is_obstacle_face(fid);
        const bool filter_obstacles_f = src_is_obstacle_f
            && params.integration_type != HighOrderContactParameters::IntegrationType::BRUTE_FORCE;

        for (const auto& other_f : f_set) {
            assert(other_f != fid);
            if (filter_obstacles_f && mesh.is_obstacle_face(other_f)) continue;
            if (skip_edge_id >= 0) {
                bool shares_edge = false;
                for (int j = 0; j < 3; j++)
                    if (mesh.faces_to_edges()(other_f, j) == skip_edge_id) { shares_edge = true; break; }
                if (shares_edge) continue;
            }
            if (corner_vertex >= 0) {
                bool has_vertex = false;
                for (int j = 0; j < 3; j++)
                    if (mesh.faces()(other_f, j) == corner_vertex) { has_vertex = true; break; }
                if (has_vertex) continue;
            }
            ++num_collision_pairs;
            if (auto pair = HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
                FaceVertexCandidate(other_f, vid),
                params, mesh, V_)) {
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        for (const auto& other_e : e_set) {
            if (filter_obstacles_f && mesh.is_obstacle_edge(other_e)) continue;
            if (other_e == skip_edge_id) continue;
            if (corner_vertex >= 0 &&
                (mesh.edges()(other_e, 0) == corner_vertex || mesh.edges()(other_e, 1) == corner_vertex))
                continue;
            ++num_collision_pairs;
            if (auto pair = HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                EdgeVertexCandidate(other_e, vid),
                params, mesh, V_)) {
                pair->weight = -1;
                insert_pair(pairs, std::shared_ptr<HighOrderCollision>(pair));
            }
        }

        for (const auto& other_v : v_set) {
            if (filter_obstacles_f && mesh.is_obstacle_vertex(other_v)) continue;
            if (other_v == corner_vertex) continue;
            if ((V_(vid) - V_(other_v)).squaredNorm() >= params.dhat * params.dhat) {
                continue;
            }
            std::shared_ptr<HighOrderCollision> pair = std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
                vid, other_v, mesh);
            ++num_collision_pairs;
            insert_pair(pairs, std::move(pair));
        }
        std::unique_ptr<HighOrderCollisionDict<PointType::FACE>> collisions = std::make_unique<HighOrderCollisionDict<PointType::FACE>>();
        collisions->initialize(std::vector<index_t>{fid}, std::vector<index_t>{mesh.faces()(fid, 0), mesh.faces()(fid, 1), mesh.faces()(fid, 2)}, pairs);
        return collisions;
    }

    Eigen::VectorXd PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::FACE>& collisions,
        const HighOrderContactParameters& params)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::VectorXd g = cc.weight * cc.gradient(cc.dof(V_extended), params);
            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t global_id = cc.vertex_id(i);
                if (global_id == n_real_vertices) {
                    // distribute grad wrt virtual vertex to real face vertices
                    for (index_t lv = 0; lv < 3; lv++) {
                        grad.segment<3>(collisions.primary_local_ids()[lv] * 3) += g.segment<3>(3 * i) / 3.;
                    }
                }
                else {
                    assert(global_id < n_real_vertices);
                    grad.segment<3>(3 * collisions.vertex_ids_inverse(global_id)) += g.segment<3>(3 * i);
                }
            }
        }

        return grad;
    }

    Eigen::MatrixXd PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::FACE>& collisions,
        const HighOrderContactParameters& params,
        PSDProjectionMethod project_to_psd)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(collisions.vertex_ids().size() * 3, collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::MatrixXd h = cc.hessian(cc.dof(V_extended), params);
            // if (project_to_psd != PSDProjectionMethod::NONE) {
            //     h = ipc::project_to_psd(h, project_to_psd);
            // }
            h *= cc.weight;

            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t gi = cc.vertex_id(i);
                for (index_t j = 0; j < cc.num_vertices(); j++) {
                    const index_t gj = cc.vertex_id(j);
                    if (gi == n_real_vertices && gj == n_real_vertices) {
                        // distribute grad wrt virtual vertex to real face vertices
                        for (index_t li = 0; li < 3; li++) {
                            for (index_t lj = 0; lj < 3; lj++) {
                                H.block<3, 3>(collisions.primary_local_ids()[li] * 3,
                                              collisions.primary_local_ids()[lj] * 3) +=
                                    h.block<3, 3>(3 * i, 3 * j) / 9.;
                            }
                        }
                    }
                    else if (gi == n_real_vertices) {
                        for (index_t li = 0; li < 3; li++) {
                            H.block<3, 3>(collisions.primary_local_ids()[li] * 3,
                                collisions.vertex_ids_inverse(gj) * 3) +=
                                h.block<3, 3>(3 * i, 3 * j) / 3.;
                        }
                    }
                    else if (gj == n_real_vertices) {
                        for (index_t lj = 0; lj < 3; lj++) {
                            H.block<3, 3>(collisions.vertex_ids_inverse(gi) * 3,
                                collisions.primary_local_ids()[lj] * 3) +=
                                h.block<3, 3>(3 * i, 3 * j) / 3.;
                        }
                    }
                    else {
                        assert(gi < n_real_vertices);
                        assert(gj < n_real_vertices);
                        H.block<3, 3>(3 * collisions.vertex_ids_inverse(gi), 3 * collisions.vertex_ids_inverse(gj)) += h.block<3, 3>(3 * i, 3 * j);
                    }
                }
            }
        }

        if (project_to_psd != PSDProjectionMethod::NONE) {
            H = ipc::project_to_psd(H, project_to_psd);
        }
        return H;
    }

    double PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::FACE>& collisions,
        const HighOrderContactParameters& params)
    {
        double potential = 0;
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            potential += cc.weight * cc(cc.dof(V_extended), params);
        }

        return potential;
    }

    // -------------------------------------------------------------------------
    // General face-interior point (arbitrary barycentric coordinates λ)
    // These replace the hard-coded 1/3 (centroid) chain-rule factor with λk.
    // For λ = (1/3, 1/3, 1/3) the results are identical to the face-centre
    // variants above.
    // -------------------------------------------------------------------------

    Eigen::VectorXd PointPotentialHelper::evaluate_potential_gradient_at_face_interior_point_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::FACE>& collisions,
        const HighOrderContactParameters& params,
        const std::array<double, 3>& lambda)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(collisions.vertex_ids().size() * 3);
        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::VectorXd g = cc.weight * cc.gradient(cc.dof(V_extended), params);
            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t global_id = cc.vertex_id(i);
                if (global_id == n_real_vertices) {
                    // Chain rule: dq/dv_k = lambda[k]
                    for (index_t lv = 0; lv < 3; lv++) {
                        grad.segment<3>(collisions.primary_local_ids()[lv] * 3) +=
                            g.segment<3>(3 * i) * lambda[lv];
                    }
                } else {
                    assert(global_id < n_real_vertices);
                    grad.segment<3>(3 * collisions.vertex_ids_inverse(global_id)) +=
                        g.segment<3>(3 * i);
                }
            }
        }
        return grad;
    }

    Eigen::MatrixXd PointPotentialHelper::evaluate_potential_hessian_at_face_interior_point_with_cached_collisions(
        VertexMatrixView<3> V_extended,
        const HighOrderCollisionDict<PointType::FACE>& collisions,
        const HighOrderContactParameters& params,
        const std::array<double, 3>& lambda,
        PSDProjectionMethod project_to_psd)
    {
        const index_t n_real_vertices = V_extended.rows() - 1;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(
            collisions.vertex_ids().size() * 3,
            collisions.vertex_ids().size() * 3);

        for (int ci = 0; ci < collisions.size(); ci++) {
            const auto& cc = collisions[ci];
            Eigen::MatrixXd h = cc.hessian(cc.dof(V_extended), params);
            h *= cc.weight;

            for (index_t i = 0; i < cc.num_vertices(); i++) {
                const index_t gi = cc.vertex_id(i);
                for (index_t j = 0; j < cc.num_vertices(); j++) {
                    const index_t gj = cc.vertex_id(j);
                    if (gi == n_real_vertices && gj == n_real_vertices) {
                        // Both indices refer to the virtual vertex q.
                        // d²P / (dv_li dv_lj) = λ_li · λ_lj · d²P/dq²
                        for (index_t li = 0; li < 3; li++) {
                            for (index_t lj = 0; lj < 3; lj++) {
                                H.block<3, 3>(
                                    collisions.primary_local_ids()[li] * 3,
                                    collisions.primary_local_ids()[lj] * 3) +=
                                    h.block<3, 3>(3 * i, 3 * j)
                                    * lambda[li] * lambda[lj];
                            }
                        }
                    } else if (gi == n_real_vertices) {
                        // d²P / (dv_li d(other)) = λ_li · d²P / (dq d(other))
                        for (index_t li = 0; li < 3; li++) {
                            H.block<3, 3>(
                                collisions.primary_local_ids()[li] * 3,
                                collisions.vertex_ids_inverse(gj) * 3) +=
                                h.block<3, 3>(3 * i, 3 * j) * lambda[li];
                        }
                    } else if (gj == n_real_vertices) {
                        // Symmetric to the gi == n_real_vertices case.
                        for (index_t lj = 0; lj < 3; lj++) {
                            H.block<3, 3>(
                                collisions.vertex_ids_inverse(gi) * 3,
                                collisions.primary_local_ids()[lj] * 3) +=
                                h.block<3, 3>(3 * i, 3 * j) * lambda[lj];
                        }
                    } else {
                        assert(gi < n_real_vertices);
                        assert(gj < n_real_vertices);
                        H.block<3, 3>(
                            3 * collisions.vertex_ids_inverse(gi),
                            3 * collisions.vertex_ids_inverse(gj)) +=
                            h.block<3, 3>(3 * i, 3 * j);
                    }
                }
            }
        }

        if (project_to_psd != PSDProjectionMethod::NONE) {
            H = ipc::project_to_psd(H, project_to_psd);
        }
        return H;
    }
}
