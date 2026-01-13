#include "quadrature_potential.hpp"

#include <absl/strings/str_format.h>

#include "ipc/candidates/candidates.hpp"
#include "ipc/distance/edge_edge.hpp"
#include "ipc/high_order_contact/high_order_collisions_builder.hpp"
#include "ipc/utils/area_gradient.hpp"
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
            if (auto iter = collisions.find(collision->get_hash()); iter != collisions.end()) {
                iter->second->weight += collision->weight;
                if (iter->second->weight == 0) {
                    collisions.erase(iter);
                }
            } else {
                collisions[collision->get_hash()] = collision;
            }
        }
    }

    QuadraturePotential::QuadraturePotential(
        const CollisionMesh& _mesh,
        const Eigen::MatrixXd& V,
        const double _dhat) : mesh(_mesh), dhat(_dhat)
    {
        HighOrderContactParameters params(dhat, 0., 2, 0);
        collisions.build(mesh, V, params, false, make_default_broad_phase());

        point_potential = std::make_unique<PointPotential>(mesh, collisions, params);
    }

    double PointPotential::evaluate_potential_at_vertex(
        const Eigen::MatrixXd& V,
        const index_t vid) const
    {
        unordered_map<std::pair<index_t, index_t>, std::shared_ptr<HighOrderCollision>> pairs;

        const auto& v_set = collisions.m_candidates.vv_set(vid);
        const auto& e_set = collisions.m_candidates.ve_set(vid);
        const auto& f_set = collisions.m_candidates.vf_set(vid);

        for (const auto& other_f : f_set) {
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
                FaceVertexCandidate(other_f, vid),
                params, mesh, V); pair->is_active()) {
                    insert_pair(pairs, pair);
                }
        }

        for (const auto& other_e : e_set) {
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                EdgeVertexCandidate(other_e, vid),
                params, mesh, V); pair->is_active()) {
                    pair->weight = -1;
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

        double potential = 0;
        for (const auto& cc : pairs) {
            potential += cc.second->weight * (*(cc.second))(cc.second->dof(V), params);
        }

        return potential;
    }

    double PointPotential::evaluate_potential_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        unordered_map<std::array<index_t, 3>, std::shared_ptr<TriplePairCollision>> pairs;

        const auto& v_set = collisions.m_candidates.ev_set(e0);
        const auto& e_set = collisions.m_candidates.ee_set(e0);
        const auto& f_set = collisions.m_candidates.ef_set(e0);

        // Compute closest point
        const index_t e00 = mesh.edges()(e0, 0);
        const index_t e01 = mesh.edges()(e0, 1);
        const index_t e10 = mesh.edges()(e1, 0);
        const index_t e11 = mesh.edges()(e1, 1);
        const EdgeEdgeDistanceType dtype = edge_edge_distance_type(
            V.row(e00), V.row(e01),
            V.row(e10), V.row(e11)
        );
        if (dtype != EdgeEdgeDistanceType::EA_EB) {
            log_and_throw_error("Can only handle edge-edge distance type!");
        }

        if (edge_edge_distance(V.row(e00), V.row(e01),
                               V.row(e10), V.row(e11), dtype) >= params.dhat * params.dhat)
            return 0.;

        const Eigen::Vector2d closest_uvs = line_line_closest_point_pairs_uv<double>(
            V.row(e00), V.row(e01),
            V.row(e10), V.row(e11));

        if (!std::isfinite(closest_uvs.norm())) {
            log_and_throw_error("Potentially parallel edges!");
        }

        const Eigen::Vector3d q = closest_uvs(0) * (V.row(e01) - V.row(e00)) + V.row(e00);

        for (const auto& other_v : v_set) {
            std::shared_ptr<TriplePairCollision> pair = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        e0, e1, other_v, mesh, params, params.dhat, V);

            if (pair->is_active()) {
                insert_pair(pairs, pair);
            }
        }

        for (const auto& other_e : e_set) {
            auto pair = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                        e0, e1, other_e, mesh, params, params.dhat, V);

            if (!pair->is_active()) {
                continue;
            }

            switch (pair->distance_type_2()) {
            case PointEdgeDistanceType::P_E0:
                {
                    std::shared_ptr<TriplePairCollision> pair2 = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                            e0, e1, mesh.edges()(other_e, 0), mesh, params, params.dhat, V);
                    pair2->weight = -1;
                    insert_pair(pairs, pair2);
                    break;
                }
            case PointEdgeDistanceType::P_E1:
                {
                    std::shared_ptr<TriplePairCollision> pair2 = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                            e0, e1, mesh.edges()(other_e, 1), mesh, params, params.dhat, V);
                    pair2->weight = -1;
                    insert_pair(pairs, pair2);
                    break;
                }
            case PointEdgeDistanceType::P_E:
                {
                    pair->weight = -1;
                    insert_pair(pairs, std::shared_ptr<TriplePairCollision>(pair));
                    break;
                }
            default:
                assert(false);
                break;
            }
        }

        for (const auto& other_f : f_set) {
            auto pair = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Face3P1>>(
                        e0, e1, other_f, mesh, params, params.dhat, V);

            if (!pair->is_active()) {
                continue;
            }

            switch (pair->distance_type_2()) {
            case PointTriangleDistanceType::P_T0:
                {
                    insert_pair(pairs, std::shared_ptr<TriplePairCollision>(std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        e0, e1, mesh.faces()(other_f, 0), mesh, params, params.dhat, V)));
                    break;
                }
            case PointTriangleDistanceType::P_T1:
                {
                    insert_pair(pairs, std::shared_ptr<TriplePairCollision>(std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        e0, e1, mesh.faces()(other_f, 1), mesh, params, params.dhat, V)));
                    break;
                }
            case PointTriangleDistanceType::P_T2:
                {
                    insert_pair(pairs, std::shared_ptr<TriplePairCollision>(std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        e0, e1, mesh.faces()(other_f, 2), mesh, params, params.dhat, V)));
                    break;
                }
            case PointTriangleDistanceType::P_E0:
                {
                    insert_pair(pairs,
                        std::shared_ptr<TriplePairCollision>(std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                            e0, e1, mesh.faces_to_edges()(other_f, 0), mesh, params, params.dhat, V)));
                    break;
                }
            case PointTriangleDistanceType::P_E1:
                {
                    insert_pair(pairs,
                        std::shared_ptr<TriplePairCollision>(std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                            e0, e1, mesh.faces_to_edges()(other_f, 1), mesh, params, params.dhat, V)));
                    break;
                }
            case PointTriangleDistanceType::P_E2:
                {
                    insert_pair(pairs,
                        std::shared_ptr<TriplePairCollision>(std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                            e0, e1, mesh.faces_to_edges()(other_f, 2), mesh, params, params.dhat, V)));
                    break;
                }
            case PointTriangleDistanceType::P_T:
                {
                    insert_pair(pairs, std::shared_ptr<TriplePairCollision>(pair));
                    break;
                }
            default:
                assert(false);
                break;
            }
        }

        double potential = 0;
        for (const auto& pair : pairs) {
            const auto& cc = pair.second;
            double term = (*cc)(cc->dof(V), params);
            assert(std::isfinite(term));
            potential += cc->weight * term;
        }

        return potential;
    }

    double PointPotential::evaluate_potential_at_face_center(
        const Eigen::MatrixXd& V,
        const index_t fid) const
    {
        const index_t t0 = mesh.faces()(fid, 0);
        const index_t t1 = mesh.faces()(fid, 1);
        const index_t t2 = mesh.faces()(fid, 2);

        // Create a virtual vertex as the face center

        Eigen::MatrixXd V_(V.rows() + 1, 3);
        V_.topRows(V.rows()) = V;
        V_.row(V.rows()) = (V.row(t0) + V.row(t1) + V.row(t2)) / 3.0;
        const index_t vid = V.rows();

        unordered_map<std::pair<index_t, index_t>, std::shared_ptr<HighOrderCollision>> pairs;

        const auto& v_set = collisions.m_candidates.fv_set(fid);
        const auto& e_set = collisions.m_candidates.fe_set(fid);
        const auto& f_set = collisions.m_candidates.ff_set(fid);

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

        double potential = 0;
        for (const auto& pair : pairs) {
            const auto& cc = pair.second;
            potential += cc->weight * (*cc)(cc->dof(V_), params);
        }

        return potential;
    }

    double QuadraturePotential::evaluate_per_face(
        const Eigen::MatrixXd& V,
        const index_t face_id)
    {
        const Eigen::Vector3d f0 = V.row(mesh.faces()(face_id, 0));
        const Eigen::Vector3d f1 = V.row(mesh.faces()(face_id, 1));
        const Eigen::Vector3d f2 = V.row(mesh.faces()(face_id, 2));
        const double area = 0.5 * (f1 - f0).cross(f2 - f0).norm();

        double total = 0.;
        for (index_t le = 0; le < 3; le++) {
            const index_t edge_id = mesh.faces_to_edges()(face_id, le);
            const index_t ea = mesh.edges()(edge_id, 0);
            const index_t eb = mesh.edges()(edge_id, 1);

            const std::set<index_t> close_edges = collisions.m_candidates.ee_set(edge_id);

            std::vector<EdgePairClosestPoint> points = {
                EdgePairClosestPoint(0.), EdgePairClosestPoint(1.)
            };
            for (index_t other_edge_id : close_edges) {
                const index_t ec = mesh.edges()(other_edge_id, 0);
                const index_t ed = mesh.edges()(other_edge_id, 1);

                // Skip adjacent edges
                if (ea == ec || ea == ed || eb == ec || eb == ed) {
                    continue;
                }

                const auto dtype = edge_edge_distance_type(
                    V.row(ea), V.row(eb),
                    V.row(ec), V.row(ed));

                if (dtype != EdgeEdgeDistanceType::EA_EB) {
                    continue;
                }

                if (is_parallel_edge_edge(
                    V.row(ea), V.row(eb),
                    V.row(ec), V.row(ed))) {
                    continue;
                }

                const double dist = sqrt(edge_edge_distance(
                    V.row(ea), V.row(eb),
                    V.row(ec), V.row(ed)));

                if (dist >= dhat) {
                    continue;
                }

                Eigen::Vector<double, 2> closest_points_uv = line_line_closest_point_pairs_uv<double>(
                    V.row(ea), V.row(eb),
                    V.row(ec), V.row(ed));

                assert(closest_points_uv(0) > 0 && closest_points_uv(0) < 1);

                std::array<HeavisideType, 4> mtypes{{HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT}};
                double mollifier = Math<double>::cubic_spline(dist / dhat) * 1.5;
                mollifier *= edge_edge_mollifier<double>(
                        V.row(ea), V.row(eb),
                        V.row(ec), V.row(ed),
                        mtypes, dist * dist);

                if (mollifier == 0) {
                    continue;
                }

                points.push_back(EdgePairClosestPoint(closest_points_uv(0), other_edge_id, mollifier));
            }

            // sort uv from small to large
            std::sort(points.begin(), points.end(),
                      [](const EdgePairClosestPoint& a, const EdgePairClosestPoint& b) {
                          return a.uv0 < b.uv0;
                      });

            assert(points.front().uv0 == 0.);
            assert(points.back().uv0 == 1.);

            assert(points[0].beta == 0.);
            assert(points.back().beta == 1.);

            double norm_fac = 0.;
            for (index_t i = 0; i < points.size() - 1; i++) {
                const auto& pts_a = points[i];
                const auto& pts_b = points[i + 1];
                norm_fac += (pts_b.beta - pts_a.beta) * (pts_b.uv0 * pts_b.mollifier + pts_a.uv0 * pts_a.mollifier);
            }

            for (auto& pts : points) {
                pts.beta /= norm_fac;
            }

            const double P_q_center = point_potential->evaluate_potential_at_face_center(V, face_id);

            std::vector<double> P_q_i(points.size(), 0.0);
            {
                assert(points[0].uv0 == 0.);
                P_q_i[0] = point_potential->evaluate_potential_at_vertex(
                    V, mesh.edges()(edge_id, 0));
            }
            {
                assert(points[P_q_i.size() - 1].uv0 == 1.);
                P_q_i.back() = point_potential->evaluate_potential_at_vertex(
                    V, mesh.edges()(edge_id, 1));
            }
            for (index_t i = 1; i < P_q_i.size() - 1; i++) {
                    assert(points[i].uv0 < 1.);
                    assert(points[i].uv0 > 0.);
                    assert(points[i].e1 >= 0);
                    P_q_i[i] = point_potential->evaluate_potential_at_edge_edge_closest_point(
                        V, edge_id, points[i].e1);
            }

            double cur_val = 0.;
            for (index_t i = 1; i < P_q_i.size() - 1; i++) {
                cur_val += P_q_i[i] * (points[i + 1].beta - points[i - 1].beta) * points[i].mollifier;
            }

            // two vertices do not need mollifier
            cur_val += P_q_i[0] * (points[1].beta - points[0].beta) + P_q_i.back() * (points.back().beta - points[points.size() - 2].beta);

            cur_val += P_q_center;
            total += cur_val * area / 9.;
        }

        return total;
    }

    Eigen::VectorXd QuadraturePotential::evaluate_per_face_gradient(
        const Eigen::MatrixXd& V,
        const index_t face_id)
    {
        log_and_throw_error("Not implemented");
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(3 * mesh.num_vertices());
        return grad;
    }
}
