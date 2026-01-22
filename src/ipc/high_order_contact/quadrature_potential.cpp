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
            if (auto iter = collisions.find(collision->get_typed_hash()); iter != collisions.end()) {
                iter->second->weight += collision->weight;
                if (iter->second->weight == 0) {
                    collisions.erase(iter);
                }
            } else {
                collisions[collision->get_typed_hash()] = collision;
            }
        }
    }

    QuadraturePotential::QuadraturePotential(
        const CollisionMesh& _mesh,
        const Eigen::MatrixXd& V,
        const double _dhat) : mesh(_mesh), dhat(_dhat)
    {
        HighOrderContactParameters params(dhat, 0., 2, 0);

        double inflation_radius = dhat / 2;
        candidates.build(mesh, V, inflation_radius, make_default_broad_phase(), true);
        candidates.convert_candidates_to_sets();

        point_potential = std::make_unique<PointPotential>(mesh, candidates, params);
    }

    unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
        PointPotential::build_collisions_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const
    {
        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>> pairs;

        const auto& v_set = candidates.vv_set(vid);
        const auto& e_set = candidates.ve_set(vid);
        const auto& f_set = candidates.vf_set(vid);

        for (const auto& other_f : f_set) {
            if (std::shared_ptr<HighOrderCollision> pair = HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
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
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params)
{
    double potential = 0;
    for (const auto& pair : collisions) {
        const auto& cc = pair.second;
        potential += cc->weight * (*cc)(cc->dof(V), params);
    }

    return potential;
}

    double PointPotential::evaluate_potential_at_vertex(
        const Eigen::MatrixXd& V,
        const index_t vid) const
    {
        const auto pairs = build_collisions_at_vertex(V, vid);

        return PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
            V, pairs, params);
    }

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
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
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
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

    Eigen::SparseMatrix<double> PointPotential::evaluate_potential_gradient_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const
    {
        const auto pairs = build_collisions_at_vertex(V, vid);

        return PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(V, pairs, params);
    }

    Eigen::SparseMatrix<double> PointPotential::evaluate_potential_hessian_at_vertex(
            const Eigen::MatrixXd& V,
            const index_t vid) const
    {
        const auto pairs = build_collisions_at_vertex(V, vid);

        return PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(V, pairs, params);
    }

    unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>
    PointPotential::build_collisions_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>> pairs;

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

        if (dtype != EdgeEdgeDistanceType::EA_EB && dtype != EdgeEdgeDistanceType::EA_EB0 && dtype != EdgeEdgeDistanceType::EA_EB1) {
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
            closest_uv = t.dot(d) / t.squaredNorm();
        }
        else if (dtype == EdgeEdgeDistanceType::EA_EB1) {
            Eigen::RowVector3d t = V.row(e01) - V.row(e00);
            const double a = t.squaredNorm();
            const double b = t.dot(V.row(e11) - V.row(e10));
            const double d = t.dot(V.row(e00) - V.row(e10));
            closest_uv = (-d + b) / a;
        }
        else
            log_and_throw_error("Invalid dtype!");

        if (!std::isfinite(closest_uv)) {
            log_and_throw_error("Potentially parallel edges!");
        }

        bool for_debug = false;
        if (closest_uv < 1e-15)
            for_debug = true;

        assert(closest_uv > 0);

        const Eigen::Vector3d q = closest_uv * (V.row(e01) - V.row(e00)) + V.row(e00);

        // for (const auto& other_v : v_set) {
        for (index_t other_v = 0; other_v < mesh.num_vertices(); ++other_v) {
            std::shared_ptr<TriplePairCollision> pair = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        e0, e1, other_v, mesh, params, params.dhat, V);

            if (pair->is_active()) {
                insert_pair(pairs, pair);
            }
        }

        // for (const auto& other_e : e_set) {
        for (index_t other_e = 0; other_e < mesh.num_edges(); ++other_e) {
            if (other_e == e0)
                continue;

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

        // for (const auto& other_f : f_set) {
        for (index_t other_f = 0; other_f < mesh.num_faces(); ++other_f) {
            if (mesh.edges_to_faces()(e0, 0) == other_f || mesh.edges_to_faces()(e0, 1) == other_f)
                continue;

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

        if (for_debug) {
            logger().debug("edge-edge collision P(q) terms for q {} size is {} between edge {} {} and {} {}", closest_uv, pairs.size(), e00, e01, e10, e11);
            for (const auto& pair : pairs) {
                const auto& cc = *(pair.second);
                logger().debug("name {}, id {}, w {}, distance {}", cc.name(), cc[1], cc.weight, sqrt(cc.compute_distance(V)));
            }
        }

        return pairs;
    }

    unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
    PointPotential::build_collisions_at_edge_edge_closest_point_advanced(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>> pairs;

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
        if (dtype != EdgeEdgeDistanceType::EA_EB && dtype != EdgeEdgeDistanceType::EA_EB0 && dtype != EdgeEdgeDistanceType::EA_EB1) {
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

            auto dtype2 = point_edge_distance_type(V_.row(vid), V_.row(mesh.edges()(other_e, 0)), V_.row(mesh.edges()(other_e, 1)));

            switch (dtype2) {
            case PointEdgeDistanceType::P_E0:
                {
                    std::shared_ptr<HighOrderCollision> pair2 = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                            vid, mesh.edges()(other_e, 0), mesh, params, params.dhat, V_);
                    pair2->weight = -1;
                    insert_pair(pairs, pair2);
                    break;
                }
            case PointEdgeDistanceType::P_E1:
                {
                    std::shared_ptr<HighOrderCollision> pair2 = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
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

            auto dtype2 = point_triangle_distance_type(V_.row(vid), V_.row(mesh.faces()(other_f, 0)), V_.row(mesh.faces()(other_f, 1)), V_.row(mesh.faces()(other_f, 2)));

            switch (dtype2) {
            case PointTriangleDistanceType::P_T0:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                        vid, mesh.faces()(other_f, 0), mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_T1:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                        vid, mesh.faces()(other_f, 1), mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_T2:
                {
                    insert_pair(pairs, std::shared_ptr<HighOrderCollision>(std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                        vid, mesh.faces()(other_f, 2), mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_E0:
                {
                    insert_pair(pairs,
                        std::shared_ptr<HighOrderCollision>(std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                            mesh.faces_to_edges()(other_f, 0), vid, mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_E1:
                {
                    insert_pair(pairs,
                        std::shared_ptr<HighOrderCollision>(std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                            mesh.faces_to_edges()(other_f, 1), vid, mesh, params, params.dhat, V_)));
                    break;
                }
            case PointTriangleDistanceType::P_E2:
                {
                    insert_pair(pairs,
                        std::shared_ptr<HighOrderCollision>(std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
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
        const Eigen::MatrixXd& V_extended,
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
        const HighOrderContactParameters& params)
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

    double PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
        const Eigen::MatrixXd& V,
        const unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>& collisions,
        const HighOrderContactParameters& params)
    {
        double potential = 0;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            double term = (*cc)(cc->dof(V), params);
            potential += cc->weight * term;
        }

        return potential;
    }

    double PointPotential::evaluate_potential_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        const auto pairs = build_collisions_at_edge_edge_closest_point(V, e0, e1);

        return PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                V, pairs, params);
    }

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>& collisions,
            const HighOrderContactParameters& params)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::VectorXd g = cc->weight * cc->gradient(cc->dof(V), params);
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

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V_extended,
            const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
            const HighOrderContactParameters& params,
            Eigen::ConstRef<Eigen::Vector4i> vids,
            const Eigen::Vector3<ADGrad<12>>& q)
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

     Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
         const Eigen::MatrixXd& V_extended,
         const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
         const HighOrderContactParameters& params,
         Eigen::ConstRef<Eigen::Vector4i> vids,
         const Eigen::Vector3<ADHessian<12>>& q)
     {
         const index_t n_real_vertices = V_extended.rows() - 1;
         std::vector<Eigen::Triplet<double>> triplets;
         for (const auto& pair : collisions) {
             const auto& cc = pair.second;
             Eigen::MatrixXd h = cc->weight * cc->hessian(cc->dof(V_extended), params);
             Eigen::MatrixXd g = cc->weight * cc->gradient(cc->dof(V_extended), params);

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
                                         triplets.emplace_back(vids[li] * 3 + di, vids[lj] * 3 + dj, local_hess(3 * li + di, 3 * lj + dj));
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
                         for (index_t dj = 0; dj < 3; dj++) {
                             for (index_t di = 0; di < 3; di++) {
                                 for (index_t lj = 0; lj < 4; lj++) {
                                     triplets.emplace_back(vids[lj] * 3 + dj, gi * 3 + di, h(3 * i + dj, 3 * j + di));
                                     triplets.emplace_back(gi * 3 + di, vids[lj] * 3 + dj, h(3 * i + dj, 3 * j + di));
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

    Eigen::SparseMatrix<double> PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
            const Eigen::MatrixXd& V,
            const unordered_map<std::array<index_t, 4>, std::shared_ptr<TriplePairCollision>>& collisions,
            const HighOrderContactParameters& params)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            Eigen::MatrixXd h = cc->weight * cc->hessian(cc->dof(V), params);
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

    Eigen::SparseMatrix<double> PointPotential::evaluate_potential_gradient_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        const auto pairs = build_collisions_at_edge_edge_closest_point(V, e0, e1);

        return PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(V, pairs, params);
    }

    Eigen::SparseMatrix<double> PointPotential::evaluate_potential_hessian_at_edge_edge_closest_point(
        const Eigen::MatrixXd& V,
        const index_t e0,
        const index_t e1) const
    {
        const auto pairs = build_collisions_at_edge_edge_closest_point(V, e0, e1);

        return PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(V, pairs, params);
    }

    unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
    PointPotential::build_collisions_at_face_center(
        const Eigen::MatrixXd& V,
        const index_t fid) const
    {
        // the fake vertex id
        const index_t vid = V.rows();

        Eigen::MatrixXd V_(V.rows() + 1, 3);
        V_.topRows(V.rows()) = V;
        V_.row(vid) = (V.row(mesh.faces()(fid, 0)) + V.row(mesh.faces()(fid, 1)) + V.row(mesh.faces()(fid, 2))) / 3.;

        unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>> pairs;

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
        const Eigen::MatrixXd& V_extended,
        Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
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
        const Eigen::MatrixXd& V_extended,
        Eigen::ConstRef<Eigen::Vector3<index_t>> vids,
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
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
                                        triplets.emplace_back(vids[li] * 3 + di, vids[lj] * 3 + dj, h(3 * i + di, 3 * j + dj) / 9.);
                                    }
                                }
                            }
                        }
                    }
                    else if (gi == n_real_vertices) {
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t li = 0; li < 3; li++) {
                                    triplets.emplace_back(vids[li] * 3 + di, gj * 3 + dj, h(3 * i + di, 3 * j + dj) / 3.);
                                }
                            }
                        }
                    }
                    else if (gj == n_real_vertices) {
                        for (index_t di = 0; di < 3; di++) {
                            for (index_t dj = 0; dj < 3; dj++) {
                                for (index_t lj = 0; lj < 3; lj++) {
                                    triplets.emplace_back(gi * 3 + di, vids[lj] * 3 + dj, h(3 * i + di, 3 * j + dj) / 3.);
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

    Eigen::SparseMatrix<double> PointPotential::evaluate_potential_gradient_at_face_center(
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

        const auto pairs = build_collisions_at_face_center(V, fid);

        return PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(V_, mesh.faces().row(fid), pairs, params);
    }

    Eigen::SparseMatrix<double> PointPotential::evaluate_potential_hessian_at_face_center(
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

        const auto pairs = build_collisions_at_face_center(V, fid);

        return PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(
            V_, mesh.faces().row(fid), pairs, params);
    }

    double PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
        const Eigen::MatrixXd& V_extended,
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& collisions,
        const HighOrderContactParameters& params)
    {
        double potential = 0;
        for (const auto& pair : collisions) {
            const auto& cc = pair.second;
            potential += cc->weight * (*cc)(cc->dof(V_extended), params);
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

        const auto pairs = build_collisions_at_face_center(V, fid);

        return PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(V_, pairs, params);
    }

    double QuadraturePotential::evaluate_per_face(
        const Eigen::MatrixXd& V,
        const index_t face_id) const
    {
        const double area = mesh.face_areas()(face_id);

        double total = 0.;
        for (index_t le = 0; le < 3; le++) {
            const index_t edge_id = mesh.faces_to_edges()(face_id, le);
            const index_t ea = mesh.edges()(edge_id, 0);
            const index_t eb = mesh.edges()(edge_id, 1);

            const std::set<index_t> close_edges = candidates.ee_set(edge_id);

            std::vector<EdgePairClosestPoint<double>> points;
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

                if (dtype != EdgeEdgeDistanceType::EA_EB && dtype != EdgeEdgeDistanceType::EA_EB0 && dtype != EdgeEdgeDistanceType::EA_EB1) {
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

                double closest_uv = 0;
                if (dtype == EdgeEdgeDistanceType::EA_EB) {
                    closest_uv = line_line_closest_point_pairs_uv<double>(
                        V.row(ea), V.row(eb),
                        V.row(ec), V.row(ed))(0);
                }
                else if (dtype == EdgeEdgeDistanceType::EA_EB0) {
                    Eigen::RowVector3d p = V.row(ec);
                    Eigen::RowVector3d d = p - V.row(ea);
                    Eigen::RowVector3d t = V.row(eb) - V.row(ea);
                    closest_uv = d.dot(t) / t.squaredNorm();
                }
                else if (dtype == EdgeEdgeDistanceType::EA_EB1) {
                    Eigen::RowVector3d p = V.row(ed);
                    Eigen::RowVector3d d = p - V.row(ea);
                    Eigen::RowVector3d t = V.row(eb) - V.row(ea);
                    closest_uv = d.dot(t) / t.squaredNorm();
                }
                else
                    log_and_throw_error("Invalid dtype!");

                double mollifier = Math<double>::cubic_spline(dist / dhat) * 1.5;
                mollifier *= half_edge_edge_mollifier<double>(
                        V.row(ea), V.row(eb),
                        V.row(ec), V.row(ed),
                        dist * dist);

                if (mollifier == 0) {
                    continue;
                }

                points.push_back(EdgePairClosestPoint(closest_uv, other_edge_id, mollifier));
            }

            double cur_val = 0.;
            if (points.size() > 0) {
                std::vector<double> P_q_i(points.size(), 0.0);
                for (index_t i = 0; i < P_q_i.size(); i++) {
                    assert(points[i].uv0 < 1.);
                    assert(points[i].uv0 > 0.);
                    assert(points[i].e1 >= 0);
                    P_q_i[i] = point_potential->evaluate_potential_at_edge_edge_closest_point(
                        V, edge_id, points[i].e1);
                }

                for (index_t i = 0; i < P_q_i.size(); i++) {
                    cur_val += P_q_i[i] * points[i].mollifier;
                }
            }

            // two vertices and face center do not need mollifier
            cur_val += point_potential->evaluate_potential_at_vertex(
                    V, mesh.edges()(edge_id, 0)) +
                point_potential->evaluate_potential_at_vertex(
                    V, mesh.edges()(edge_id, 1)) +
                point_potential->evaluate_potential_at_face_center(V, face_id);

            total += cur_val * area / 9.;
        }

        return total;
    }

    Eigen::SparseMatrix<double> QuadraturePotential::evaluate_per_face_gradient(
        const Eigen::MatrixXd& V,
        const index_t face_id) const
    {
        Eigen::SparseMatrix<double> grad(3 * mesh.num_vertices(), 1);

        const double area = mesh.face_areas()(face_id);

        for (index_t le = 0; le < 3; le++) {
            const index_t edge_id = mesh.faces_to_edges()(face_id, le);
            const index_t ea = mesh.edges()(edge_id, 0);
            const index_t eb = mesh.edges()(edge_id, 1);

            const std::set<index_t> close_edges = candidates.ee_set(edge_id);

            using T = ADGrad<12>;

            std::vector<EdgePairClosestPoint<T>> points;
            for (index_t other_edge_id : close_edges) {
                const index_t ec = mesh.edges()(other_edge_id, 0);
                const index_t ed = mesh.edges()(other_edge_id, 1);

                Eigen::Vector<double, 12> positions;
                positions << V.row(ea).transpose(), V.row(eb).transpose(), V.row(ec).transpose(), V.row(ed).transpose();

                Eigen::Matrix<T, 4, 3> positionsT = slice_positions<T, 4, 3>(positions);

                // Skip adjacent edges
                if (ea == ec || ea == ed || eb == ec || eb == ed) {
                    continue;
                }

                const auto dtype = edge_edge_distance_type(
                    V.row(ea), V.row(eb),
                    V.row(ec), V.row(ed));

                if (dtype != EdgeEdgeDistanceType::EA_EB && dtype != EdgeEdgeDistanceType::EA_EB0 && dtype != EdgeEdgeDistanceType::EA_EB1) {
                    continue;
                }

                if (is_parallel_edge_edge(
                    V.row(ea), V.row(eb),
                    V.row(ec), V.row(ed))) {
                    continue;
                }

                const T dist = sqrt(edge_edge_sqr_distance<T>(
                    positionsT.row(0), positionsT.row(1),
                    positionsT.row(2), positionsT.row(3), dtype));

                if (dist >= dhat) {
                    continue;
                }

                T closest_uv = 0;
                if (dtype == EdgeEdgeDistanceType::EA_EB) {
                    closest_uv = line_line_closest_point_pairs_uv<T>(
                    positionsT.row(0), positionsT.row(1),
                    positionsT.row(2), positionsT.row(3))(0);
                }
                else if (dtype == EdgeEdgeDistanceType::EA_EB0) {
                    Eigen::RowVector3<T> p = positionsT.row(2);
                    Eigen::RowVector3<T> d = p - positionsT.row(0);
                    Eigen::RowVector3<T> t = positionsT.row(1) - positionsT.row(0);
                    closest_uv = d.dot(t) / t.squaredNorm();
                }
                else if (dtype == EdgeEdgeDistanceType::EA_EB1) {
                    Eigen::RowVector3<T> p = positionsT.row(2);
                    Eigen::RowVector3<T> d = p - positionsT.row(0);
                    Eigen::RowVector3<T> t = positionsT.row(1) - positionsT.row(0);
                    closest_uv = d.dot(t) / t.squaredNorm();
                }
                else
                    log_and_throw_error("Invalid dtype!");

                T mollifier = Math<T>::cubic_spline(dist / dhat) * 1.5;
                mollifier *= half_edge_edge_mollifier<T>(
                positionsT.row(0), positionsT.row(1),
                positionsT.row(2), positionsT.row(3),
                        dist * dist);

                if (mollifier == 0.) {
                    continue;
                }

                points.push_back(EdgePairClosestPoint<T>(closest_uv, other_edge_id, mollifier));
            }

            Eigen::SparseMatrix<double> cur_grad(V.size(), 1);
            if (points.size() > 0) {
                std::vector<double> P_q_i_values(points.size());
                std::vector<Eigen::SparseMatrix<double>> P_q_i_grad(points.size());

                for (index_t i = 0; i < P_q_i_grad.size(); i++) {
                        assert(points[i].uv0 < 1.);
                        assert(points[i].uv0 > 0.);
                        assert(points[i].e1 >= 0);
                        P_q_i_grad[i] = point_potential->evaluate_potential_gradient_at_edge_edge_closest_point(
                            V, edge_id, points[i].e1);
                        P_q_i_values[i] = point_potential->evaluate_potential_at_edge_edge_closest_point(
                            V, edge_id, points[i].e1);
                }

                for (index_t i = 0; i < P_q_i_grad.size(); i++) {
                    cur_grad += P_q_i_grad[i] * points[i].mollifier.val;

                    Vector12d cur_grad_2 = P_q_i_values[i] * points[i].mollifier.grad;
                    for (int d = 0; d < 3; d++) {
                        cur_grad.coeffRef(ea * 3 + d, 0) += cur_grad_2(d + 0);
                        cur_grad.coeffRef(eb * 3 + d, 0) += cur_grad_2(d + 3);

                        cur_grad.coeffRef(mesh.edges()(points[i].e1, 0) * 3 + d, 0) += cur_grad_2(d + 6);
                        cur_grad.coeffRef(mesh.edges()(points[i].e1, 1) * 3 + d, 0) += cur_grad_2(d + 9);
                    }
                }
            }

            // two vertices do not need mollifier
            cur_grad += point_potential->evaluate_potential_gradient_at_vertex(
                        V, mesh.edges()(edge_id, 0)) +
                    point_potential->evaluate_potential_gradient_at_vertex(
                        V, mesh.edges()(edge_id, 1)) +
                    point_potential->evaluate_potential_gradient_at_face_center(V, face_id);

            grad += cur_grad * (area / 9.);
        }

        return grad;
    }
}
