#include "high_order_collisions_builder.hpp"
#include <ipc/high_order_contact/quadrature_potential.hpp>
#include <ipc/high_order_contact/collisions/triangular_quadrature.hpp>

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

#include <tbb/enumerable_thread_specific.h>

namespace ipc {

using IntegrationType = HighOrderContactParameters::IntegrationType;

namespace {
    template <typename TCollision, typename THash>
    void add_collision(
        const std::shared_ptr<TCollision> pair,
        unordered_map<THash, index_t>& cc_to_id,
        std::vector<std::shared_ptr<TCollision>>& collisions)
    {
        assert(pair != nullptr);
        // filters dupes
        auto found_item = cc_to_id.find(pair->get_hash());
        if (found_item == cc_to_id.end()) {
            // New collision, so add it to the end of collisions
            cc_to_id.emplace(pair->get_hash(), collisions.size());
            collisions.push_back(pair);
        }
        else {
            collisions[found_item->second]->weight += pair->weight;
        }
    }
} // namespace

void HighOrderCollisionsBuilder<2>::add_edge_vertex_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeVertexCandidate>& candidates,
    const HighOrderContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const size_t start_i,
    const size_t end_i)
{
    if (params.quad_order == 0) throw std::logic_error("Vertex integration temporarily removed");
    const double dhat = params.dhat;
    const double dhat2 = dhat * dhat;

    // go over EV candidates and add those with nonzero potential.
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, vi] = candidates[i];
        const auto &v = vertices.row(vi);
        const auto &ei0 = vertices.row(mesh.edges()(ei, 0));
        const auto &ei1 = vertices.row(mesh.edges()(ei, 1));
        const double d2 = point_edge_distance(v, ei0, ei1, point_edge_distance_type(v, ei0, ei1));
        if (d2 < dhat2) {
            const auto pair = std::make_shared<HighOrderCollisionTemplate<Edge2P1, Vertex2>>(
                ei, vi, mesh, params, dhat, vertices);
            pair->weight = -1;
            add_collision<HighOrderCollision>(pair, vert_edge_2_to_id, collisions);
        }
    }
}

void HighOrderCollisionsBuilder<2>::add_edge_edge_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeEdgeCandidate>& candidates,
    const HighOrderContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const size_t start_i,
    const size_t end_i)
{
    if (params.quad_order == 0) throw std::logic_error("Vertex integration temporarily removed");
    const double dhat = params.dhat;
    //const double dhat2 = dhat * dhat;

    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, ej] = candidates[i];
        auto collision = reduce_edge_edge_collision(ei, ej, dhat, mesh, vertices, params);
        if (!collision) {
            continue;
        }
        if (collision->type() == HighOrderCollisionType::EDGE_EDGE) {
            add_collision<HighOrderCollision>(
                std::static_pointer_cast<HighOrderCollisionTemplate<Edge2P1, Edge2P1>>(collision),
                edge_edge_2_to_id, collisions);
        } else {
            add_collision<HighOrderCollision>(
                std::static_pointer_cast<HighOrderCollisionTemplate<Edge2P1, Vertex2>>(collision),
                vert_edge_2_to_id, collisions);
        }
    }
}

std::shared_ptr<HighOrderCollision> HighOrderCollisionsBuilder<2>::reduce_edge_edge_collision(
    const index_t ei,
    const index_t ej,
    const double dhat,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const HighOrderContactParameters& params)
{
    const auto& ea0 = vertices.row(mesh.edges()(ei, 0));
    const auto& ea1 = vertices.row(mesh.edges()(ei, 1));
    const auto& eb0 = vertices.row(mesh.edges()(ej, 0));
    const auto& eb1 = vertices.row(mesh.edges()(ej, 1));

    const auto dtype0 = point_edge_distance_type(ea0, eb0, eb1);
    const auto dtype1 = point_edge_distance_type(ea1, eb0, eb1);

    if (dtype0 == dtype1 && (dtype0 == PointEdgeDistanceType::P_E0 || dtype0 == PointEdgeDistanceType::P_E1)) {
        const index_t vi = (dtype0 == PointEdgeDistanceType::P_E0) ? mesh.edges()(ej, 0) : mesh.edges()(ej, 1);
        if (point_edge_distance(vertices.row(vi), ea0, ea1) >= dhat * dhat) {
            return nullptr;
        }
        return std::make_shared<HighOrderCollisionTemplate<Edge2P1, Vertex2>>(
            ei, vi, mesh, params, dhat, vertices);
    }
    else {
        const double dist_sqr = std::min({
            point_edge_distance(ea0, eb0, eb1),
            point_edge_distance(ea1, eb0, eb1),
            point_edge_distance(eb0, ea0, ea1),
            point_edge_distance(eb1, ea0, ea1)
        });
        if (dist_sqr >= dhat * dhat) {
            return nullptr;
        }
        return std::make_shared<HighOrderCollisionTemplate<Edge2P1, Edge2P1>>(
            ei, ej, mesh, params, dhat, vertices);
    }
}

void HighOrderCollisionsBuilder<2>::merge(
    const ParallelCacheType<HighOrderCollisionsBuilder<2>>& local_storage,
    HighOrderCollisions& merged_collisions)
{
    unordered_map<std::pair<index_t, index_t>, index_t> edge_edge_2_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_vert_2_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_edge_2_to_id;

    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // merge
    for (const auto& builder : local_storage) {
        for (const auto& ve : builder.vert_edge_2_to_id) {
            add_collision<HighOrderCollision>(builder.collisions[ve.second], vert_edge_2_to_id, merged_collisions.collisions);
        }
        for (const auto& ee : builder.edge_edge_2_to_id) {
            add_collision<HighOrderCollision>(builder.collisions[ee.second], edge_edge_2_to_id, merged_collisions.collisions);
        }
    }

    // remove 0-weight collisions
    merged_collisions.collisions.erase(
    std::remove_if(
        merged_collisions.collisions.begin(), merged_collisions.collisions.end(),
        [&](std::shared_ptr<HighOrderCollision> cc) {
            return cc->weight == 0;
        }), merged_collisions.collisions.end());

    int edge_edge_count = edge_edge_2_to_id.size();
    int vert_vert_count = vert_vert_2_to_id.size();
    int vert_edge_count = vert_edge_2_to_id.size();

    logger().trace(
        "VV pairs: {}; VE pairs: {}; EE pairs: {}.",
        vert_vert_count, vert_edge_count, edge_edge_count);
}

// ============================================================================

std::shared_ptr<HighOrderCollision> HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
    const FaceVertexCandidate& candidate,
    const HighOrderContactParameters& params,
    const CollisionMesh& mesh,
    const VertexMatrixView<3>& vertices,
    PointTriangleDistanceType dtype)
{
    const index_t vi = candidate.vertex_id;
    const index_t fi = candidate.face_id;

    const index_t t0 = mesh.faces()(fi, 0);
    const index_t t1 = mesh.faces()(fi, 1);
    const index_t t2 = mesh.faces()(fi, 2);

    const index_t e0 = mesh.faces_to_edges()(fi, 0);
    const index_t e1 = mesh.faces_to_edges()(fi, 1);
    const index_t e2 = mesh.faces_to_edges()(fi, 2);

    assert(vi != t0 && vi != t1 && vi != t2);

    if (dtype == PointTriangleDistanceType::AUTO) {
        dtype = point_triangle_distance_type(vertices(vi),
            vertices(t0),
            vertices(t1),
            vertices(t2));
    }

    const double dist_sqr = point_triangle_distance(vertices(vi),
            vertices(t0),
            vertices(t1),
            vertices(t2), dtype);
    if (dist_sqr >= params.dhat * params.dhat) {
        return nullptr;
    }

    switch (dtype) {
    case PointTriangleDistanceType::P_T0:
        return std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
            t0, vi, mesh);

    case PointTriangleDistanceType::P_T1:
        return std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
            t1, vi, mesh);

    case PointTriangleDistanceType::P_T2:
        return std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
            t2, vi, mesh);

    case PointTriangleDistanceType::P_E0:
        return std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
            e0, vi, mesh);

    case PointTriangleDistanceType::P_E1:
        return std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
            e1, vi, mesh);

    case PointTriangleDistanceType::P_E2:
        return std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
            e2, vi, mesh);

    case PointTriangleDistanceType::P_T:
        return std::make_shared<HighOrderCollision3DTemplate<Face3P1, Vertex3>>(
            fi, vi, mesh);

    case PointTriangleDistanceType::AUTO:
    default:
        assert(false);
        return std::make_shared<HighOrderCollision3DTemplate<Face3P1, Vertex3>>(
            fi, vi, mesh);
    }
}

std::shared_ptr<HighOrderCollision> HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
    const EdgeVertexCandidate& candidate,
    const HighOrderContactParameters& params,
    const CollisionMesh& mesh,
    const VertexMatrixView<3>& vertices,
    PointEdgeDistanceType dtype)
{
    const index_t vi = candidate.vertex_id;
    const index_t ei = candidate.edge_id;

    const index_t t0 = mesh.edges()(ei, 0);
    const index_t t1 = mesh.edges()(ei, 1);

    if (dtype == PointEdgeDistanceType::AUTO) {
        dtype = point_edge_distance_type(vertices(vi),
            vertices(t0),
            vertices(t1));
    }

    const double dist_sqr = point_edge_distance(vertices(vi),
            vertices(t0),
            vertices(t1), dtype);
    if (dist_sqr >= params.dhat * params.dhat) {
        return nullptr;
    }

    switch (dtype) {
    case PointEdgeDistanceType::P_E0:
        return std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
            t0, vi, mesh);
    case PointEdgeDistanceType::P_E1:
        return std::make_shared<HighOrderCollision3DTemplate<Vertex3, Vertex3>>(
            t1, vi, mesh);
    case PointEdgeDistanceType::P_E:
        return std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
            ei, vi, mesh);
    default:
        assert(false);
        return std::make_shared<HighOrderCollision3DTemplate<Edge3P1, Vertex3>>(
            ei, vi, mesh);
    }
}

// ============================================================================
// QuadratureCollisionsBuilder

QuadratureCollisionsBuilder::QuadratureCollisionsBuilder(
    const CollisionMesh& mesh,
    const Candidates& candidates,
    const HighOrderContactParameters& params)
    : point_potential(
        std::make_shared<PointPotential>(mesh, candidates, params))
{
}

QuadratureCollisionsBuilder::~QuadratureCollisionsBuilder() = default;

QuadratureCollisionsBuilder::QuadratureCollisionsBuilder(const QuadratureCollisionsBuilder& other)
{
    point_potential = other.point_potential;
    vertex_collisions.clear();
    for (const auto& cc : other.vertex_collisions) {
        vertex_collisions.push_back(std::make_unique<HighOrderCollisionDict<PointType::VERTEX>>(*cc));
    }
    edge_edge_collisions.clear();
    for (const auto& cc : other.edge_edge_collisions) {
        edge_edge_collisions.push_back(std::make_unique<HighOrderCollisionDict<PointType::EDGE>>(*cc));
    }
    face_collisions.clear();
    for (const auto& [fi, dicts] : other.face_collisions) {
        std::vector<std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>> copied;
        for (const auto& d : dicts) {
            copied.push_back(std::make_unique<HighOrderCollisionDict<PointType::FACE>>(*d));
        }
        face_collisions.push_back({fi, std::move(copied)});
    }
}
QuadratureCollisionsBuilder& QuadratureCollisionsBuilder::operator=(const QuadratureCollisionsBuilder& other)
{
    point_potential = other.point_potential;
    vertex_collisions.clear();
    for (const auto& cc : other.vertex_collisions) {
        vertex_collisions.push_back(std::make_unique<HighOrderCollisionDict<PointType::VERTEX>>(*cc));
    }
    edge_edge_collisions.clear();
    for (const auto& cc : other.edge_edge_collisions) {
        edge_edge_collisions.push_back(std::make_unique<HighOrderCollisionDict<PointType::EDGE>>(*cc));
    }
    face_collisions.clear();
    for (const auto& [fi, dicts] : other.face_collisions) {
        std::vector<std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>> copied;
        for (const auto& d : dicts) {
            copied.push_back(std::make_unique<HighOrderCollisionDict<PointType::FACE>>(*d));
        }
        face_collisions.push_back({fi, std::move(copied)});
    }
    return *this;
}

void QuadratureCollisionsBuilder::build_vertex_collisions(
    const Eigen::MatrixXd& vertices,
    const std::vector<index_t>& vertex_indices,
    const size_t start_i,
    const size_t end_i)
{
    const CollisionMesh& mesh = point_potential->mesh;
    const HighOrderContactParameters& params = point_potential->params;
    for (size_t i = start_i; i < end_i; i++) {
        const index_t vi = vertex_indices[i];
        if (params.integration_type == IntegrationType::NO_OBST && mesh.is_obstacle_vertex(vi)) continue;
        if (params.integration_type != IntegrationType::BRUTE_FORCE && mesh.is_obstacle_vertex(vi)) {
            const auto v_set = point_potential->candidates.vv_set(vi);
            const auto e_set = point_potential->candidates.ve_set(vi);
            const auto f_set = point_potential->candidates.vf_set(vi);
            const bool has_non_obstacle =
                std::any_of(v_set.begin(), v_set.end(), [&](index_t v){ return !mesh.is_obstacle_vertex(v); }) ||
                std::any_of(e_set.begin(), e_set.end(), [&](index_t e){ return !mesh.is_obstacle_edge(e); }) ||
                std::any_of(f_set.begin(), f_set.end(), [&](index_t f){ return !mesh.is_obstacle_face(f); });
            if (!has_non_obstacle) continue;
        }
        size_t n = 0;
        vertex_collisions.push_back(point_potential->build_collisions_at_vertex(vertices, vi, n));
        num_collision_pairs += n;
    }
}

void QuadratureCollisionsBuilder::build_face_collisions(
    const Eigen::MatrixXd& vertices,
    const std::vector<index_t>& face_indices,
    const size_t start_i,
    const size_t end_i)
{
    const CollisionMesh& mesh = point_potential->mesh;
    const auto& face_quad_rule = TriangularQuadrature::get_rule(point_potential->params.quad_order);
    if (face_quad_rule.empty()) return;

    const HighOrderContactParameters& params = point_potential->params;
    for (size_t i = start_i; i < end_i; i++) {
        const index_t fi = face_indices[i];
        if (params.integration_type == IntegrationType::NO_OBST && mesh.is_obstacle_face(fi)) continue;
        if (params.integration_type != IntegrationType::BRUTE_FORCE && mesh.is_obstacle_face(fi)) {
            const auto v_set = point_potential->candidates.fv_set(fi);
            const auto e_set = point_potential->candidates.fe_set(fi);
            const auto f_set = point_potential->candidates.ff_set(fi);
            const bool has_non_obstacle =
                std::any_of(v_set.begin(), v_set.end(), [&](index_t v){ return !mesh.is_obstacle_vertex(v); }) ||
                std::any_of(e_set.begin(), e_set.end(), [&](index_t e){ return !mesh.is_obstacle_edge(e); }) ||
                std::any_of(f_set.begin(), f_set.end(), [&](index_t f){ return !mesh.is_obstacle_face(f); });
            if (!has_non_obstacle) continue;
        }

        std::vector<std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>> per_qp_dicts;
        per_qp_dicts.reserve(face_quad_rule.size());
        for (const auto& qp : face_quad_rule) {
            size_t n = 0;
            per_qp_dicts.push_back(
                point_potential->build_collisions_at_face_interior_point(vertices, fi, qp.lambda, n));
            num_collision_pairs += n;
        }
        face_collisions.push_back({fi, std::move(per_qp_dicts)});
    }
}

void QuadratureCollisionsBuilder::build_edge_edge_collisions(
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeEdgeCandidate>& ee_candidates,
    const size_t start_i,
    const size_t end_i)
{
    const HighOrderContactParameters& params = point_potential->params;
    const CollisionMesh& mesh = point_potential->mesh;

    // Returns true if edge e (which is an obstacle) has at least one non-obstacle candidate.
    // Used in NORMAL mode to skip placing a QP on an obstacle edge with only obstacle candidates.
    auto obstacle_edge_has_non_obstacle_candidates = [&](index_t e) -> bool {
        const auto v_set = point_potential->candidates.ev_set(e);
        const auto e_set = point_potential->candidates.ee_set(e);
        const auto f_set = point_potential->candidates.ef_set(e);
        return std::any_of(v_set.begin(), v_set.end(), [&](index_t v){ return !mesh.is_obstacle_vertex(v); })
            || std::any_of(e_set.begin(), e_set.end(), [&](index_t e2){ return !mesh.is_obstacle_edge(e2); })
            || std::any_of(f_set.begin(), f_set.end(), [&](index_t f){ return !mesh.is_obstacle_face(f); });
    };

    for (size_t i = start_i; i < end_i; i++) {
        const auto& candidate = ee_candidates[i];
        const index_t ei = candidate.edge0_id;
        const index_t ej = candidate.edge1_id;

        const index_t ea = mesh.edges()(ei, 0);
        const index_t eb = mesh.edges()(ei, 1);
        const index_t ec = mesh.edges()(ej, 0);
        const index_t ed = mesh.edges()(ej, 1);

        if (ea == ec || ea == ed || eb == ec || eb == ed) {
            continue;
        }

        if (params.integration_type != IntegrationType::BRUTE_FORCE
            && mesh.is_obstacle_edge(ei) && mesh.is_obstacle_edge(ej)) {
            continue;
        }


        if (is_parallel_edge_edge(
            vertices.row(ea), vertices.row(eb),
            vertices.row(ec), vertices.row(ed))) {
            continue;
        }

        const auto dtype = edge_edge_distance_type(
            vertices.row(ea), vertices.row(eb),
            vertices.row(ec), vertices.row(ed));

        const double dist_sq = edge_edge_distance(
            vertices.row(ea), vertices.row(eb),
            vertices.row(ec), vertices.row(ed), dtype);

        if (dist_sq >= params.dbar * params.dbar) {
            continue;
        }

        const bool ei_is_obs = mesh.is_obstacle_edge(ei);
        const bool ej_is_obs = mesh.is_obstacle_edge(ej);

        if ((params.integration_type != IntegrationType::NO_OBST || !ei_is_obs)
            && (!ei_is_obs || params.integration_type == IntegrationType::BRUTE_FORCE || obstacle_edge_has_non_obstacle_candidates(ei))
            && (dtype == EdgeEdgeDistanceType::EA_EB ||
                dtype == EdgeEdgeDistanceType::EA_EB0 ||
                dtype == EdgeEdgeDistanceType::EA_EB1)) {
            size_t n = 0;
            edge_edge_collisions.push_back(point_potential->build_collisions_at_edge_edge_closest_point(vertices, ei, ej, dtype, n));
            num_collision_pairs += n;
        }

        if ((params.integration_type != IntegrationType::NO_OBST || !ej_is_obs)
            && (!ej_is_obs || params.integration_type == IntegrationType::BRUTE_FORCE || obstacle_edge_has_non_obstacle_candidates(ej))
            && (dtype == EdgeEdgeDistanceType::EA_EB ||
                dtype == EdgeEdgeDistanceType::EA0_EB ||
                dtype == EdgeEdgeDistanceType::EA1_EB)) {
            size_t n = 0;
            edge_edge_collisions.push_back(point_potential->build_collisions_at_edge_edge_closest_point(vertices, ej, ei, reflectEdgeEdgeDistanceType(dtype), n));
            num_collision_pairs += n;
        }
    }
}

void QuadratureCollisionsBuilder::merge(
    ParallelCacheType<QuadratureCollisionsBuilder>& local_storage,
    HighOrderCollisions& merged_collisions)
{
    // Reserve space
    size_t total_v = 0, total_ee = 0, total_f = 0;
    for (const auto& storage : local_storage) {
        total_v += storage.vertex_collisions.size();
        total_ee += storage.edge_edge_collisions.size();
        total_f += storage.face_collisions.size();
    }
    merged_collisions.vertex_collisions.reserve(total_v);
    merged_collisions.edge_edge_collisions.reserve(total_ee);
    merged_collisions.face_collisions.reserve(total_f);

    for (auto& storage : local_storage) {
        for (auto& cc : storage.vertex_collisions) {
            merged_collisions.vertex_collisions.insert(std::make_pair<index_t, std::unique_ptr<HighOrderCollisionDict<PointType::VERTEX>>>(cc->primitive_id(), std::move(cc)));
        }
        for (auto& cc : storage.edge_edge_collisions) {
            const auto id = cc->primitive_ids();
            merged_collisions.edge_edge_collisions.insert(std::make_pair(std::make_pair(id[0], id[1]), std::move(cc)));
        }
        for (auto& [fi, dicts] : storage.face_collisions) {
            merged_collisions.face_collisions.emplace(fi, std::move(dicts));
        }
        merged_collisions.num_quadrature_collision_pairs += storage.num_collision_pairs;
    }
}

} // namespace ipc
