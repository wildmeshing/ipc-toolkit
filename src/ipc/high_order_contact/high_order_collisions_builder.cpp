#include "high_order_collisions_builder.hpp"

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

#include <tbb/enumerable_thread_specific.h>

namespace ipc {

namespace {
    template <typename TCollision>
    void add_collision(
        const std::shared_ptr<TCollision>& pair,
        unordered_map<std::pair<index_t, index_t>, std::shared_ptr<TCollision>>&
            cc_to_id,
        std::vector<std::shared_ptr<HighOrderCollision>>& collisions)
    {
        if (pair->is_active()) { // filters dupes
            auto found_item = cc_to_id.find(pair->get_hash());
            if (found_item == cc_to_id.end()) {
                // New collision, so add it to the end of collisions
                cc_to_id.emplace(pair->get_hash(), pair);
                collisions.push_back(pair);
            }
            else {
                found_item->second->weight += pair->weight;
            }
        }
    }

    template <typename TCollision, typename THash>
    void add_collision(
        const std::shared_ptr<TCollision>& pair,
        unordered_map<THash, index_t>& cc_to_id,
        std::vector<std::shared_ptr<TCollision>>& collisions)
    {
        if (pair->is_active()) {
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
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, vi] = candidates[i];
		const PointEdgeDistanceType pe_dtype = point_edge_distance_type(
			vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
			vertices.row(mesh.edges()(ei, 1)));

        const double distance_sqr = point_edge_distance(
            vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)), pe_dtype);
        assert(distance_sqr >= 0);
        const double dhat_EV = std::min(vert_dhat(vi), edge_dhat(ei));
        if (distance_sqr < dhat_EV * dhat_EV) {
            add_collision(
                std::make_shared<HighOrderCollisionTemplate<Edge2P1, Vertex2>>(
                    ei, vi, mesh, params, dhat_EV, vertices),
                vert_edge_2_to_id, collisions);
        }

        if (params.quad_points == 0) {
            // vertex-vertex
            for (int j = 0; j < 2; j++) {
                const index_t vj = mesh.edges()(ei, j);
                const double dhat_VV = std::min(vert_dhat(vi), vert_dhat(vj));
                if ((vertices.row(vi) - vertices.row(vj)).norm() < dhat_VV) {
                    add_collision(
                        std::make_shared<HighOrderCollisionTemplate<Vertex2, Vertex2>>(
                            std::min(vi, vj), std::max(vi, vj), mesh, params,
                            dhat_VV, vertices),
                        vert_vert_2_to_id, collisions);
                }
            }
        }
        else {
            // edge-edge
            for (const index_t ej : mesh.vertices_to_edges()[vi]) {
                const double dhat_EE = std::min(edge_dhat(ei), edge_dhat(ej));
                add_collision(
                    std::make_shared<HighOrderCollisionTemplate<Edge2P1, Edge2P1>>(
                        std::min<index_t>(ei, ej), std::max<index_t>(ei, ej),
                        mesh, params, dhat_EE, vertices),
                    edge_edge_2_to_id, collisions);
            }
		}
    }
}

void HighOrderCollisionsBuilder<2>::merge(
    const ParallelCacheType<HighOrderCollisionsBuilder<2>>& local_storage,
    HighOrderCollisions& merged_collisions)
{
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge2P1, Edge2P1>>>
        edge_edge_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Vertex2, Vertex2>>>
        vert_vert_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge2P1, Vertex2>>>
        vert_edge_2_to_id;

    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // merge
    for (const auto& builder : local_storage) {
        edge_edge_2_to_id.insert(
            builder.edge_edge_2_to_id.begin(), builder.edge_edge_2_to_id.end());
        vert_vert_2_to_id.insert(
            builder.vert_vert_2_to_id.begin(), builder.vert_vert_2_to_id.end());
        vert_edge_2_to_id.insert(
            builder.vert_edge_2_to_id.begin(), builder.vert_edge_2_to_id.end());
    }
    int edge_edge_count = edge_edge_2_to_id.size();
    int vert_vert_count = vert_vert_2_to_id.size();
    int vert_edge_count = vert_edge_2_to_id.size();

    for (const auto& [key, val] : edge_edge_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : vert_vert_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : vert_edge_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }

    logger().trace(
        "VV pairs: {}; VE pairs: {}; EE pairs: {}.",
        vert_vert_count, vert_edge_count, edge_edge_count);
}

// ============================================================================


void HighOrderCollisionsBuilder<3>::add_edge_edge_face_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<std::array<index_t, 3>>& candidates,
    const HighOrderContactParameters& params,
    const double dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, ej, fk] = candidates[i];

        const EdgeEdgeDistanceType dtype = edge_edge_distance_type(
            vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)),
            vertices.row(mesh.edges()(ej, 0)),
            vertices.row(mesh.edges()(ej, 1))
        );

        if (dtype != EdgeEdgeDistanceType::EA_EB) {
            continue;
        }

        const double dist_sqr = edge_edge_distance(
            vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)),
            vertices.row(mesh.edges()(ej, 0)),
            vertices.row(mesh.edges()(ej, 1)),
            dtype
        );

        if (dist_sqr >= dhat * dhat)
            continue;

        auto pair = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Face3P1>>(
            ei, ej, fk, mesh, params, dhat, vertices);

        if (!pair->is_active()) {
            continue;
        }

        // slow version
        // add_collision<TriplePairCollision, std::array<index_t, 3>>(
        //     pair, eef_3_to_id, triple_collisions);

        // fast version
        switch (pair->distance_type_2()) {
        case PointTriangleDistanceType::P_T0:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        ei, ej, mesh.faces()(fk, 0), mesh, params, dhat, vertices), eev_3_to_id, triple_collisions);
                break;
            }
        case PointTriangleDistanceType::P_T1:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        ei, ej, mesh.faces()(fk, 1), mesh, params, dhat, vertices), eev_3_to_id, triple_collisions);
                break;
            }
        case PointTriangleDistanceType::P_T2:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        ei, ej, mesh.faces()(fk, 2), mesh, params, dhat, vertices), eev_3_to_id, triple_collisions);
                break;
            }
        case PointTriangleDistanceType::P_E0:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                        ei, ej, mesh.faces_to_edges()(fk, 0), mesh, params, dhat, vertices), eee_3_to_id, triple_collisions);
                break;
            }
        case PointTriangleDistanceType::P_E1:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                        ei, ej, mesh.faces_to_edges()(fk, 1), mesh, params, dhat, vertices), eee_3_to_id, triple_collisions);
                break;
            }
        case PointTriangleDistanceType::P_E2:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                        ei, ej, mesh.faces_to_edges()(fk, 2), mesh, params, dhat, vertices), eee_3_to_id, triple_collisions);
                break;
            }
        case PointTriangleDistanceType::P_T:
            {
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    pair, eef_3_to_id, triple_collisions);
                break;
            }
        default:
            assert(false);
            break;
        }
    }
}

void HighOrderCollisionsBuilder<3>::add_edge_edge_vertex_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<std::array<index_t, 3>>& candidates,
    const HighOrderContactParameters& params,
    const double dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, ej, vk] = candidates[i];

        const EdgeEdgeDistanceType dtype = edge_edge_distance_type(
            vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)),
            vertices.row(mesh.edges()(ej, 0)),
            vertices.row(mesh.edges()(ej, 1))
        );

        if (dtype != EdgeEdgeDistanceType::EA_EB) {
            continue;
        }

        const double dist_sqr = edge_edge_distance(
            vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)),
            vertices.row(mesh.edges()(ej, 0)),
            vertices.row(mesh.edges()(ej, 1)),
            dtype
        );

        if (dist_sqr >= dhat * dhat)
            continue;

        add_collision<TriplePairCollision, std::array<index_t, 3>>(
            std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                ei, ej, vk, mesh, params, dhat, vertices),
            eev_3_to_id, triple_collisions);
    }
}

void HighOrderCollisionsBuilder<3>::add_negative_edge_edge_edge_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<std::array<index_t, 3>>& candidates,
    const HighOrderContactParameters& params,
    const double dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, ej, ek] = candidates[i];

        const EdgeEdgeDistanceType dtype = edge_edge_distance_type(
            vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)),
            vertices.row(mesh.edges()(ej, 0)),
            vertices.row(mesh.edges()(ej, 1))
        );

        if (dtype != EdgeEdgeDistanceType::EA_EB) {
            continue;
        }

        const double dist_sqr = edge_edge_distance(
            vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)),
            vertices.row(mesh.edges()(ej, 0)),
            vertices.row(mesh.edges()(ej, 1)),
            dtype
        );

        if (dist_sqr >= dhat * dhat)
            continue;

        auto triple = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>>(
                ei, ej, ek, mesh, params, dhat, vertices);

        // slow version
        // triple->weight = -1;
        // add_collision<TriplePairCollision, std::array<index_t, 3>>(
        //     triple,
        //     eee_3_to_id, triple_collisions);

        // fast version
        if (!triple->is_active()) {
            continue;
        }

        switch (triple->distance_type_2()) {
            case PointEdgeDistanceType::P_E0:
            {
                auto triple2 = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        ei, ej, mesh.edges()(ek, 0), mesh, params, dhat, vertices);
                triple2->weight = -1;
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    triple2, eev_3_to_id, triple_collisions);
                break;
            }
            case PointEdgeDistanceType::P_E1:
            {
                auto triple2 = std::make_shared<TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>>(
                        ei, ej, mesh.edges()(ek, 1), mesh, params, dhat, vertices);
                triple2->weight = -1;
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    triple2, eev_3_to_id, triple_collisions);
                break;
            }
            case PointEdgeDistanceType::P_E:
            {
                triple->weight = -1;
                add_collision<TriplePairCollision, std::array<index_t, 3>>(
                    triple,
                    eee_3_to_id, triple_collisions);
                break;
            }
            default:
                assert(false);
                break;
        }
    }
}

std::shared_ptr<HighOrderCollision> HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
    const FaceVertexCandidate& candidate,
    const HighOrderContactParameters& params,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
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

    if (dtype == PointTriangleDistanceType::AUTO) {
        dtype = point_triangle_distance_type(vertices.row(vi),
            vertices.row(t0),
            vertices.row(t1),
            vertices.row(t2));
    }

    switch (dtype) {
    case PointTriangleDistanceType::P_T0:
        return std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
            std::min(t0, vi), std::max(t0, vi), mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::P_T1:
        return std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
            std::min(t1, vi), std::max(t1, vi), mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::P_T2:
        return std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
            std::min(t2, vi), std::max(t2, vi), mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::P_E0:
        return std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
            e0, vi, mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::P_E1:
        return std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
            e1, vi, mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::P_E2:
        return std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
            e2, vi, mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::P_T:
        return std::make_shared<HighOrderCollisionTemplate<Face3P1, Vertex3>>(
            fi, vi, mesh, params, params.dhat, vertices);

    case PointTriangleDistanceType::AUTO:
    default:
        assert(false);
        return std::make_shared<HighOrderCollisionTemplate<Face3P1, Vertex3>>(
            fi, vi, mesh, params, params.dhat, vertices);
    }
}

std::shared_ptr<HighOrderCollision> HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
    const EdgeVertexCandidate& candidate,
    const HighOrderContactParameters& params,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    PointEdgeDistanceType dtype)
{
    const index_t vi = candidate.vertex_id;
    const index_t ei = candidate.edge_id;

    const index_t t0 = mesh.edges()(ei, 0);
    const index_t t1 = mesh.edges()(ei, 1);

    if (dtype == PointEdgeDistanceType::AUTO) {
        dtype = point_edge_distance_type(vertices.row(vi),
            vertices.row(t0),
            vertices.row(t1));
    }

    switch (dtype) {
    case PointEdgeDistanceType::P_E0:
        return std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
            std::min(t0, vi), std::max(t0, vi), mesh, params, params.dhat, vertices);
    case PointEdgeDistanceType::P_E1:
        return std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
            std::min(t1, vi), std::max(t1, vi), mesh, params, params.dhat, vertices);
    case PointEdgeDistanceType::P_E:
        return std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
            ei, vi, mesh, params, params.dhat, vertices);
    default:
        assert(false);
        return std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
            ei, vi, mesh, params, params.dhat, vertices);
    }
}

void HighOrderCollisionsBuilder<3>::add_face_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<FaceVertexCandidate>& candidates,
        const HighOrderContactParameters& params,
        const size_t start_i,
        const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [fi, vi] = candidates[i];
        assert(mesh.faces()(fi, 0) != vi && mesh.faces()(fi, 1) != vi && mesh.faces()(fi, 2) != vi);
        const auto [v, f0, f1, f2] =
            candidates[i].vertices(vertices, mesh.edges(), mesh.faces());

        // Compute distance type
        const PointTriangleDistanceType dtype =
            point_triangle_distance_type(v, f0, f1, f2);
        const double distance_sqr =
            point_triangle_distance(v, f0, f1, f2, dtype);

        if (distance_sqr >= params.dhat * params.dhat) {
            continue;
        }

        // slow version
        // add_collision<HighOrderCollision>(
        //     std::make_shared<HighOrderCollisionTemplate<Face3P1, Vertex3>>(
        //         fi, vi, mesh, params, params.dhat, vertices),
        //     vert_face_3_to_id, collisions);

        // fast version
        const index_t t0 = mesh.faces()(fi, 0);
        const index_t t1 = mesh.faces()(fi, 1);
        const index_t t2 = mesh.faces()(fi, 2);

        const index_t e0 = mesh.faces_to_edges()(fi, 0);
        const index_t e1 = mesh.faces_to_edges()(fi, 1);
        const index_t e2 = mesh.faces_to_edges()(fi, 2);

        switch (dtype) {
        case PointTriangleDistanceType::P_T0:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                    std::min(t0, vi), std::max(t0, vi), mesh, params, params.dhat, vertices), vert_vert_3_to_id,
                collisions);
            break;

        case PointTriangleDistanceType::P_T1:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                    std::min(t1, vi), std::max(t1, vi), mesh, params, params.dhat, vertices), vert_vert_3_to_id,
                collisions);
            break;

        case PointTriangleDistanceType::P_T2:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                    std::min(t2, vi), std::max(t2, vi), mesh, params, params.dhat, vertices), vert_vert_3_to_id,
                collisions);
            break;

        case PointTriangleDistanceType::P_E0:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                    e0, vi, mesh, params, params.dhat, vertices), vert_edge_3_to_id,
                collisions);
            break;

        case PointTriangleDistanceType::P_E1:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                    e1, vi, mesh, params, params.dhat, vertices), vert_edge_3_to_id,
                collisions);
            break;

        case PointTriangleDistanceType::P_E2:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                    e2, vi, mesh, params, params.dhat, vertices), vert_edge_3_to_id,
                collisions);
            break;

        case PointTriangleDistanceType::P_T:
            add_collision<HighOrderCollision>(
                std::make_shared<HighOrderCollisionTemplate<Face3P1, Vertex3>>(
                    fi, vi, mesh, params, params.dhat, vertices),
                vert_face_3_to_id, collisions);
            break;

        case PointTriangleDistanceType::AUTO:
        default:
            assert(false);
            break;
        }
    }
}

void HighOrderCollisionsBuilder<3>::add_face_vertex_negative_edge_vertex_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeVertexCandidate>& candidates,
    const HighOrderContactParameters& params,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, vi] = candidates[i];
        assert(mesh.edges()(ei, 0) != vi && mesh.edges()(ei, 1) != vi);
        const auto [v, e0, e1, _] =
            candidates[i].vertices(vertices, mesh.edges(), mesh.faces());

        // Compute distance type
        const PointEdgeDistanceType dtype =
            point_edge_distance_type(v, e0, e1);
        const double distance_sqr =
            point_edge_distance(v, e0, e1, dtype);

        if (distance_sqr >= params.dhat * params.dhat) {
            continue;
        }

        // slow version
        // auto pair = std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
        //         ei, vi, mesh, params, params.dhat, vertices);
        // pair->weight = -1;
        // add_collision<HighOrderCollision>(
        //     pair,
        //     vert_edge_3_to_id, collisions);

        // fast version
        const index_t t0 = mesh.edges()(ei, 0);
        const index_t t1 = mesh.edges()(ei, 1);

        switch (dtype) {
        case PointEdgeDistanceType::P_E0:
            {
                auto pair = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                    std::min(t0, vi), std::max(t0, vi), mesh, params, params.dhat, vertices);
                pair->weight = -1;
                add_collision<HighOrderCollision>(
                    pair,
                    vert_vert_3_to_id, collisions);
                break;
            }
        case PointEdgeDistanceType::P_E1:
            {
                auto pair = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                    std::min(t1, vi), std::max(t1, vi), mesh, params, params.dhat, vertices);
                pair->weight = -1;
                add_collision<HighOrderCollision>(
                    pair,
                    vert_vert_3_to_id, collisions);
                break;
            }
        case PointEdgeDistanceType::P_E:
            {
                auto pair = std::make_shared<HighOrderCollisionTemplate<Edge3P1, Vertex3>>(
                    ei, vi, mesh, params, params.dhat, vertices);
                pair->weight = -1;
                add_collision<HighOrderCollision>(
                    pair,
                    vert_edge_3_to_id, collisions);
                break;
            }
        default:
            assert(false);
            break;
        }
    }
}

void HighOrderCollisionsBuilder<3>::add_face_vertex_positive_vertex_vertex_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<VertexVertexCandidate>& candidates,
    const HighOrderContactParameters& params,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [vi, vj] = candidates[i];
        assert(vi != vj);

        // vertex-vertex
        auto pair = std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                std::min(vi, vj), std::max(vi, vj), mesh, params, params.dhat, vertices);
        pair->weight = 1;
        add_collision<HighOrderCollision>(
            pair,
            vert_vert_3_to_id, collisions);
    }
}

void HighOrderCollisionsBuilder<3>::merge(
    const ParallelCacheType<HighOrderCollisionsBuilder<3>>& local_storage,
    HighOrderCollisions& merged_collisions)
{
    unordered_map<std::pair<index_t, index_t>, index_t> vert_vert_3_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_edge_3_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_face_3_to_id;

    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // merge
    for (const auto& builder : local_storage) {
        for (const auto& vv : builder.vert_vert_3_to_id) {
            add_collision<HighOrderCollision>(builder.collisions[vv.second], vert_vert_3_to_id, merged_collisions.collisions);
        }
        for (const auto& ve : builder.vert_edge_3_to_id) {
            add_collision<HighOrderCollision>(builder.collisions[ve.second], vert_edge_3_to_id, merged_collisions.collisions);
        }
        for (const auto& vf : builder.vert_face_3_to_id) {
            add_collision<HighOrderCollision>(builder.collisions[vf.second], vert_face_3_to_id, merged_collisions.collisions);
        }
    }

    merged_collisions.collisions.erase(
        std::remove_if(
            merged_collisions.collisions.begin(), merged_collisions.collisions.end(),
            [&](std::shared_ptr<HighOrderCollision> cc) {
                return cc->weight == 0;
            }),
        merged_collisions.collisions.end());

    int vert_vert_count = vert_vert_3_to_id.size();
    int vert_edge_count = vert_edge_3_to_id.size();
    int vert_face_count = vert_face_3_to_id.size();

    logger().trace(
        "VV pairs: {}; VE pairs: {}; VF pairs: {}.",
        vert_vert_count, vert_edge_count, vert_face_count);


    unordered_map<std::array<index_t, 3>, index_t> eev_3_to_id;
    unordered_map<std::array<index_t, 3>, index_t> eee_3_to_id;
    unordered_map<std::array<index_t, 3>, index_t> eef_3_to_id;

    // size up the hash items
    total = 0;
    for (const auto& storage : local_storage) {
        total += storage.triple_collisions.size();
    }

    merged_collisions.triple_collisions.reserve(total);

    // merge
    for (const auto& builder : local_storage) {
        for (const auto& eev : builder.eev_3_to_id) {
            add_collision<TriplePairCollision>(builder.triple_collisions[eev.second], eev_3_to_id, merged_collisions.triple_collisions);
        }
        for (const auto& eee : builder.eee_3_to_id) {
            add_collision<TriplePairCollision>(builder.triple_collisions[eee.second], eee_3_to_id, merged_collisions.triple_collisions);
        }
        for (const auto& eef : builder.eef_3_to_id) {
            add_collision<TriplePairCollision>(builder.triple_collisions[eef.second], eef_3_to_id, merged_collisions.triple_collisions);
        }
    }

    merged_collisions.triple_collisions.erase(
    std::remove_if(
        merged_collisions.triple_collisions.begin(), merged_collisions.triple_collisions.end(),
        [&](std::shared_ptr<TriplePairCollision> cc) {
            return cc->weight == 0;
        }),
    merged_collisions.triple_collisions.end());

    int eev_count = eev_3_to_id.size();
    int eee_count = eee_3_to_id.size();
    int eef_count = eef_3_to_id.size();

    logger().trace(
        "EEV pairs: {}; EEE pairs: {}; EEF pairs: {}.",
        eev_count, eee_count, eef_count);
}

} // namespace ipc
