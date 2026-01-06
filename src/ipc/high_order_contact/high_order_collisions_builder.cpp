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
        if (pair->is_active()
            && cc_to_id.find(pair->get_hash()) == cc_to_id.end()) { // filters dupes
            // New collision, so add it to the end of collisions
            cc_to_id.emplace(pair->get_hash(), pair);
            collisions.push_back(pair);
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


void HighOrderCollisionsBuilder<3>::add_edge_edge_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeEdgeCandidate>& candidates,
    const HighOrderContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const std::function<double(const index_t)>& face_dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, ej] = candidates[i];
        const size_t vi0 = mesh.edges()(ei, 0);
        const size_t vi1 = mesh.edges()(ei, 1);
        const size_t vj0 = mesh.edges()(ej, 0);
        const size_t vj1 = mesh.edges()(ej, 1);
		const EdgeEdgeDistanceType pe_dtype = edge_edge_distance_type(
			vertices.row(vi0), vertices.row(vi1),
			vertices.row(vj0), vertices.row(vj1));

        const double distance_sqr = edge_edge_distance(
            vertices.row(vi0), vertices.row(vi1),
            vertices.row(vj0), vertices.row(vj1), pe_dtype);
        assert(distance_sqr >= 0);
        const double dhat_EE = std::min(edge_dhat(ej), edge_dhat(ei));
        if (params.quad_points == 0 && distance_sqr < dhat_EE * dhat_EE) {
            add_collision(
                std::make_shared<HighOrderCollisionTemplate<Edge3P1, Edge3P1>>(
                    std::min<int>(ei, ej), std::max<int>(ei, ej), mesh, params, dhat_EE, vertices),
                edge_edge_3_to_id, collisions);
        }

        if (params.quad_points != 0) {
            // edge-face
            const auto& edge_face_adj = mesh.edge_face_adjacencies();

            for (const int fi : edge_face_adj[ei]) {
                const double dhat_EF = std::min(edge_dhat(ej), face_dhat(fi));
                if (distance_sqr < dhat_EF * dhat_EF) {
                    add_collision(
                        std::make_shared<HighOrderCollisionTemplate<Edge3P1, Face3P1>>(
                            ej, fi, mesh, params, dhat_EF, vertices),
                        edge_face_3_to_id, collisions);
                }
            }

            for (const int fj : edge_face_adj[ej]) {
                const double dhat_EF = std::min(edge_dhat(ei), face_dhat(fj));
                if (distance_sqr < dhat_EF * dhat_EF) {
                    add_collision(
                        std::make_shared<HighOrderCollisionTemplate<Edge3P1, Face3P1>>(
                            ei, fj, mesh, params, dhat_EF, vertices),
                        edge_face_3_to_id, collisions);
                }
            }
		}
    }
}

void HighOrderCollisionsBuilder<3>::add_face_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<FaceVertexCandidate>& candidates,
        const HighOrderContactParameters& params,
        const std::function<double(const index_t)>& vert_dhat,
        const std::function<double(const index_t)>& edge_dhat,
        const std::function<double(const index_t)>& face_dhat,
        const size_t start_i,
        const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [fi, vi] = candidates[i];
        const auto [v, f0, f1, f2] =
            candidates[i].vertices(vertices, mesh.edges(), mesh.faces());

        // Compute distance type
        const PointTriangleDistanceType dtype =
            point_triangle_distance_type(v, f0, f1, f2);
        const double distance_sqr =
            point_triangle_distance(v, f0, f1, f2, dtype);

        // vertex-face
        const double dhat_FV = std::min(face_dhat(fi), vert_dhat(vi));
        if (distance_sqr < dhat_FV * dhat_FV) {
            add_collision(
                std::make_shared<HighOrderCollisionTemplate<Face3P1, Vertex3>>(
                    fi, vi, mesh, params, dhat_FV, vertices),
                vert_face_3_to_id, collisions);
        }

        // face-face
        for (const int fj : mesh.vertices_to_faces()[vi]) {
            const double dhat_FF = std::min(face_dhat(fi), face_dhat(fj));
            if (distance_sqr < dhat_FF * dhat_FF) {
                add_collision(
                    std::make_shared<HighOrderCollisionTemplate<Face3P1, Face3P1>>(
                        std::min(fi, fj), std::max(fi, fj), mesh, params, dhat_FF, vertices),
                    face_face_3_to_id, collisions);
            }
        }
    }
}

void HighOrderCollisionsBuilder<3>::merge(
    const ParallelCacheType<HighOrderCollisionsBuilder<3>>& local_storage,
    HighOrderCollisions& merged_collisions)
{
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Vertex3, Vertex3>>>
        vert_vert_3_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge3P1, Vertex3>>>
        vert_edge_3_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Face3P1, Vertex3>>>
        vert_face_3_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge3P1, Edge3P1>>>
        edge_edge_3_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge3P1, Face3P1>>>
        edge_face_3_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Face3P1, Face3P1>>>
        face_face_3_to_id;

    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // merge
    for (const auto& builder : local_storage) {
        vert_vert_3_to_id.insert(
            builder.vert_vert_3_to_id.begin(), builder.vert_vert_3_to_id.end());
        vert_edge_3_to_id.insert(
            builder.vert_edge_3_to_id.begin(), builder.vert_edge_3_to_id.end());
        vert_face_3_to_id.insert(
            builder.vert_face_3_to_id.begin(), builder.vert_face_3_to_id.end());
        edge_edge_3_to_id.insert(
        builder.edge_edge_3_to_id.begin(), builder.edge_edge_3_to_id.end());
        edge_face_3_to_id.insert(
            builder.edge_face_3_to_id.begin(), builder.edge_face_3_to_id.end());
        face_face_3_to_id.insert(
            builder.face_face_3_to_id.begin(), builder.face_face_3_to_id.end());
    }
    int vert_vert_count = vert_vert_3_to_id.size();
    int vert_edge_count = vert_edge_3_to_id.size();
    int vert_face_count = vert_face_3_to_id.size();
    int edge_edge_count = edge_edge_3_to_id.size();
    int edge_face_count = edge_face_3_to_id.size();
    int face_face_count = face_face_3_to_id.size();

    for (const auto& [key, val] : vert_vert_3_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : vert_edge_3_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : vert_face_3_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : edge_edge_3_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : edge_face_3_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : face_face_3_to_id) {
        merged_collisions.collisions.push_back(val);
    }

    logger().trace(
        "VV pairs: {}; VE pairs: {}; EE pairs: {}; VF pairs: {}; EF pairs: {}; FF pairs: {}.",
        vert_vert_count, vert_edge_count, edge_edge_count, vert_face_count, edge_face_count, face_face_count);
}

} // namespace ipc
