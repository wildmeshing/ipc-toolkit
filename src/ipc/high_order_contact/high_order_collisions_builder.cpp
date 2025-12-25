#include "high_order_collisions_builder.hpp"

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/point_edge.hpp>

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
		const double dhat = std::min(edge_dhat(ei), vert_dhat(vi));
		const PointEdgeDistanceType pe_dtype = point_edge_distance_type(
			vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
			vertices.row(mesh.edges()(ei, 1)));

		if (pe_dtype == PointEdgeDistanceType::P_E) {
			const double distance_sqr = point_edge_distance(
				vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
				vertices.row(mesh.edges()(ei, 1)), pe_dtype);
			assert(distance_sqr >= 0);
			if (distance_sqr < dhat * dhat) {
				add_collision(
					std::make_shared<HighOrderCollisionTemplate<Edge2P1, Vertex2>>(
						ei, vi, mesh, params, dhat, vertices),
					vert_edge_2_to_id, collisions);
			}
		}

		// vertex-vertex
		for (int j = 0; j < 2; j++) {
			const index_t vj = mesh.edges()(ei, j);
			const double vv_dhat = std::min(vert_dhat(vi), vert_dhat(vj));
			if ((vertices.row(vi) - vertices.row(vj)).norm() < vv_dhat) {
				add_collision(
					std::make_shared<HighOrderCollisionTemplate<Vertex2, Vertex2>>(
						std::min(vi, vj), std::max(vi, vj), mesh, params,
						vv_dhat, vertices),
					vert_vert_2_to_id, collisions);
			}
		}

		// edge-edge
		for (const index_t ej : mesh.vertices_to_edges()[vi]) {
			add_collision(
				std::make_shared<HighOrderCollisionTemplate<Edge2P1, Edge2P1>>(
					std::min<index_t>(ei, ej), std::max<index_t>(ei, ej),
					mesh, params, dhat, vertices),
				edge_edge_2_to_id, collisions);
		}
    }
}

// ============================================================================

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

} // namespace ipc
