#include "high_order_collisions_builder.hpp"

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

#include <tbb/enumerable_thread_specific.h>

namespace ipc {

namespace {
    template <int dim, typename TCollision>
    void add_collision(
        const std::shared_ptr<TCollision>& pair,
        unordered_map<std::pair<index_t, index_t>, std::shared_ptr<TCollision>>&
            cc_to_id,
        std::vector<std::shared_ptr<SmoothCollision>>& collisions)
    {
        if (pair->is_active()
            && cc_to_id.find(pair->get_hash()) == cc_to_id.end()) { // filters dupes
            // New collision, so add it to the end of collisions
            cc_to_id.emplace(pair->get_hash(), pair);
            collisions.push_back(pair);
        }
    }

    template <int dim, typename TCollision>
    void add_collision(
        const std::shared_ptr<TCollision>& pair,
        std::vector<std::shared_ptr<SmoothCollision>>& collisions)
    {
        if (pair->is_active()) {
            collisions.push_back(pair);
        }
    }
} // namespace

void HighOrderCollisionsBuilder<2>::add_edge_vertex_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeVertexCandidate>& candidates,
    const SmoothContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, vi] = candidates[i];
		const auto& adj = mesh.vertices_to_edges()[vi];

		const double dhat = std::min(edge_dhat(ei), vert_dhat(vi));
		const PointEdgeDistanceType pe_dtype = point_edge_distance_type(
			vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
			vertices.row(mesh.edges()(ei, 1)));
		const double distance_sqr = point_edge_distance(
			vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
			vertices.row(mesh.edges()(ei, 1)), pe_dtype);
		if (distance_sqr >= dhat * dhat) continue;

		for (int ej : adj) {
			const auto ee_dtype = EdgeEdgeDistanceType::AUTO; // TODO compute this
			add_collision<2, HighOrderCollisionTemplate<Edge2, Edge2>>(
				std::make_shared<HighOrderCollisionTemplate<Edge2, Edge2>>(
					std::min<index_t>(ei, ej), std::max<index_t>(ei, ej),
					ee_dtype, mesh, params,
					dhat, vertices),
				edge_edge_2_to_id, collisions);
		}
    }
}

// ============================================================================

void HighOrderCollisionsBuilder<2>::merge(
    const ParallelCacheType<HighOrderCollisionsBuilder<2>>& local_storage,
    SmoothCollisions& merged_collisions)
{
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Point2, Point2>>>
        vert_vert_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge2, Point2>>>
        vert_edge_2_to_id;

    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // merge
    for (const auto& builder : local_storage) {
        vert_vert_2_to_id.insert(
            builder.vert_vert_2_to_id.begin(), builder.vert_vert_2_to_id.end());
        vert_edge_2_to_id.insert(
            builder.vert_edge_2_to_id.begin(), builder.vert_edge_2_to_id.end());
    }
    int edge_vert_count = vert_edge_2_to_id.size();
    int vert_vert_count = vert_vert_2_to_id.size();

    for (const auto& [key, val] : vert_vert_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : vert_edge_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }

    logger().trace(
        "edge-vert pairs {}, vert-vert pairs {}", edge_vert_count,
        vert_vert_count);
}

} // namespace ipc
