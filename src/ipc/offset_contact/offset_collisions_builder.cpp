#include "offset_collisions_builder.hpp"

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
        std::vector<std::shared_ptr<OffsetCollision>>& collisions)
    {
        if (pair->is_active()
            && cc_to_id.find(pair->get_hash()) == cc_to_id.end()) { // filters dupes
            // New collision, so add it to the end of collisions
            cc_to_id.emplace(pair->get_hash(), pair);
            collisions.push_back(pair);
        }
    }
} // namespace

void OffsetCollisionsBuilder<2>::add_edge_vertex_collisions(
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const std::vector<EdgeVertexCandidate>& candidates,
    const OffsetContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, vi] = candidates[i];
        const double dhat_EV = std::min(vert_dhat(vi), edge_dhat(ei));
		const PointEdgeDistanceType pe_dtype = point_edge_distance_type(
			vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
			vertices.row(mesh.edges()(ei, 1)));

        const double distance_sqr = point_edge_distance(
            vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)), pe_dtype);
        assert(distance_sqr >= 0);
        if (distance_sqr < dhat_EV * dhat_EV) {
            add_collision(
                std::make_shared<OffsetCollisionTemplate<ogcEdge2, ogcVert2>>(
                    ei, vi, mesh, params, dhat_EV, vertices),
                vert_edge_2_to_id, collisions);
        }

		// vertex-vertex
		for (int j = 0; j < 2; j++) {
			const index_t vj = mesh.edges()(ei, j);
            const double dhat_VV = std::min(vert_dhat(vi), vert_dhat(vj));
			if ((vertices.row(vi) - vertices.row(vj)).norm() < dhat_VV) {
				add_collision(
					std::make_shared<OffsetCollisionTemplate<ogcVert2, ogcVert2>>(
						std::min(vi, vj), std::max(vi, vj), mesh, params,
						dhat_VV, vertices),
					vert_vert_2_to_id, collisions);
			}
		}
    }
}

// ============================================================================

void OffsetCollisionsBuilder<2>::merge(
    const ParallelCacheType<OffsetCollisionsBuilder<2>>& local_storage,
    OffsetCollisions& merged_collisions)
{
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<OffsetCollisionTemplate<ogcVert2, ogcVert2>>>
        vert_vert_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<OffsetCollisionTemplate<ogcEdge2, ogcVert2>>>
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
    int vert_vert_count = vert_vert_2_to_id.size();
    int vert_edge_count = vert_edge_2_to_id.size();

    for (const auto& [key, val] : vert_vert_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }
    for (const auto& [key, val] : vert_edge_2_to_id) {
        merged_collisions.collisions.push_back(val);
    }

    logger().trace(
        "VV pairs: {}; VE pairs: {}.",
        vert_vert_count, vert_edge_count);
}

} // namespace ipc
