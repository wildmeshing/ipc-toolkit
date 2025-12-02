#pragma once

#include "high_order_collisions.hpp"

#include <ipc/collision_mesh.hpp>
#include <ipc/utils/maybe_parallel_for.hpp>

#include <Eigen/Core>

namespace ipc {

template <int dim> class HighOrderCollisionsBuilder;

template <> class HighOrderCollisionsBuilder<2> {
public:
    HighOrderCollisionsBuilder() { }

    void add_edge_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeVertexCandidate>& candidates,
        const HighOrderContactParameters& params,
        const std::function<double(const index_t)>& vert_dhat,
        const std::function<double(const index_t)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);

	/*
    void add_edge_edge_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeEdgeCandidate>& candidates,
        const HighOrderContactParameters& params,
        const std::function<double(const index_t)>& vert_dhat,
        const std::function<double(const index_t)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);
	*/

    // -------------------------------------------------------------------------

    static void merge(
        const ParallelCacheType<HighOrderCollisionsBuilder<2>>& local_storage,
        HighOrderCollisions& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<HighOrderCollision>> collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollision>>
        edge_edge_2_to_id;
};

} // namespace ipc
