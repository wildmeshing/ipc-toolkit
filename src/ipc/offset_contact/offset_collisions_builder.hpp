#pragma once

#include <ipc/offset_contact/offset_collisions.hpp>

#include <ipc/collision_mesh.hpp>
#include <ipc/utils/maybe_parallel_for.hpp>

#include <Eigen/Core>

namespace ipc {

template <int dim> class OffsetCollisionsBuilder;

template <> class OffsetCollisionsBuilder<2> {
public:
    OffsetCollisionsBuilder() { }

    void add_edge_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeVertexCandidate>& candidates,
        const OffsetContactParameters& params,
        const std::function<double(const index_t)>& vert_dhat,
        const std::function<double(const index_t)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);

    // -------------------------------------------------------------------------

    static void merge(
        const ParallelCacheType<OffsetCollisionsBuilder<2>>& local_storage,
        OffsetCollisions& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<OffsetCollision>> collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<OffsetCollisionTemplate<Vertex2, Vertex2>>>
        vert_vert_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<OffsetCollisionTemplate<Edge2P1, Vertex2>>>
        vert_edge_2_to_id;
};

} // namespace ipc
