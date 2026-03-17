#pragma once

#include <ipc/high_order_contact/high_order_collisions.hpp>

#include <ipc/collision_mesh.hpp>
#include <ipc/utils/maybe_parallel_for.hpp>

#include <Eigen/Core>

#include <memory>
#include "collisions/triple_pair_collision.hpp"

namespace ipc {

template <int dim> class HighOrderCollisionsBuilder;
class PointPotential;
class QuadratureCollisionsBuilder;

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

    void add_edge_edge_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeEdgeCandidate>& candidates,
        const HighOrderContactParameters& params,
        const std::function<double(const index_t)>& vert_dhat,
        const std::function<double(const index_t)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);

    static std::shared_ptr<HighOrderCollision> reduce_edge_edge_collision(
        const index_t ei,
        const index_t ej,
        const double dhat,
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const HighOrderContactParameters& params);

    // -------------------------------------------------------------------------

    static void merge(
        const ParallelCacheType<HighOrderCollisionsBuilder<2>>& local_storage,
        HighOrderCollisions& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<HighOrderCollision>> collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    /*
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Vertex2, Vertex2>>>
        vert_vert_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge2P1, Vertex2>>>
        vert_edge_2_to_id;
    unordered_map<
        std::pair<index_t, index_t>,
        std::shared_ptr<HighOrderCollisionTemplate<Edge2P1, Edge2P1>>>
        edge_edge_2_to_id;
    */

    unordered_map<std::pair<index_t, index_t>, index_t> vert_vert_2_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_edge_2_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> edge_edge_2_to_id;
};

template <> class HighOrderCollisionsBuilder<3> {
public:
    HighOrderCollisionsBuilder() { }

    static std::shared_ptr<HighOrderCollision> reduce_point_triangle_collision(
        const FaceVertexCandidate& candidate,
        const HighOrderContactParameters& params,
        const CollisionMesh& mesh,
        const VertexMatrixView<3>& vertices,
        PointTriangleDistanceType dtype = PointTriangleDistanceType::AUTO);

    static std::shared_ptr<HighOrderCollision> reduce_point_edge_collision(
        const EdgeVertexCandidate& candidate,
        const HighOrderContactParameters& params,
        const CollisionMesh& mesh,
        const VertexMatrixView<3>& vertices,
        PointEdgeDistanceType dtype = PointEdgeDistanceType::AUTO);

    void add_edge_edge_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeEdgeCandidate>& candidates,
        const HighOrderContactParameters& params,
        const std::function<double(const index_t)>& vert_dhat,
        const std::function<double(const index_t)>& edge_dhat,
        const std::function<double(const index_t)>& face_dhat,
        const size_t start_i,
        const size_t end_i);

    void add_face_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<FaceVertexCandidate>& candidates,
        const HighOrderContactParameters& params,
        const size_t start_i,
        const size_t end_i);

    void add_face_vertex_negative_edge_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeVertexCandidate>& candidates,
        const HighOrderContactParameters& params,
        const size_t start_i,
        const size_t end_i);

    void add_face_vertex_positive_vertex_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<VertexVertexCandidate>& candidates,
        const HighOrderContactParameters& params,
        const size_t start_i,
        const size_t end_i);

    /*/ -------------------------------------------------------------------------

    void add_negative_edge_edge_edge_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<std::array<index_t, 3>>& candidates,
        const HighOrderContactParameters& params,
        const double dhat,
        const size_t start_i,
        const size_t end_i);

    void add_edge_edge_face_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<std::array<index_t, 3>>& candidates,
        const HighOrderContactParameters& params,
        const double dhat,
        const size_t start_i,
        const size_t end_i);

    void add_edge_edge_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<std::array<index_t, 3>>& candidates,
        const HighOrderContactParameters& params,
        const double dhat,
        const size_t start_i,
        const size_t end_i);

    /*/// -------------------------------------------------------------------------

    static void merge(
        const ParallelCacheType<HighOrderCollisionsBuilder<3>>& local_storage,
        HighOrderCollisions& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<HighOrderCollision>> collisions;
    std::vector<std::shared_ptr<TriplePairCollision>> triple_collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    unordered_map<std::pair<index_t, index_t>, index_t> vert_vert_3_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_edge_3_to_id;
    unordered_map<std::pair<index_t, index_t>, index_t> vert_face_3_to_id;

    unordered_map<std::array<index_t, 3>, index_t> eev_3_to_id;
    unordered_map<std::array<index_t, 3>, index_t> eef_3_to_id;
    unordered_map<std::array<index_t, 3>, index_t> eee_3_to_id;
};

class QuadratureCollisionsBuilder {
public:
    QuadratureCollisionsBuilder(
        const CollisionMesh& mesh,
        const Candidates& candidates,
        const HighOrderContactParameters& params);
    QuadratureCollisionsBuilder(QuadratureCollisionsBuilder&&) = default;
    QuadratureCollisionsBuilder& operator=(QuadratureCollisionsBuilder&&) = default;
    QuadratureCollisionsBuilder(const QuadratureCollisionsBuilder& other);
    QuadratureCollisionsBuilder& operator=(const QuadratureCollisionsBuilder& other);
    ~QuadratureCollisionsBuilder();

    void build_vertex_collisions(
        const Eigen::MatrixXd& vertices,
        const std::vector<index_t>& vertex_indices,
        size_t start, size_t end);

    void build_face_collisions(
        const Eigen::MatrixXd& vertices,
        const std::vector<index_t>& face_indices,
        size_t start, size_t end);

    void build_edge_edge_collisions(
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeEdgeCandidate>& ee_candidates,
        const size_t start_i,
        const size_t end_i);

    static void merge(
        ParallelCacheType<QuadratureCollisionsBuilder>& local_storage,
        HighOrderCollisions& merged_collisions);

    // Local storage
    std::vector<std::unique_ptr<HighOrderCollisionDict<PointType::VERTEX>>> vertex_collisions;
    std::vector<std::unique_ptr<HighOrderCollisionDict<PointType::EDGE>>> edge_edge_collisions;
    std::vector<std::unique_ptr<HighOrderCollisionDict<PointType::FACE>>> face_collisions;

    size_t num_collision_pairs = 0;

    std::shared_ptr<PointPotential> point_potential;
};
} // namespace ipc
