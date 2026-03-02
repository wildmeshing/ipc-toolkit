#pragma once
#include "high_order_collision.hpp"
#include <ipc/utils/unordered_map_and_set.hpp>

namespace ipc {

enum class PointType : std::uint8_t
{
    VERTEX,
    EDGE,
    FACE
};

/// @brief A collection of collision pairs, they can be (Vert, Vert), (Vert, Edge), or (Vert, Face)
/// The first entry of the pairs is always "Vert", which could be actually:
///     1. A real vertex
///     2. A point on an edge, as the closest point between a pair of edges
///     3. A point at the face center
/// In 2 and 3, the "Vert" is a virtual vertex that does not exist in the CollisionMesh, the ID of a
/// virtual vertex is always #n_verts, i.e. immediately after all real vertices.
template <PointType pType> class HighOrderCollisionDict
{
public:
    static constexpr int dim = 3;

    HighOrderCollisionDict() = default;
    ~HighOrderCollisionDict() = default;

    void initialize(
        const std::vector<index_t>& primary_vertex_ids,
        const unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>& map
        );

    const std::array<index_t, 4>& primary_vertex_ids() const { return m_primary_vertex_ids; }
    int size() const { return vv_collisions.size() + ev_collisions.size() + fv_collisions.size(); }

    HighOrderCollision& operator[](int i);
    const HighOrderCollision& operator[](int i) const;

    /* These functions are only available after calling finish_insertion() */

    // Global indices of DoFs
    std::vector<index_t> dofs() const;
    // Global indices of vertices
    const std::vector<index_t>& vertex_ids() const;
    // Map from global vertex index to local vertex index
    index_t vertex_ids_inverse(index_t id) const;

private:
    std::vector<HighOrderCollisionTemplate<Vertex3, Vertex3>> vv_collisions;
    std::vector<HighOrderCollisionTemplate<Edge3P1, Vertex3>> ev_collisions;
    std::vector<HighOrderCollisionTemplate<Face3P1, Vertex3>> fv_collisions;

    /// @brief Primary vertices used to compute the virtual vertex
    /// When the quadrature point q is
    ///     - a vertex, this is that vertex id
    ///     - an edge point, this is the four vertices of the edge-edge pair
    ///     - a face point, this is the three vertices of the face
    /// When the size is smaller than 4, append -1 to entries not used.
    std::array<index_t, 4> m_primary_vertex_ids{{-1,-1,-1,-1}};

    /// @brief Collection of all vertices in collision pairs, including the primary vertices, but not the virtual vertex
    std::vector<index_t> m_vertex_ids;
    /// @brief Inverse of m_vertex_ids
    std::map<index_t, index_t> m_vertex_ids_inverse;
};
} // namespace ipc