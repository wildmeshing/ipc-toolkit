#pragma once

#include <ipc/collision_mesh.hpp>
#include <ipc/smooth_contact/high_order_contact_parameters.hpp>
#include <ipc/utils/eigen_ext.hpp>

namespace ipc {

/**
 * @brief Base class for primitives used in high-order contact models.
 *
 * This class defines the common interface for geometric primitives (like
 * vertices and edges) involved in a high-order contact. Derived classes
 * are responsible for implementing the specific logic for their geometry type.
 */
class HighOrderPrimitive {
public:
    HighOrderPrimitive(const index_t id)
        : m_id(id)
    {
    }

    virtual ~HighOrderPrimitive() = default;

    bool operator==(const HighOrderPrimitive& other) const
    {
        return id() == other.id();
    }

    /// @brief Get the ID of this primitive (e.g., vertex ID, edge ID).
    index_t id() const { return m_id; }

    /// @brief Get the number of vertices in the primitive's stencil.
    virtual int n_vertices() const = 0;

    /// @brief Get the number of degrees of freedom for this primitive.
    virtual int n_dofs() const = 0;

    /// @brief Get the vertex IDs of the primitive's stencil.
    const std::vector<index_t>& vertex_ids() const { return m_vertex_ids; }

protected:
    /// @brief Vertex IDs of the stencil for this primitive.
    std::vector<index_t> m_vertex_ids;
    /// @brief The ID of this primitive.
    index_t m_id;
};

namespace {
    // Helper function to find the vertices adjacent to a given vertex in a 2D mesh.
    std::vector<index_t> find_vertex_neighbors_2D(const CollisionMesh& mesh, const index_t v_id)
    {
        std::vector<index_t> neighbors;
        for (const auto& edge_id : mesh.vertex_edge_adjacencies()[v_id]) {
            const auto& edge = mesh.edges().row(edge_id);
            if (edge[0] == v_id) {
                neighbors.push_back(edge[1]);
            } else {
                neighbors.push_back(edge[0]);
            }
        }
        return neighbors;
    }
}

class Vertex2 : public HighOrderPrimitive {
public:
    static constexpr int N_CORE_POINTS = 1;
    static constexpr int DIM = 2;

    Vertex2(
        const index_t id,
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& V)
        : HighOrderPrimitive(id)
    {
        m_vertex_ids.push_back(id);
        std::vector<index_t> neighbors = find_vertex_neighbors_2D(mesh, id);
        for (const auto& neighbor_id : neighbors) {
            m_vertex_ids.push_back(neighbor_id);
        }
    }

    int n_vertices() const override { return m_vertex_ids.size(); }
    int n_dofs() const override { return n_vertices() * DIM; }
};

class Edge2P1 : public HighOrderPrimitive {
public:
    static constexpr int N_CORE_POINTS = 2;
    static constexpr int DIM = 2;

    Edge2P1(
        const index_t id,
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& V)
        : HighOrderPrimitive(id)
    {
        m_vertex_ids = { mesh.edges()(id, 0), mesh.edges()(id, 1) };
    }

    int n_vertices() const override { return m_vertex_ids.size(); }
    int n_dofs() const override { return n_vertices() * DIM; }
};

} // namespace ipc