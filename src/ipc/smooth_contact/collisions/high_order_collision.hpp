#pragma once
#include "smooth_collision.hpp"
#include "../high_order_contact_parameters.hpp"

namespace ipc {

/// @brief Edge-edge collision in 2d
class HighOrderCollision {
public:
    static constexpr int ELEMENT_SIZE = 3 * MAX_VERT_3D; //TODO check this

    static constexpr int N_CORE_DOFS = 8;
    double weight = 1;

    HighOrderCollision(
        index_t primitive0,
        index_t primitive1,
        const CollisionMesh& mesh,
        const HighOrderContactParameters& params,
        const double dhat,
        const Eigen::MatrixXd& V
    );

    ~HighOrderCollision() = default;

    /// @brief Check if this contact pair is active (depending on both orientation and distance)
    bool is_active() const { return m_is_active; }

    /// @brief dhat value for this contact pair
    double dhat() const { return m_dhat; }

    std::string name() const { return "edge-edge"; }

    int n_dofs() const { return num_vertices() * 2; }
    CollisionType type() const { return CollisionType::EDGE_EDGE; }

    int num_vertices() const { return 4; }

    /// @brief Get the vertex IDs of the collision stencil.
    /// @return The vertex IDs of the collision stencil. Size is always 4, but elements i > num_vertices() are -1.
    std::vector<index_t> vertex_ids() const { return m_vertex_ids; }

    /// @brief Select this stencil's DOF from the full matrix of DOF.
    /// @param X Full matrix of DOF (rowwise).
    /// @return This stencil's DOF.
    Eigen::VectorXd dof(Eigen::ConstRef<Eigen::MatrixXd> X) const;

    bool operator==(const HighOrderCollision& other) const
    {
        return (
            m_primitive0 == other.m_primitive0 && m_primitive1 == other.m_primitive1);
    }

    index_t operator[](int idx) const
    {
        if (idx == 0) {
            return m_primitive0;
        } else if (idx == 1) {
            return m_primitive1;
        } else {
            throw std::runtime_error("Invalid index in high_order_collision!");
        }
    }

    std::pair<index_t, index_t> get_hash() const
    {
        return std::make_pair(m_primitive0, m_primitive1);
    }

    // ---- non distance type potential ----

    /// @brief Compute the GCP potential
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential value
    double operator()(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const;

    /// @brief Compute the potential gradient wrt. positions
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential gradient
    Vector<double, -1, ELEMENT_SIZE> gradient(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const;

    /// @brief Compute the potential Hessian wrt. positions
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential Hessian
    MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE> hessian(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const;

    // ---- distance ----

    /// @brief Compute the minimum squared distance between two primitives
    double compute_distance(Eigen::ConstRef<Eigen::MatrixXd> vertices) const;

protected:
    bool m_is_active = true;
    index_t m_primitive0, m_primitive1;
    double m_dhat;
    std::vector<index_t> m_vertex_ids;
};
} // namespace ipc
