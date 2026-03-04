#pragma once
#include "high_order_collision.hpp"

namespace ipc {

/// @brief Templated class for various types of contact pairs
template <typename PrimitiveA, typename PrimitiveB>
class HighOrderCollision3DTemplate : public HighOrderCollision {
public:
    using Super = HighOrderCollision;
    static constexpr int N_CORE_POINTS =
        PrimitiveA::N_CORE_POINTS + PrimitiveB::N_CORE_POINTS;
    static constexpr int DIM = PrimitiveA::DIM;
    static constexpr int N_CORE_DOFS_A = PrimitiveA::N_CORE_POINTS * DIM;
    static constexpr int N_CORE_DOFS_B = PrimitiveB::N_CORE_POINTS * DIM;
    static constexpr int N_CORE_DOFS = N_CORE_POINTS * DIM;
    static constexpr int ELEMENT_SIZE = Super::ELEMENT_SIZE;

    HighOrderCollision3DTemplate(
        index_t primitive0,
        index_t primitive1,
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& V);

    virtual ~HighOrderCollision3DTemplate() = default;

    std::string name() const override;

    int n_dofs() const override
    {
        return primitive_a.n_dofs() + primitive_b.n_dofs();
    }
    HighOrderCollisionType type() const override;

    std::pair<index_t, index_t> get_hash() const override
    {
        return std::make_pair(primitive_a.id(), primitive_b.id());
    }

    std::array<index_t, 3> get_typed_hash() const override
    {
        return {{static_cast<index_t>(type()), primitive_a.id(), primitive_b.id()}};
    }

    index_t operator[](int idx) const override
    {
        if (idx == 0) {
            return primitive_a.id();
        } else if (idx == 1) {
            return primitive_b.id();
        } else {
            throw std::runtime_error("Invalid index in high order collision!");
        }
    }

    int num_vertices() const override
    {
        return primitive_a.n_vertices() + primitive_b.n_vertices();
    }

    index_t vertex_id(index_t i) const override;

    size_t n_vertices_a() const override { return primitive_a.n_vertices(); }
    size_t n_vertices_b() const override { return primitive_b.n_vertices(); }

    // ---- non distance type potential ----

    /// @brief Compute the GCP potential
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential value
    double operator()(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const override;

    /// @brief Compute the potential gradient wrt. positions
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential gradient
    VectorMax<double, ELEMENT_SIZE> gradient(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const override;

    /// @brief Compute the potential Hessian wrt. positions
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential Hessian
    MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE> hessian(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const override;

    // ---- distance ----

    /// @brief Compute the minimum squared distance between two primitives
    double
    compute_distance(Eigen::ConstRef<Eigen::MatrixXd> vertices) const override;

private:
    /// @brief The first primitive in the contact pair
    PrimitiveA primitive_a;
    /// @brief The second primitive in the contact pair
    PrimitiveB primitive_b;
};
}
