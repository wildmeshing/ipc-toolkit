#pragma once
#include "smooth_collision.hpp"

namespace ipc {

/// @brief Templated class for various types of contact pairs
template <typename PrimitiveA, typename PrimitiveB>
class HighOrderCollisionTemplate : public SmoothCollision {
public:
    using Super = SmoothCollision;
    /// @brief Distance type of the contact pair
    using DTYPE = typename PrimitiveDistType<PrimitiveA, PrimitiveB>::type;
    /// @brief Number of points needed to compute the distance between two primitives
    static constexpr int N_CORE_POINTS =
        PrimitiveA::N_CORE_POINTS + PrimitiveB::N_CORE_POINTS;
    static constexpr int DIM = PrimitiveA::DIM;
    static constexpr int N_CORE_DOFS_A = PrimitiveA::N_CORE_POINTS * DIM;
    static constexpr int N_CORE_DOFS_B = PrimitiveB::N_CORE_POINTS * DIM;
    static constexpr int N_CORE_DOFS = N_CORE_POINTS * DIM;
    static constexpr int ELEMENT_SIZE = Super::ELEMENT_SIZE;

    HighOrderCollisionTemplate(
        index_t primitive0,
        index_t primitive1,
        DTYPE dtype,
        const CollisionMesh& mesh,
        const SmoothContactParameters& params,
        const double dhat,
        const Eigen::MatrixXd& V);

    virtual ~HighOrderCollisionTemplate() = default;

    std::string name() const override;

    int n_dofs() const override
    {
        return primitive_a->n_dofs() + primitive_b->n_dofs();
    }
    CollisionType type() const override;

    Vector<int, N_CORE_DOFS> get_core_indices() const;
    std::array<index_t, N_CORE_DOFS> core_vertex_ids() const;

    int num_vertices() const override
    {
        return primitive_a->n_vertices() + primitive_b->n_vertices();
    }

    template <typename T>
    Vector<T, N_CORE_DOFS> core_dof(const Eigen::MatrixX<T>& X) const
    {
        return this->dof(X)(get_core_indices());
    }

    // ---- non distance type potential ----

    /// @brief Compute the GCP potential
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential value
    double operator()(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const SmoothContactParameters& params) const override;

    /// @brief Compute the potential gradient wrt. positions
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential gradient
    Vector<double, -1, ELEMENT_SIZE> gradient(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const SmoothContactParameters& params) const override;

    /// @brief Compute the potential Hessian wrt. positions
    /// @param positions Vertex positions
    /// @param params GCP parameters
    /// @return GCP potential Hessian
    MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE> hessian(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const SmoothContactParameters& params) const override;

    // ---- distance ----

    /// @brief Compute the minimum squared distance between two primitives
    double
    compute_distance(Eigen::ConstRef<Eigen::MatrixXd> vertices) const override;
private:
    /// @brief The first primitive in the contact pair
    std::unique_ptr<PrimitiveA> primitive_a;
    /// @brief The second primitive in the contact pair
    std::unique_ptr<PrimitiveB> primitive_b;
};
} // namespace ipc
