#pragma once
#include "smooth_collision.hpp"

namespace ipc {

/// @brief Edge-edge collision in 2d
class HighOrderCollision : public SmoothCollision {
public:
    using Super = SmoothCollision;

    HighOrderCollision(
        index_t primitive0,
        index_t primitive1,
        const CollisionMesh& mesh,
        const SmoothContactParameters& params,
        const double dhat,
        const Eigen::MatrixXd& V
    );

    virtual ~HighOrderCollision() = default;

    std::string name() const override { return "edge-edge"; }

    int n_dofs() const override { return num_vertices() * 2; }
    CollisionType type() const override { return CollisionType::EDGE_EDGE; }

    int num_vertices() const override { return 4; }

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
    double compute_distance(Eigen::ConstRef<Eigen::MatrixXd> vertices) const override {return 0;};
};
} // namespace ipc
