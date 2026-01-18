#pragma once

#include "high_order_primitives.hpp"
#include <ipc/high_order_contact/high_order_contact_parameters.hpp>
#include <ipc/utils/math.hpp>
#include <ipc/utils/autodiff_types.hpp>

#include "pair_distance.hpp"

namespace ipc {

class TriplePairCollision
{
public:
    static constexpr int MAX_VERT_3D = 3 * 3;
    static constexpr int ELEMENT_SIZE = 3 * MAX_VERT_3D;

    TriplePairCollision(
        const index_t _primitive0,
        const index_t _primitive1,
        const index_t _primitive2,
        const double _dhat,
        const CollisionMesh& mesh)
        : primitive0(_primitive0)
        , primitive1(_primitive1)
        , primitive2(_primitive2)
        , m_dhat(_dhat)
        {}

    virtual ~TriplePairCollision() = default;

    bool is_active() const { return m_is_active; }
    double dhat() const { return m_dhat; }
    std::vector<index_t> vertex_ids() const { return m_vertex_ids; }
    Eigen::MatrixXd vertices(Eigen::ConstRef<Eigen::MatrixXd> vertices) const;
    Eigen::VectorXd dof(Eigen::ConstRef<Eigen::MatrixXd> X) const;

    virtual std::array<index_t, 4> get_typed_hash() const = 0;

    bool operator==(const TriplePairCollision& other) const
    {
        return (primitive0 == other.primitive0 && primitive1 == other.primitive1 && primitive2 == other.primitive2);
    }

    bool operator!=(const TriplePairCollision& other) const
    {
        return !(*this == other);
    }

    index_t operator[](int idx) const
    {
        if (idx == 0) {
            return primitive0;
        } else if (idx == 1) {
            return primitive1;
        } else if (idx == 2) {
            return primitive2;
        } else {
            throw std::runtime_error("Invalid index in high order collision!");
        }
    }

    std::array<index_t, 3> get_hash() const
    {
        return {{primitive0, primitive1, primitive2}};
    }

    // pure virtual functions

    virtual std::string name() const = 0;
    virtual int n_dofs() const = 0;
    virtual int num_vertices() const = 0;

    virtual double compute_distance(Eigen::ConstRef<Eigen::MatrixXd> positions) const = 0;

    /// @brief Compute the value of the GCP potential
    virtual double operator()(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const = 0;

    /// @brief Compute the gradient of the GCP potential wrt. vertices involved
    virtual Vector<double, -1, ELEMENT_SIZE> gradient(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const = 0;

    /// @brief Compute the Hessian of the GCP potential wrt. vertices involved
    virtual MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE> hessian(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const = 0;

public:
    double weight = 1;

protected:
    bool m_is_active = true;
    index_t primitive0, primitive1, primitive2;
    double m_dhat;
    std::vector<index_t> m_vertex_ids;

    Eigen::VectorXd m_positions_init;
};

template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
class TriplePairCollisionTemplate : public TriplePairCollision {
public:
    using Super = TriplePairCollision;
    static constexpr int N_POINTS =
        PrimitiveA::N_POINTS + PrimitiveB::N_POINTS + PrimitiveC::N_POINTS;
    static constexpr int DIM = PrimitiveA::DIM;
    static constexpr int N_DOFS = N_POINTS * DIM;
    static constexpr int ELEMENT_SIZE = Super::ELEMENT_SIZE;

    TriplePairCollisionTemplate(
        index_t primitive0,
        index_t primitive1,
        index_t primitive2,
        const CollisionMesh& mesh,
        const HighOrderContactParameters& params,
        const double dhat,
        const Eigen::MatrixXd& V);

    ~TriplePairCollisionTemplate() = default;

    std::string name() const override;
    int n_dofs() const override { return N_DOFS; }
    int num_vertices() const override { return N_POINTS; }

    static index_t get_type_as_int();

    // include type as part of the hash so that we can put different types into the same hash table
    std::array<index_t, 4> get_typed_hash() const override
    {
        return {{get_type_as_int(), primitive0, primitive1, primitive2}};
    }

    typename PairDistType<Vertex3, PrimitiveC>::type distance_type_2() const { return dtype2; }

    template <typename T>
    T evaluate(Eigen::ConstRef<Vector<T, N_DOFS>> positions,
        const HighOrderContactParameters& params) const;

    /// @brief Compute the value of the potential
    double operator()(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const override;

    /// @brief Compute the gradient of the potential wrt. vertices involved
    Vector<double, -1, ELEMENT_SIZE> gradient(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const override;

    /// @brief Compute the Hessian of the potential wrt. vertices involved
    MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE> hessian(
        Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const override;

    double compute_distance(Eigen::ConstRef<Eigen::MatrixXd> positions) const override;

    /// @brief Compute the closest point pair between primitive A and B, for now only supports edge-edge
    template <typename T>
    Eigen::Matrix<T, DIM, 1> closest_point_pair_a(Eigen::ConstRef<Vector<T, -1, ELEMENT_SIZE>> positions) const;

private:
    PrimitiveA primitive_a;
    PrimitiveB primitive_b;
    PrimitiveC primitive_c;

    typename PairDistType<PrimitiveA, PrimitiveB>::type dtype1;
    typename PairDistType<Vertex3, PrimitiveC>::type dtype2;
};

}
