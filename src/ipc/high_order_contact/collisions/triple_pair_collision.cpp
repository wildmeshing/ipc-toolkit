#include "triple_pair_collision.hpp"
#include "pair_distance.hpp"
#include <ipc/smooth_contact/distance/edge_edge.hpp>

namespace ipc
{
    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::TriplePairCollisionTemplate(
        index_t primitive0_,
        index_t primitive1_,
        index_t primitive2_,
        const CollisionMesh& mesh,
        const HighOrderContactParameters& params,
        const double dhat,
        const Eigen::MatrixXd& V)
        : TriplePairCollision(primitive0_, primitive1_, primitive2_, dhat, mesh),
          primitive_a(primitive0_, mesh, V),
          primitive_b(primitive1_, mesh, V),
          primitive_c(primitive2_, mesh, V)
    {
        int i = 0;
        m_vertex_ids.assign(
            primitive_a.n_vertices() + primitive_b.n_vertices() + primitive_c.n_vertices(),
            -1);
        for (auto& v : primitive_a.vertex_ids()) {
            m_vertex_ids[i++] = v;
        }
        for (auto& v : primitive_b.vertex_ids()) {
            m_vertex_ids[i++] = v;
        }
        for (auto& v : primitive_c.vertex_ids()) {
            m_vertex_ids[i++] = v;
        }
        assert(i == primitive_a.n_vertices() + primitive_b.n_vertices() + primitive_c.n_vertices());

        Eigen::VectorXd X = this->dof(V);
        dtype1 = PairDistance<PrimitiveA, PrimitiveB, double>::compute_distance_type(X.head<PairDistance<PrimitiveA, PrimitiveB, double>::N_DOFS>());

        const Eigen::Matrix<double, DIM, 1> closest_point = closest_point_pair_a<double>(X);

        Eigen::Vector<double, Eigen::Dynamic> Y(DIM + PrimitiveC::N_DOFS);
        Y << closest_point, X.template tail<PrimitiveC::N_DOFS>();

        dtype2 = PairDistance<Vertex3, PrimitiveC, double>::compute_distance_type(Y);

        const double dist_sqr_2 = PairDistance<Vertex3, PrimitiveC, double>::compute_distance(Y, dtype2);
        if (dist_sqr_2 >= dhat * dhat) {
            m_is_active = false;
        }

        m_positions_init = std::move(X);
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    template <typename T>
    auto TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::closest_point_pair_a(Eigen::ConstRef<VectorMax<T, TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::ELEMENT_SIZE> > positions) const -> Eigen::Matrix<T, DIM, 1>
    {
        if (m_positions_init.size() > 0 && (m_positions_init - positions).array().abs().maxCoeff() > 0) {
            log_and_throw_error("Inconsistent positions wrt initialization!");
        }
        return positions.template segment<DIM>(0) +
            closest_point_uv<T>(
                positions.template segment<DIM>(0),
                positions.template segment<DIM>(DIM),
                positions.template segment<DIM>(2*DIM),
                positions.template segment<DIM>(3*DIM), dtype1) * (
            positions.template segment<DIM>(DIM) - positions.template segment<DIM>(0));
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    double TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::compute_distance(Eigen::ConstRef<Eigen::MatrixXd> positions) const
    {
        assert(positions.cols() == DIM);
        Eigen::VectorXd X = this->dof(positions);
        const Eigen::Matrix<double, DIM, 1> closest_point = closest_point_pair_a<double>(X);

        static_assert(DIM == 3);

        Eigen::Vector<double, Eigen::Dynamic> Y(DIM + PrimitiveC::N_DOFS);
        Y << closest_point, X.template tail<PrimitiveC::N_DOFS>();

        return PairDistance<Vertex3, PrimitiveC, double>::compute_distance(Y, dtype2);
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    template <typename T>
    T TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::evaluate(
        Eigen::ConstRef<VectorMax<T, N_DOFS>> positions,
        const HighOrderContactParameters& params) const
    {
        const Eigen::Matrix<T, DIM, 1> closest_point = closest_point_pair_a<T>(positions);

        static_assert(DIM == 3);

        Eigen::Vector<T, Eigen::Dynamic> X(DIM + PrimitiveC::N_DOFS);
        X << closest_point, positions.template tail<PrimitiveC::N_DOFS>();

        const T dist_sqr = PairDistance<Vertex3, PrimitiveC, T>::compute_distance(X, dtype2);
        T total = Math<T>::log_barrier(sqrt(dist_sqr) / params.dhat);

        return total;
    }

    Eigen::MatrixXd TriplePairCollision::vertices(Eigen::ConstRef<Eigen::MatrixXd> _vertices) const
    {
        const int dim = _vertices.cols();
        Eigen::MatrixXd stencil_vertices(vertex_ids().size(), dim);
        for (int i = 0; i < vertex_ids().size(); i++) {
            stencil_vertices.row(i) = _vertices.row(vertex_ids()[i]);
        }

        return stencil_vertices;
    }

    Eigen::VectorXd TriplePairCollision::dof(Eigen::ConstRef<Eigen::MatrixXd> X) const
    {
        const int dim = X.cols();
        Eigen::VectorXd x(num_vertices() * dim);
        if (dim == 2) {
            for (int i = 0; i < num_vertices(); i++) {
                x.segment<2>(i * 2) = X.row(m_vertex_ids[i]);
            }
        }
        else if (dim == 3) {
            for (int i = 0; i < num_vertices(); i++) {
                x.segment<3>(i * 3) = X.row(m_vertex_ids[i]);
            }
        }
        else {
            throw std::runtime_error("Invalid dimension!");
        }
        return x;
    }

    template <>
    std::string TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>::name() const { return "eev_3d"; }

    template <>
    std::string TriplePairCollisionTemplate<Edge3P1, Edge3P1, Face3P1>::name() const { return "eef_3d"; }

    template <>
    std::string TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>::name() const { return "eee_3d"; }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    double TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::operator()(
        Eigen::ConstRef<VectorMax<double, TriplePairCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        assert(N_DOFS == positions.size());
        return evaluate<double>(positions.template head<N_DOFS>(), params);
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    VectorMax<double, TriplePairCollision::ELEMENT_SIZE> TriplePairCollisionTemplate<
        PrimitiveA, PrimitiveB, PrimitiveC>::gradient(
        Eigen::ConstRef<VectorMax<double, TriplePairCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        using T = ADGrad<N_DOFS>;
        ScalarBase::setVariableCount(N_DOFS);
        const Eigen::Matrix<T, N_POINTS * DIM, 1> X = slice_positions<T, N_POINTS * DIM, 1>(positions);

        return evaluate<T>(X, params).grad;
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    MatrixMax<double, TriplePairCollision::ELEMENT_SIZE, TriplePairCollision::ELEMENT_SIZE>
    TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::hessian(
        Eigen::ConstRef<VectorMax<double, TriplePairCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        using T = ADHessian<N_DOFS>;
        ScalarBase::setVariableCount(N_DOFS);
        const Eigen::Matrix<T, N_POINTS * DIM, 1> X = slice_positions<T, N_POINTS * DIM, 1>(positions);

        return evaluate<T>(X, params).Hess;
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    index_t TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::get_type_as_int()
    {
        if (std::is_same_v<PrimitiveC, Face3P1>) {
            return 2;
        }
        else if (std::is_same_v<PrimitiveC, Edge3P1>) {
            return 1;
        }
        else if (std::is_same_v<PrimitiveC, Vertex3>) {
            return 0;
        }
        assert(false);
        return -1;
    }

    template class TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>;
    template class TriplePairCollisionTemplate<Edge3P1, Edge3P1, Face3P1>;
    template class TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>;
}
