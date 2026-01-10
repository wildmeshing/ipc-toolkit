#include "triple_pair_collision.hpp"
#include "pair_distance.hpp"

namespace ipc
{
    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::TriplePairCollisionTemplate(
        index_t primitive0,
        index_t primitive1,
        index_t primitive2,
        const CollisionMesh& mesh,
        const HighOrderContactParameters& params,
        const double dhat,
        const Eigen::MatrixXd& V)
        : TriplePairCollision(primitive0, primitive1, primitive2, dhat, mesh),
          primitive_a(primitive0, mesh, V),
          primitive_b(primitive1, mesh, V),
          primitive_c(primitive2, mesh, V)
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
        const double dist_sqr = PairDistance<PrimitiveA, PrimitiveB, double>::compute_distance(
            X.head<PairDistance<PrimitiveA, PrimitiveB, double>::N_DOFS>(), dtype1);
        if (dist_sqr >= dhat * dhat) {
            m_is_active = false;
        }
        else {
            const Eigen::Matrix<double, DIM, 2> closest_points = closest_point_pair_ab<double>(X);

            Eigen::Vector<double, Eigen::Dynamic> Y(DIM + PrimitiveC::N_DOFS);
            Y << closest_points.col(0), X.template tail<PrimitiveC::N_DOFS>();
            dtype2 = PairDistance<Vertex3, PrimitiveC, double>::compute_distance_type(Y);
            const double dist_sqr = PairDistance<Vertex3, PrimitiveC, double>::compute_distance(Y, dtype2);
            if (dist_sqr >= dhat * dhat) {
                m_is_active = false;
            }
        }
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    template <typename T>
    Eigen::Matrix<T, TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::DIM, 2>
    TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::closest_point_pair_ab(
        Eigen::ConstRef<Vector<T, -1, ELEMENT_SIZE>> positions) const
    {
        static_assert(std::is_same_v<PrimitiveA, Edge3P1>);
        static_assert(std::is_same_v<PrimitiveB, Edge3P1>);

        assert(dtype1 == EdgeEdgeDistanceType::EA_EB);
        return line_line_closest_point_pairs<T>(
            positions.template segment<DIM>(0),
            positions.template segment<DIM>(DIM),
            positions.template segment<DIM>(2 * DIM),
            positions.template segment<DIM>(3 * DIM));
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    template <typename T>
    T TriplePairCollisionTemplate<PrimitiveA, PrimitiveB, PrimitiveC>::evaluate(
        Eigen::ConstRef<Vector<T, N_DOFS>> positions,
        const HighOrderContactParameters& params) const
    {
        const Eigen::Matrix<T, DIM, 2> closest_points = closest_point_pair_ab<T>(positions);

        static_assert(DIM == 3);
        T total(0.);
        const int i = 0;
        {
            Eigen::Vector<T, Eigen::Dynamic> X(DIM + PrimitiveC::N_DOFS);
            X << closest_points.col(i), positions.template tail<PrimitiveC::N_DOFS>();
            const T dist_sqr = PairDistance<Vertex3, PrimitiveC, T>::compute_distance(X, dtype2);
            total += Math<T>::inv_barrier(sqrt(dist_sqr) / params.dhat, params.r);
        }

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
        Eigen::ConstRef<Vector<double, -1, TriplePairCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        assert(N_DOFS == positions.size());
        return evaluate<double>(positions.template head<N_DOFS>(), params);
    }

    template <typename PrimitiveA, typename PrimitiveB, typename PrimitiveC>
    Vector<double, -1, TriplePairCollision::ELEMENT_SIZE> TriplePairCollisionTemplate<
        PrimitiveA, PrimitiveB, PrimitiveC>::gradient(
        Eigen::ConstRef<Vector<double, -1, TriplePairCollision::ELEMENT_SIZE>> positions,
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
        Eigen::ConstRef<Vector<double, -1, TriplePairCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        using T = ADHessian<N_DOFS>;
        ScalarBase::setVariableCount(N_DOFS);
        const Eigen::Matrix<T, N_POINTS * DIM, 1> X = slice_positions<T, N_POINTS * DIM, 1>(positions);

        return evaluate<T>(X, params).Hess;
    }

    template class TriplePairCollisionTemplate<Edge3P1, Edge3P1, Vertex3>;
    template class TriplePairCollisionTemplate<Edge3P1, Edge3P1, Face3P1>;
    template class TriplePairCollisionTemplate<Edge3P1, Edge3P1, Edge3P1>;
}
