#include "high_order_collision_3d.hpp"
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/smooth_contact/distance/point_edge.hpp>

namespace ipc {

template <> HighOrderCollisionType HighOrderCollision3DTemplate<Vertex3, Vertex3>::type() const { return HighOrderCollisionType::VERTEX_VERTEX; }
template <> HighOrderCollisionType HighOrderCollision3DTemplate<Edge3P1, Vertex3>::type() const { return HighOrderCollisionType::EDGE_VERTEX; }
template <> HighOrderCollisionType HighOrderCollision3DTemplate<Face3P1, Vertex3>::type() const { return HighOrderCollisionType::FACE_VERTEX; }

template <> std::string HighOrderCollision3DTemplate<Vertex3, Vertex3>::name() const { return "vv_3d"; }
template <> std::string HighOrderCollision3DTemplate<Edge3P1, Vertex3>::name() const { return "ev_3d"; }
template <> std::string HighOrderCollision3DTemplate<Face3P1, Vertex3>::name() const { return "fv_3d"; }

template <typename PrimitiveA, typename PrimitiveB>
HighOrderCollision3DTemplate<PrimitiveA, PrimitiveB>::HighOrderCollision3DTemplate(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& V)
    :   primitive_a(_primitive0, mesh, V),
        primitive_b(_primitive1, mesh, V)
{
    static_assert(!(std::is_same_v<PrimitiveA, Vertex3> && std::is_same_v<PrimitiveB, Vertex3>));
}

template <>
HighOrderCollision3DTemplate<Vertex3, Vertex3>::HighOrderCollision3DTemplate(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& V)
    :   primitive_a(std::min(_primitive0, _primitive1), mesh, V),
        primitive_b(std::max(_primitive0, _primitive1), mesh, V)
{
}

template <typename PrimitiveA, typename PrimitiveB>
index_t HighOrderCollision3DTemplate<PrimitiveA, PrimitiveB>::vertex_id(index_t i) const
{
    if (i < primitive_a.n_vertices()) {
        return primitive_a.vertex_ids()[i];
    }
    else {
        i -= primitive_a.n_vertices();
        assert(primitive_b.n_vertices() > i);
        return primitive_b.vertex_ids()[i];
    }
}

template<>
double HighOrderCollision3DTemplate<Vertex3, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1)) {
        return point_point_distance(
            vertices.row(vertex_id(0)), vertices.row(vertex_id(n_vertices_a())));
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

template<>
double HighOrderCollision3DTemplate<Edge3P1, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1) && n_verts > vertex_id(2)) {
        return point_edge_distance(
            vertices.row(vertex_id(n_vertices_a())), vertices.row(vertex_id(0)),
            vertices.row(vertex_id(1)));
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

template<>
double HighOrderCollision3DTemplate<Edge3P1, Edge3P1>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1) && n_verts > vertex_id(2) && n_verts > vertex_id(3)) {
        const auto& ea0 = vertices.row(vertex_id(0));
        const auto& ea1 = vertices.row(vertex_id(1));
        const auto& eb0 = vertices.row(vertex_id(2));
        const auto& eb1 = vertices.row(vertex_id(3));
        return edge_edge_distance(ea0, ea1, eb0, eb1);
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

template<>
double HighOrderCollision3DTemplate<Face3P1, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1) && n_verts > vertex_id(2) && n_verts > vertex_id(3)) {
        const auto& f0 = vertices.row(vertex_id(0));
        const auto& f1 = vertices.row(vertex_id(1));
        const auto& f2 = vertices.row(vertex_id(2));
        const auto& v = vertices.row(vertex_id(3));
        return point_triangle_distance(v, f0, f1, f2);
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

// ----------------------------------------------------

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollision3DTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    return 0;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollision3DTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollision3DTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(
        n_dofs(), n_dofs());
}

// ---- distance ----

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollision3DTemplate<PrimitiveA, PrimitiveB>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    // This generic implementation is not used.
    // Specializations will provide their own implementation.
    log_and_throw_error("Not implemented");
    return 0;
}

// ----------------------------------------------------

template <>
double HighOrderCollision3DTemplate<Vertex3, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = (positions.template head<3>() - positions.template segment<3>(3)).norm();
    return Math<double>::log_barrier(dist / params.dhat);
}

template <>
double HighOrderCollision3DTemplate<Edge3P1, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = sqrt(point_edge_distance(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3)));
    return Math<double>::log_barrier(dist / params.dhat);
}

template <>
double HighOrderCollision3DTemplate<Face3P1, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = sqrt(point_triangle_distance(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6)));
    return Math<double>::log_barrier(dist / params.dhat);
}

template <>
auto HighOrderCollision3DTemplate<Vertex3, Vertex3>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    assert(positions.size() == 6);
    const double dist = (positions.template head<3>() - positions.template tail<3>()).norm();
    double deriv = Math<double>::log_barrier_grad(dist / params.dhat);
    deriv *= 1. / params.dhat / dist / 2.;

    Vector6d grad = deriv * point_point_distance_gradient(positions.template head<3>(), positions.template tail<3>());

    return grad;
}

template <>
auto HighOrderCollision3DTemplate<Edge3P1, Vertex3>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    assert(positions.size() == 9);

    auto dtype = point_edge_distance_type(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3));

    const double dist = sqrt(point_edge_distance(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3), dtype));

    double deriv = Math<double>::log_barrier_grad(dist / params.dhat);
    deriv *= 1. / params.dhat / dist / 2.;

    Vector9d grad = point_edge_distance_gradient(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3), dtype);
    grad *= deriv;

    grad = grad({3,4,5,6,7,8,0,1,2}).eval();

    return grad;
}

template <>
auto HighOrderCollision3DTemplate<Face3P1, Vertex3>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    assert(positions.size() == 12);

    auto dtype = point_triangle_distance_type(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6));

    const double dist = sqrt(point_triangle_distance(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6), dtype));

    double deriv = Math<double>::log_barrier_grad(dist / params.dhat);
    deriv *= 1. / params.dhat / dist / 2.;

    Vector12d grad = point_triangle_distance_gradient(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6), dtype);
    grad *= deriv;

    grad = grad({3,4,5,6,7,8,9,10,11,0,1,2}).eval();

    return grad;
}

template <>
auto HighOrderCollision3DTemplate<Vertex3, Vertex3>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    assert(positions.size() == 6);
    const double dist = (positions.template head<3>() - positions.template tail<3>()).norm();
    double deriv1 = Math<double>::log_barrier_grad(dist / params.dhat);
    double deriv2 = Math<double>::log_barrier_hess(dist / params.dhat);
    deriv2 = deriv2 * (1. / params.dhat / params.dhat / 4 / dist / dist) - deriv1 * (1. / params.dhat / 4 / dist / dist / dist);
    deriv1 *= 1. / params.dhat / dist / 2.;

    const Vector6d g = point_point_distance_gradient(positions.template head<3>(), positions.template tail<3>());
    const Matrix6d h = point_point_distance_hessian(positions.template head<3>(), positions.template tail<3>());

    return g * deriv2 * g.transpose() + h * deriv1;
}

template <>
auto HighOrderCollision3DTemplate<Edge3P1, Vertex3>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    assert(positions.size() == 9);

    auto dtype = point_edge_distance_type(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3));

    const double dist = sqrt(point_edge_distance(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3), dtype));

    double deriv1 = Math<double>::log_barrier_grad(dist / params.dhat);
    double deriv2 = Math<double>::log_barrier_hess(dist / params.dhat);
    deriv2 = deriv2 * (1. / params.dhat / params.dhat / 4 / dist / dist) - deriv1 * (1. / params.dhat / 4 / dist / dist / dist);
    deriv1 *= 1. / params.dhat / dist / 2.;

    const Vector9d g = point_edge_distance_gradient(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3), dtype);
    const Matrix9d h = point_edge_distance_hessian(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3), dtype);

    Matrix9d hess = g * deriv2 * g.transpose() + h * deriv1;

    std::vector<int> reorder{3,4,5,6,7,8,0,1,2};

    return hess(reorder, reorder);
}

template <>
auto HighOrderCollision3DTemplate<Face3P1, Vertex3>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    assert(positions.size() == 12);

    auto dtype = point_triangle_distance_type(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6));

    const double dist = sqrt(point_triangle_distance(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6), dtype));

    double deriv1 = Math<double>::log_barrier_grad(dist / params.dhat);
    double deriv2 = Math<double>::log_barrier_hess(dist / params.dhat);
    deriv2 = deriv2 * (1. / params.dhat / params.dhat / 4 / dist / dist) - deriv1 * (1. / params.dhat / 4 / dist / dist / dist);
    deriv1 *= 1. / params.dhat / dist / 2.;

    const Vector12d g = point_triangle_distance_gradient(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6), dtype);
    const Matrix12d h = point_triangle_distance_hessian(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6), dtype);

    Matrix12d hess = g * deriv2 * g.transpose() + h * deriv1;

    std::vector<int> reorder{3,4,5,6,7,8,9,10,11,0,1,2};

    return hess(reorder, reorder);
}

template class HighOrderCollision3DTemplate<Vertex3, Vertex3>;
template class HighOrderCollision3DTemplate<Edge3P1, Vertex3>;
template class HighOrderCollision3DTemplate<Face3P1, Vertex3>;
}