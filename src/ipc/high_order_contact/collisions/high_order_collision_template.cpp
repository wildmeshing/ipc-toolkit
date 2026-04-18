#include "high_order_collision_template.hpp"
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/smooth_contact/distance/point_edge.hpp>
#include <ipc/utils/eigen_ext.hpp>

namespace ipc {

// ---- type ----

template <> HighOrderCollisionType HighOrderCollisionTemplate<Vertex3, Vertex3>::type() const { return HighOrderCollisionType::VERTEX_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Edge3P1, Vertex3>::type() const { return HighOrderCollisionType::EDGE_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Face3P1, Vertex3>::type() const { return HighOrderCollisionType::FACE_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Vertex2, Vertex2>::type() const { return HighOrderCollisionType::VERTEX_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Vertex2, Edge2P1>::type()  const { return HighOrderCollisionType::EDGE_VERTEX; }

// ---- name ----

template <> std::string HighOrderCollisionTemplate<Vertex3, Vertex3>::name() const { return "vv_3d"; }
template <> std::string HighOrderCollisionTemplate<Edge3P1, Vertex3>::name() const { return "ev_3d"; }
template <> std::string HighOrderCollisionTemplate<Face3P1, Vertex3>::name() const { return "fv_3d"; }
template <> std::string HighOrderCollisionTemplate<Vertex2, Vertex2>::name() const { return "vv_2d_pt"; }
template <> std::string HighOrderCollisionTemplate<Vertex2, Edge2P1>::name()  const { return "ev_2d_pt"; }

// ---- constructors ----

template <typename PrimitiveA, typename PrimitiveB>
HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::HighOrderCollisionTemplate(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh)
    : primitive_a(_primitive0, mesh)
    , primitive_b(_primitive1, mesh)
{
    static_assert(Eigen::internal::packet_traits<double>::size == 1,
                "Eigen vectorization is NOT disabled!");
}

template <>
HighOrderCollisionTemplate<Vertex3, Vertex3>::HighOrderCollisionTemplate(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh)
    : primitive_a(std::min(_primitive0, _primitive1), mesh)
    , primitive_b(std::max(_primitive0, _primitive1), mesh)
{
}

// ---- vertex_id ----

template <typename PrimitiveA, typename PrimitiveB>
index_t HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::vertex_id(index_t i) const
{
    if (i < (index_t)primitive_a.n_vertices()) {
        return primitive_a.vertex_ids()[i];
    }
    i -= primitive_a.n_vertices();
    assert((index_t)primitive_b.n_vertices() > i);
    return primitive_b.vertex_ids()[i];
}

// ---- generic stubs ----

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> /*positions*/,
    const HighOrderContactParameters& /*params*/) const
{
    return 0;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> /*positions*/,
    const HighOrderContactParameters& /*params*/) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> /*positions*/,
    const HighOrderContactParameters& /*params*/) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> /*vertices*/) const
{
    log_and_throw_error("Not implemented");
    return 0;
}

// ---- 3D specializations ----

template<>
double HighOrderCollisionTemplate<Vertex3, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1)) {
        return point_point_distance(
            vertices.row(vertex_id(0)), vertices.row(vertex_id(n_vertices_a())));
    }
    return std::numeric_limits<double>::max();
}

template<>
double HighOrderCollisionTemplate<Edge3P1, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1) && n_verts > vertex_id(2)) {
        return point_edge_distance(
            vertices.row(vertex_id(n_vertices_a())), vertices.row(vertex_id(0)),
            vertices.row(vertex_id(1)));
    }
    return std::numeric_limits<double>::max();
}

template<>
double HighOrderCollisionTemplate<Edge3P1, Edge3P1>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1) && n_verts > vertex_id(2) && n_verts > vertex_id(3)) {
        return edge_edge_distance(
            vertices.row(vertex_id(0)), vertices.row(vertex_id(1)),
            vertices.row(vertex_id(2)), vertices.row(vertex_id(3)));
    }
    return std::numeric_limits<double>::max();
}

template<>
double HighOrderCollisionTemplate<Face3P1, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > vertex_id(0) && n_verts > vertex_id(1) && n_verts > vertex_id(2) && n_verts > vertex_id(3)) {
        return point_triangle_distance(
            vertices.row(vertex_id(3)),
            vertices.row(vertex_id(0)), vertices.row(vertex_id(1)), vertices.row(vertex_id(2)));
    }
    return std::numeric_limits<double>::max();
}

template <>
double HighOrderCollisionTemplate<Vertex3, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = (positions.template head<3>() - positions.template segment<3>(3)).norm();
    const double eps = params.get_dhat(safety_mode);
    return (*params.barrier)(dist, eps);
}

template <>
double HighOrderCollisionTemplate<Edge3P1, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = sqrt(point_edge_distance(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3)));
    const double eps = params.get_dhat(safety_mode);
    return (*params.barrier)(dist, eps);
}

template <>
double HighOrderCollisionTemplate<Face3P1, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = sqrt(point_triangle_distance(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6)));
    const double eps = params.get_dhat(safety_mode);
    return (*params.barrier)(dist, eps);
}

template <>
auto HighOrderCollisionTemplate<Vertex3, Vertex3>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    assert(positions.size() == 6);
    const double dist = (positions.template head<3>() - positions.template tail<3>()).norm();
    const double eps = params.get_dhat(safety_mode);
    const double deriv = params.barrier->first_derivative(dist, eps) / (dist * 2.);
    Vector6d grad = deriv * point_point_distance_gradient(positions.template head<3>(), positions.template tail<3>());
    return grad;
}

template <>
auto HighOrderCollisionTemplate<Edge3P1, Vertex3>::gradient(
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
    const double eps = params.get_dhat(safety_mode);
    const double deriv = params.barrier->first_derivative(dist, eps) / (dist * 2.);
    Vector9d grad = point_edge_distance_gradient(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3), dtype);
    grad *= deriv;
    grad = grad({3,4,5,6,7,8,0,1,2}).eval();
    return grad;
}

template <>
auto HighOrderCollisionTemplate<Face3P1, Vertex3>::gradient(
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
    const double eps = params.get_dhat(safety_mode);
    const double deriv = params.barrier->first_derivative(dist, eps) / (dist * 2.);
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
auto HighOrderCollisionTemplate<Vertex3, Vertex3>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    assert(positions.size() == 6);
    const double dist = (positions.template head<3>() - positions.template tail<3>()).norm();
    const double eps = params.get_dhat(safety_mode);
    double deriv1 = params.barrier->first_derivative(dist, eps);
    double deriv2 = params.barrier->second_derivative(dist, eps);
    deriv2 = deriv2 / (4 * dist * dist) - deriv1 / (4 * dist * dist * dist);
    deriv1 /= (2 * dist);
    const Vector6d g = point_point_distance_gradient(positions.template head<3>(), positions.template tail<3>());
    const Matrix6d h = point_point_distance_hessian(positions.template head<3>(), positions.template tail<3>());
    return g * deriv2 * g.transpose() + h * deriv1;
}

template <>
auto HighOrderCollisionTemplate<Edge3P1, Vertex3>::hessian(
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
    const double eps = params.get_dhat(safety_mode);
    double deriv1 = params.barrier->first_derivative(dist, eps);
    double deriv2 = params.barrier->second_derivative(dist, eps);
    deriv2 = deriv2 / (4 * dist * dist) - deriv1 / (4 * dist * dist * dist);
    deriv1 /= (2 * dist);
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
auto HighOrderCollisionTemplate<Face3P1, Vertex3>::hessian(
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
    const double eps = params.get_dhat(safety_mode);
    double deriv1 = params.barrier->first_derivative(dist, eps);
    double deriv2 = params.barrier->second_derivative(dist, eps);
    deriv2 = deriv2 / (4 * dist * dist) - deriv1 / (4 * dist * dist * dist);
    deriv1 /= (2 * dist);
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

// ---- 2D specializations ----
// positions layout VV: [q_x, q_y, v_x, v_y]
// positions layout VE: [q_x, q_y, e0_x, e0_y, e1_x, e1_y]

template <>
double HighOrderCollisionTemplate<Vertex2, Vertex2>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n = vertices.rows();
    if (vertex_id(0) >= n || vertex_id(1) >= n)
        return std::numeric_limits<double>::max();
    return point_point_distance(vertices.row(vertex_id(0)), vertices.row(vertex_id(1)));
}

template <>
double HighOrderCollisionTemplate<Vertex2, Edge2P1>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n = vertices.rows();
    if (vertex_id(0) >= n || vertex_id(1) >= n || vertex_id(2) >= n)
        return std::numeric_limits<double>::max();
    return point_edge_distance(
        vertices.row(vertex_id(0)),
        vertices.row(vertex_id(1)),
        vertices.row(vertex_id(2)));
}

template <>
double HighOrderCollisionTemplate<Vertex2, Vertex2>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = (positions.template head<2>() - positions.template tail<2>()).norm();
    const double eps = params.get_dhat(safety_mode);
    return (*params.barrier)(dist, eps);
}

template <>
double HighOrderCollisionTemplate<Vertex2, Edge2P1>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = std::sqrt(point_edge_distance(
        positions.template head<2>(),
        positions.template segment<2>(2),
        positions.template segment<2>(4)));
    const double eps = params.get_dhat(safety_mode);
    return (*params.barrier)(dist, eps);
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Vertex2>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    const double dist = (positions.template head<2>() - positions.template tail<2>()).norm();
    const double eps = params.get_dhat(safety_mode);
    const double deriv = params.barrier->first_derivative(dist, eps) / (dist * 2.0);
    const VectorMax6d g = point_point_distance_gradient(
        positions.template head<2>(), positions.template tail<2>());
    return deriv * g;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Edge2P1>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    const double dist = std::sqrt(point_edge_distance(
        positions.template head<2>(),
        positions.template segment<2>(2),
        positions.template segment<2>(4)));
    const double eps = params.get_dhat(safety_mode);
    const double deriv = params.barrier->first_derivative(dist, eps) / (dist * 2.0);
    const VectorMax9d g = point_edge_distance_gradient(
        positions.template head<2>(),
        positions.template segment<2>(2),
        positions.template segment<2>(4));
    return deriv * g;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Vertex2>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    const double dist = (positions.template head<2>() - positions.template tail<2>()).norm();
    const double eps = params.get_dhat(safety_mode);
    double deriv1 = params.barrier->first_derivative(dist, eps);
    double deriv2 = params.barrier->second_derivative(dist, eps);
    deriv2 = deriv2 / (4 * dist * dist) - deriv1 / (4 * dist * dist * dist);
    deriv1 /= (2 * dist);
    const VectorMax6d g = point_point_distance_gradient(
        positions.template head<2>(), positions.template tail<2>());
    const MatrixMax6d H = point_point_distance_hessian(
        positions.template head<2>(), positions.template tail<2>());
    return g * deriv2 * g.transpose() + H * deriv1;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Edge2P1>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    const double dist = std::sqrt(point_edge_distance(
        positions.template head<2>(),
        positions.template segment<2>(2),
        positions.template segment<2>(4)));
    const double eps = params.get_dhat(safety_mode);
    double deriv1 = params.barrier->first_derivative(dist, eps);
    double deriv2 = params.barrier->second_derivative(dist, eps);
    deriv2 = deriv2 / (4 * dist * dist) - deriv1 / (4 * dist * dist * dist);
    deriv1 /= (2 * dist);
    const VectorMax9d g = point_edge_distance_gradient(
        positions.template head<2>(),
        positions.template segment<2>(2),
        positions.template segment<2>(4));
    const MatrixMax9d H = point_edge_distance_hessian(
        positions.template head<2>(),
        positions.template segment<2>(2),
        positions.template segment<2>(4));
    return g * deriv2 * g.transpose() + H * deriv1;
}

// ---- explicit instantiations ----

template class HighOrderCollisionTemplate<Vertex3, Vertex3>;
template class HighOrderCollisionTemplate<Edge3P1, Vertex3>;
template class HighOrderCollisionTemplate<Face3P1, Vertex3>;
template class HighOrderCollisionTemplate<Vertex2, Vertex2>;
template class HighOrderCollisionTemplate<Vertex2, Edge2P1>;

} // namespace ipc
