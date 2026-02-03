#include "high_order_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#include "alternating_potential_2D.hpp"
#include "ipc/smooth_contact/distance/point_edge.hpp"

namespace ipc {

namespace acp = alternating_contact_potential;

// clang-format off
template <> HighOrderCollisionType HighOrderCollisionTemplate<Vertex2, Vertex2>::type() const { return HighOrderCollisionType::VERTEX_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Edge2P1, Vertex2>::type() const { return HighOrderCollisionType::EDGE_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Edge2P1, Edge2P1>::type() const { return HighOrderCollisionType::EDGE_EDGE; }

template <> HighOrderCollisionType HighOrderCollisionTemplate<Vertex3, Vertex3>::type() const { return HighOrderCollisionType::VERTEX_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Edge3P1, Vertex3>::type() const { return HighOrderCollisionType::EDGE_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Face3P1, Vertex3>::type() const { return HighOrderCollisionType::FACE_VERTEX; }
// clang-format on

// clang-format off
template <> std::string HighOrderCollisionTemplate<Vertex2, Vertex2>::name() const { return "vv_2d"; }
template <> std::string HighOrderCollisionTemplate<Edge2P1, Vertex2>::name() const { return "ve_2d"; }
template <> std::string HighOrderCollisionTemplate<Edge2P1, Edge2P1>::name() const { return "ee_2d"; }

template <> std::string HighOrderCollisionTemplate<Vertex3, Vertex3>::name() const { return "vv_3d"; }
template <> std::string HighOrderCollisionTemplate<Edge3P1, Vertex3>::name() const { return "ev_3d"; }
template <> std::string HighOrderCollisionTemplate<Face3P1, Vertex3>::name() const { return "fv_3d"; }
// clang-format on

Eigen::VectorXd HighOrderCollision::dof(Eigen::ConstRef<Eigen::MatrixXd> X) const
{
    const int DIM = X.cols();
    Eigen::VectorXd x(num_vertices() * DIM);
    if (DIM == 2) {
        for (int i = 0; i < num_vertices(); i++) {
            x.segment<2>(i * 2) = X.row(m_vertex_ids[i]);
        }
    } else if (DIM == 3) {
        for (int i = 0; i < num_vertices(); i++) {
            assert(m_vertex_ids[i] < X.rows());
            x.segment<3>(i * 3) = X.row(m_vertex_ids[i]);
        }
    } else {
        throw std::runtime_error("Invalid dimension!");
    }
    return x;
}

Eigen::VectorXd HighOrderCollision::dof(ConcatMatrixView<3> X_extended) const
{
    Eigen::VectorXd x(num_vertices() * 3);
    for (int i = 0; i < num_vertices(); i++) {
        assert(m_vertex_ids[i] < X_extended.rows());
        x.segment<3>(i * 3) = X_extended(m_vertex_ids[i]);
    }
    return x;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::get_core_indices() const
    -> Eigen::Vector<int, N_CORE_DOFS>
{
    Eigen::Vector<int, N_CORE_DOFS> core_indices;
    core_indices << Eigen::VectorXi::LinSpaced(
        N_CORE_DOFS_A, 0, N_CORE_DOFS_A - 1),
        Eigen::VectorXi::LinSpaced(
            N_CORE_DOFS_B, primitive_a->n_dofs(),
            primitive_a->n_dofs() + N_CORE_DOFS_B - 1);
    return core_indices;
}

template <typename PrimitiveA, typename PrimitiveB>
HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::HighOrderCollisionTemplate(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh,
    const HighOrderContactParameters& params,
    const double _dhat,
    const Eigen::MatrixXd& V)
    : HighOrderCollision(_primitive0, _primitive1, _dhat, mesh)
{
    primitive_a = std::make_unique<PrimitiveA>(_primitive0, mesh, V);
    primitive_b = std::make_unique<PrimitiveB>(_primitive1, mesh, V);

    if constexpr (std::is_same_v<PrimitiveA, Edge2P1>) {
        m_area_a = mesh.edge_length(_primitive0);
    }

    if constexpr (std::is_same_v<PrimitiveB, Edge2P1>) {
        m_area_b = mesh.edge_length(_primitive1);
    }

    if constexpr (DIM == 2) {
        auto is_obstacle = [&](const auto& primitive) {
            bool any_obstacle = false;
            bool all_obstacle = true;
            for (const index_t vid : primitive->vertex_ids()) {
                if (mesh.is_obstacle_vertex(vid)) {
                    any_obstacle = true;
                } else {
                    all_obstacle = false;
                }
            }
            if (any_obstacle && !all_obstacle) {
                throw std::logic_error("Primitive has mixed obstacle and non-obstacle vertices!");
            }
            return all_obstacle;
        };
        m_is_obstacle_a = is_obstacle(primitive_a);
        m_is_obstacle_b = is_obstacle(primitive_b);
    }

    if ((primitive_a->n_vertices() + primitive_b->n_vertices()) * DIM
        > ELEMENT_SIZE) {
        logger().error(
            "Too many neighbors for collision pair! {} > {}! Increase MAX_VERT_3D in common.hpp",
            primitive_a->n_vertices() + primitive_b->n_vertices(), MAX_VERT_3D);
    }

    int i = 0;
    m_vertex_ids.assign(
        primitive_a->n_vertices() + primitive_b->n_vertices(),
        -1);
    for (auto& v : primitive_a->vertex_ids()) {
        m_vertex_ids[i++] = v;
    }
    for (auto& v : primitive_b->vertex_ids()) {
        m_vertex_ids[i++] = v;
    }
    assert(i == primitive_a->n_vertices() + primitive_b->n_vertices());

    const double dist_sq = compute_distance(V);
    m_is_active = dist_sq < m_dhat * m_dhat;
    /*

    if (d.norm() < 1e-12) {
        logger().warn(
            "pair distance {}, id {} and {}, dtype {}, active {}", d.norm(),
            _primitive0, _primitive1,
            PrimitiveDistType<PrimitiveA, PrimitiveB>::NAME, m_is_active);

        logger().warn("value {}", (*this)(this->dof(V), params));
    }
    */
}

template<>
double HighOrderCollisionTemplate<Vertex2, Vertex2>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    return point_point_distance(
        vertices.row(m_vertex_ids[0]), vertices.row(m_vertex_ids[n_vertices_a()]));
}

template<>
double HighOrderCollisionTemplate<Edge2P1, Vertex2>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    return point_edge_distance(
        vertices.row(m_vertex_ids[n_vertices_a()]), vertices.row(m_vertex_ids[0]),
        vertices.row(m_vertex_ids[1]));
}

template<>
double HighOrderCollisionTemplate<Edge2P1, Edge2P1>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const auto& ea0 = vertices.row(m_vertex_ids[0]);
    const auto& ea1 = vertices.row(m_vertex_ids[1]);
    const auto& eb0 = vertices.row(m_vertex_ids[n_vertices_a()]);
    const auto& eb1 = vertices.row(m_vertex_ids[n_vertices_a() + 1]);
    return std::min({ point_edge_distance(ea0, eb0, eb1),
                      point_edge_distance(ea1, eb0, eb1),
                      point_edge_distance(eb0, ea0, ea1),
                      point_edge_distance(eb1, ea0, ea1) });
}

template<>
double HighOrderCollisionTemplate<Vertex3, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > m_vertex_ids[0] && n_verts > m_vertex_ids[1]) {
        return point_point_distance(
            vertices.row(m_vertex_ids[0]), vertices.row(m_vertex_ids[n_vertices_a()]));
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

template<>
double HighOrderCollisionTemplate<Edge3P1, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > m_vertex_ids[0] && n_verts > m_vertex_ids[1] && n_verts > m_vertex_ids[2]) {
        return point_edge_distance(
            vertices.row(m_vertex_ids[n_vertices_a()]), vertices.row(m_vertex_ids[0]),
            vertices.row(m_vertex_ids[1]));
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

template<>
double HighOrderCollisionTemplate<Edge3P1, Edge3P1>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > m_vertex_ids[0] && n_verts > m_vertex_ids[1] && n_verts > m_vertex_ids[2] && n_verts > m_vertex_ids[3]) {
        const auto& ea0 = vertices.row(m_vertex_ids[0]);
        const auto& ea1 = vertices.row(m_vertex_ids[1]);
        const auto& eb0 = vertices.row(m_vertex_ids[2]);
        const auto& eb1 = vertices.row(m_vertex_ids[3]);
        return edge_edge_distance(ea0, ea1, eb0, eb1);
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

template<>
double HighOrderCollisionTemplate<Face3P1, Vertex3>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const int n_verts = vertices.rows();
    if (n_verts > m_vertex_ids[0] && n_verts > m_vertex_ids[1] && n_verts > m_vertex_ids[2] && n_verts > m_vertex_ids[3]) {
        const auto& f0 = vertices.row(m_vertex_ids[0]);
        const auto& f1 = vertices.row(m_vertex_ids[1]);
        const auto& f2 = vertices.row(m_vertex_ids[2]);
        const auto& v = vertices.row(m_vertex_ids[3]);
        return point_triangle_distance(v, f0, f1, f2);
    }
    else {
        return std::numeric_limits<double>::max();
    }
}

namespace acp = alternating_contact_potential;


// ----------------------------------------------------

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    return 0;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(
        n_dofs(), n_dofs());
}

// ---- distance ----

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    // This generic implementation is not used.
    // Specializations will provide their own implementation.
    log_and_throw_error("Not implemented");
    return 0;
}

// ----------------------------------------------------

template <>
double HighOrderCollisionTemplate<Edge2P1, Edge2P1>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    if (is_obstacle_a()) return 0.0;
    return acp::potential_EE(positions, params, area_a());
}

template <>
double HighOrderCollisionTemplate<Edge2P1, Vertex2>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    if (is_obstacle_a()) return 0.0;
    return acp::potential_EV(positions, params, area_a());
}


template <>
auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    if (is_obstacle_a()) return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
    return acp::gradient_EE(positions, params, area_a());
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::gradient(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> VectorMax<double, ELEMENT_SIZE>
{
    if (is_obstacle_a()) return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
    return acp::gradient_EV(positions, params, area_a());
}


template <>
auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    if (is_obstacle_a()) return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
    return acp::hessian_EE(positions, params, area_a());
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::hessian(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    if (is_obstacle_a()) return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
    return acp::hessian_EV(positions, params, area_a());
}

// ----------------------------------------------------

template <>
double HighOrderCollisionTemplate<Vertex3, Vertex3>::operator()(
    Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = (positions.template head<3>() - positions.template segment<3>(3)).norm();
    return Math<double>::log_barrier(dist / params.dhat);
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
    return Math<double>::log_barrier(dist / params.dhat);
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
    return Math<double>::log_barrier(dist / params.dhat);
}

template <>
auto HighOrderCollisionTemplate<Vertex3, Vertex3>::gradient(
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
auto HighOrderCollisionTemplate<Vertex3, Vertex3>::hessian(
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

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::core_vertex_ids() const
    -> std::array<index_t, N_CORE_DOFS>
{
    std::array<index_t, N_CORE_DOFS> vids {};
    auto ids = get_core_indices();
    for (int i = 0; i < N_CORE_DOFS; i++) {
        vids[i] = m_vertex_ids[ids[i]];
    }
    return vids;
}

// Note: Primitive pair order cannot change
template class HighOrderCollisionTemplate<Edge2P1, Vertex2>;
template class HighOrderCollisionTemplate<Vertex2, Vertex2>;
template class HighOrderCollisionTemplate<Edge2P1, Edge2P1>;

template class HighOrderCollisionTemplate<Vertex3, Vertex3>;
template class HighOrderCollisionTemplate<Edge3P1, Vertex3>;
template class HighOrderCollisionTemplate<Face3P1, Vertex3>;

} // namespace ipc