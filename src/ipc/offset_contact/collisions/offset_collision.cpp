#include "offset_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "offset_potential_linear.h"

namespace ipc {

// clang-format off
template <> OffsetCollisionType OffsetCollisionTemplate<ogcVert2, ogcVert2>::type() const { return OffsetCollisionType::VERTEX_VERTEX; }
template <> OffsetCollisionType OffsetCollisionTemplate<ogcEdge2, ogcVert2>::type() const { return OffsetCollisionType::EDGE_VERTEX; }
// clang-format on

// clang-format off
template <> std::string OffsetCollisionTemplate<ogcVert2, ogcVert2>::name() const { return "vv_2d"; }
template <> std::string OffsetCollisionTemplate<ogcEdge2, ogcVert2>::name() const { return "ve_2d"; }
// clang-format on

Eigen::VectorXd OffsetCollision::dof(Eigen::ConstRef<Eigen::MatrixXd> X) const
{
    const int DIM = X.cols();
    Eigen::VectorXd x(num_vertices() * DIM);
    if (DIM == 2) {
        for (int i = 0; i < num_vertices(); i++) {
            x.segment<2>(i * 2) = X.row(m_vertex_ids[i]);
        }
    } else if (DIM == 3) {
        for (int i = 0; i < num_vertices(); i++) {
            x.segment<3>(i * 3) = X.row(m_vertex_ids[i]);
        }
    } else {
        throw std::runtime_error("Invalid dimension!");
    }
    return x;
}

template <typename PrimitiveA, typename PrimitiveB>
auto OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::get_core_indices() const
    -> Vector<int, N_CORE_DOFS>
{
    Vector<int, N_CORE_DOFS> core_indices;
    core_indices << Eigen::VectorXi::LinSpaced(
        N_CORE_DOFS_A, 0, N_CORE_DOFS_A - 1),
        Eigen::VectorXi::LinSpaced(
            N_CORE_DOFS_B, primitive_a->n_dofs(),
            primitive_a->n_dofs() + N_CORE_DOFS_B - 1);
    return core_indices;
}

template <typename PrimitiveA, typename PrimitiveB>
OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::OffsetCollisionTemplate(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh,
    const OffsetContactParameters& params,
    const double _dhat,
    const Eigen::MatrixXd& V)
    : OffsetCollision(_primitive0, _primitive1, _dhat, mesh)
{
    primitive_a = std::make_unique<PrimitiveA>(_primitive0, mesh, V);
    primitive_b = std::make_unique<PrimitiveB>(_primitive1, mesh, V);

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
            throw std::logic_error(
                "Primitive has mixed obstacle and non-obstacle vertices!");
        }
        return all_obstacle;
    };
    m_is_obstacle0 = is_obstacle(primitive_a);
    m_is_obstacle1 = is_obstacle(primitive_b);
        
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
double OffsetCollisionTemplate<ogcVert2, ogcVert2>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    return point_point_distance(
        vertices.row(m_vertex_ids[0]), vertices.row(m_vertex_ids[n_vertices_a()]));
}

template<>
double OffsetCollisionTemplate<ogcEdge2, ogcVert2>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    return point_edge_distance(
        vertices.row(m_vertex_ids[n_vertices_a()]), vertices.row(m_vertex_ids[0]),
        vertices.row(m_vertex_ids[1]));
}


// ----------------------------------------------------

template <typename T>
T potential_VV_onesided(
    Eigen::ConstRef<Eigen::Matrix<T, Eigen::Dynamic, 2>> v_a,
    Eigen::ConstRef<Eigen::Matrix<T, Eigen::Dynamic, 2>> v_b,
    const OffsetContactParameters& params)
{
    const std::array<T, 2> point = {{ v_b(0, 0), v_b(0, 1) }}; // Query point (Vertex B)
    const std::array<T, 2> vertex_pt = {{ v_a(0, 0), v_a(0, 1) }}; // Source vertex (Vertex A)

    T phi_start_next_val;
    T phi_end_prev_val;
    const T* phi_start_next = nullptr;
    const T* phi_end_prev = nullptr;

    const Eigen::Vector2<T> p0 = v_a.row(0);

    // Neighbor 1 (Next edge: p0 -> p_next)
    if (v_a.rows() >= 2) {
        const Eigen::Vector2<T> p_next = v_a.row(1);
        Eigen::Vector2<T> t = (p_next - p0).normalized();
        Eigen::Vector2<T> n = {t.y(), -t.x()};
        
        Eigen::Vector2<T> rel = Eigen::Vector2<T>(point[0], point[1]) - p0;
        T r_q = rel.dot(n);
        T y_q = rel.dot(t);
        
        // phi at start of next edge (y=0)
        phi_start_next_val = offset_potential::phi_value(r_q, y_q, T(0));
        phi_start_next = &phi_start_next_val;
    }

    // Neighbor 2 (Prev edge: p_prev -> p0)
    if (v_a.rows() >= 3) {
        const Eigen::Vector2<T> p_prev = v_a.row(2);
        Eigen::Vector2<T> edge_vec = p0 - p_prev;
        T len = edge_vec.norm();
        Eigen::Vector2<T> t = edge_vec / len;
        Eigen::Vector2<T> n = {t.y(), -t.x()};

        Eigen::Vector2<T> rel = Eigen::Vector2<T>(point[0], point[1]) - p_prev;
        T r_q = rel.dot(n);
        T y_q = rel.dot(t);

        // phi at end of prev edge (y=len)
        phi_end_prev_val = offset_potential::phi_value(r_q, y_q, len);
        phi_end_prev = &phi_end_prev_val;
    }

    return offset_potential::polyline_vertex_potential(
        point, vertex_pt, phi_start_next, phi_end_prev, params.r, params.dhat);
}

template <typename T>
T potential_VV(
    Eigen::ConstRef<Vector<double, -1, OffsetCollision::ELEMENT_SIZE>>
        positions,
    const OffsetContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b,
    const bool obst_a,
    const bool obst_b)
{
    Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
        slice_positions<T, Eigen::Dynamic, 2>(positions);
    const Eigen::Matrix<T, Eigen::Dynamic, 2> v_a = all_pos.topRows(n_vertices_a);
    const Eigen::Matrix<T, Eigen::Dynamic, 2> v_b = all_pos.bottomRows(n_vertices_b);
    T pot = 0;
    if (!obst_a) {
        pot += potential_VV_onesided<T>(v_b, v_a, params);
    }
    if (!obst_b) {
        pot += potential_VV_onesided<T>(v_a, v_b, params);
    }
    return pot;
}

// ----------------------------------------------------

template <typename T>
T potential_VE(
    Eigen::ConstRef<Vector<double, -1, OffsetCollision::ELEMENT_SIZE>>
        positions,
    const OffsetContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b)
{
    Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
        slice_positions<T, Eigen::Dynamic, 2>(positions);
    const Eigen::Matrix<T, 2, 2> edge_pos = all_pos.topRows(n_vertices_a);
    const Eigen::Matrix<T, Eigen::Dynamic, 2> vertex_stencil = all_pos.bottomRows(n_vertices_b);
    const std::array<T, 2> vertex_pt = {{ vertex_stencil(0, 0), vertex_stencil(0, 1) }};

    // Edge geometry
    const Eigen::Vector2<T> p0 = edge_pos.row(0);
    const Eigen::Vector2<T> p1 = edge_pos.row(1);
    const Eigen::Vector2<T> t_vec = p1 - p0;
    const T len = t_vec.norm();
    const Eigen::Vector2<T> t_hat = t_vec / len;
    const Eigen::Vector2<T> n_hat = {-t_hat.y(), t_hat.x()};

    const std::array<T, 2> p0_arr = {{p0.x(), p0.y()}};
    const std::array<T, 2> t_arr = {{t_hat.x(), t_hat.y()}};
    const std::array<T, 2> n_arr = {{n_hat.x(), n_hat.y()}};

    T phi_start, phi_end;
    return offset_potential::polyline_edge_potential(
        vertex_pt, p0_arr, t_arr, n_arr, len,
        params.r, params.dhat,
        phi_start, phi_end);
}

// ----------------------------------------------------

template <typename PrimitiveA, typename PrimitiveB>
double OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const
{
    return 0;
}

template <typename PrimitiveA, typename PrimitiveB>
auto OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
    return Vector<double, -1, ELEMENT_SIZE>::Zero(n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
auto OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(
        n_dofs(), n_dofs());
}

// ---- distance ----

template <typename PrimitiveA, typename PrimitiveB>
double OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    // This generic implementation is not used.
    // Specializations will provide their own implementation.
    return 0;
}

template <>
double OffsetCollisionTemplate<ogcEdge2, ogcVert2>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const
{
    if (is_obstacle1()) return 0.0;
    return potential_VE<double>(
        positions, params, primitive_a->n_vertices(), primitive_b->n_vertices());
}

template <>
auto OffsetCollisionTemplate<ogcEdge2, ogcVert2>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const -> Vector<double, -1, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    if (is_obstacle1()) return Vector<double, -1, ELEMENT_SIZE>::Zero(n_dofs());
    return potential_VE<ADGrad<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices())
        .grad;
}

template <typename PrimitiveA, typename PrimitiveB>
auto OffsetCollisionTemplate<PrimitiveA, PrimitiveB>::core_vertex_ids() const
    -> std::array<index_t, N_CORE_DOFS>
{
    std::array<index_t, N_CORE_DOFS> vids {};
    auto ids = get_core_indices();
    for (int i = 0; i < N_CORE_DOFS; i++) {
        vids[i] = m_vertex_ids[ids[i]];
    }
    return vids;
}

template <>
auto OffsetCollisionTemplate<ogcEdge2, ogcVert2>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    if (is_obstacle1()) return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
    return potential_VE<ADHessian<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices())
        .Hess;
}

template <>
double OffsetCollisionTemplate<ogcVert2, ogcVert2>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const
{
    return potential_VV<double>(
        positions, params, primitive_a->n_vertices(), primitive_b->n_vertices(), is_obstacle0(), is_obstacle1());
}

template <>
auto OffsetCollisionTemplate<ogcVert2, ogcVert2>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const -> Vector<double, -1, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_VV<ADGrad<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices(), is_obstacle0(), is_obstacle1()).grad;
}

template <>
auto OffsetCollisionTemplate<ogcVert2, ogcVert2>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const OffsetContactParameters& params) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_VV<ADHessian<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices(), is_obstacle0(), is_obstacle1()).Hess;
}

// Note: Primitive pair order cannot change
template class OffsetCollisionTemplate<ogcEdge2, ogcVert2>;
template class OffsetCollisionTemplate<ogcVert2, ogcVert2>;

} // namespace ipc