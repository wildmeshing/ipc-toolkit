#include "high_order_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "line_segment_int_substitution_impl.h"
#include "smoothed_offset_potential_polyline.h"

namespace ipc {

// clang-format off
template <> CollisionType HighOrderCollisionTemplate<Vertex2, Vertex2>::type() const { return CollisionType::VERTEX_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Edge2P1, Vertex2>::type() const { return CollisionType::EDGE_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Edge2P1, Edge2P1>::type() const { return CollisionType::EDGE_EDGE; }
// clang-format on

// clang-format off
template <> std::string HighOrderCollisionTemplate<Vertex2, Vertex2>::name() const { return "vv_2d"; }
template <> std::string HighOrderCollisionTemplate<Edge2P1, Vertex2>::name() const { return "ve_2d"; }
template <> std::string HighOrderCollisionTemplate<Edge2P1, Edge2P1>::name() const { return "ee_2d"; }
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
            x.segment<3>(i * 3) = X.row(m_vertex_ids[i]);
        }
    } else {
        throw std::runtime_error("Invalid dimension!");
    }
    return x;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::get_core_indices() const
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
        
    if ((primitive_a->n_vertices() + primitive_b->n_vertices()) * DIM
        > ELEMENT_SIZE) {
        logger().error(
            "Too many neighbors for collision pair! {} > {}! Increase MAX_VERT_3D in common.hpp",
            primitive_a->n_vertices() + primitive_b->n_vertices(), MAX_VERT_3D);
    }

    int i = 0;
    m_vertex_ids.assign(
        primitive_a->vertex_ids().size() + primitive_b->vertex_ids().size(),
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
        vertices.row(m_vertex_ids[0]), vertices.row(m_vertex_ids[1]));
}

template<>
double HighOrderCollisionTemplate<Edge2P1, Vertex2>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    return point_edge_distance(
        vertices.row(m_vertex_ids[2]), vertices.row(m_vertex_ids[0]),
        vertices.row(m_vertex_ids[1]));
}

template<>
double HighOrderCollisionTemplate<Edge2P1, Edge2P1>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const auto& ea0 = vertices.row(m_vertex_ids[0]);
    const auto& ea1 = vertices.row(m_vertex_ids[1]);
    const auto& eb0 = vertices.row(m_vertex_ids[2]);
    const auto& eb1 = vertices.row(m_vertex_ids[3]);
    return std::min({ point_edge_distance(ea0, eb0, eb1),
                      point_edge_distance(ea1, eb0, eb1),
                      point_edge_distance(eb0, ea0, ea1),
                      point_edge_distance(eb1, ea0, ea1) });
}

template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, 2>, std::vector<double>, Eigen::Vector2<T>> sample_edge(
    Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge_positions, int quad_order, std::array<T, 2> window={{0.0, 1.0}}
){
	const Eigen::Vector2<T> p0 = edge_positions.row(0);
	const Eigen::Vector2<T> p1 = edge_positions.row(1);
	if (window[0] < 0.0 || window[1] > 1.0 || window[1] < window[0]) {
		throw std::runtime_error("Invalid window!");
	}

	std::vector<double> nodes;
	std::vector<double> weights;
	contact_potential_integration::gauss_legendre(quad_order, nodes, weights);

	Eigen::Matrix<T, Eigen::Dynamic, 2> M(quad_order, 2);
	for (size_t i = 0; i<quad_order; ++i) {
		const double t01 = (nodes.at(i)+1)/2;
		const T t = ((1-t01) * window[0] + t01 * window[1]);
		const Eigen::Vector2<T> P = ((1-t) * p0 + t * p1);
		M.row(i) = P.transpose();
	}

	Eigen::Vector2<T> edge_vec = p1 - p0;
	edge_vec.normalize();
	const Eigen::Vector2<T> edge_normal(-edge_vec.y(), edge_vec.x());

	return {M, weights, edge_normal};
}

/*
template <typename T>
T potential_EE_onesided_old(
	Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge0_pos,
	Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge1_pos,
    const HighOrderContactParameters& params
) {
	Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
	Eigen::Vector2<T> normal;
	std::vector<double> weights;
	const int qord = params.quad_points;
	const contact_potential_integration::LineSegment<T> projected_segment(
		{{ edge0_pos(0, 0), edge0_pos(0, 1) }},
		{{ edge0_pos(1, 0), edge0_pos(1, 1) }});
	const contact_potential_integration::LineSegment<T> sampled_segment(
		{{ edge1_pos(0, 0), edge1_pos(0, 1) }},
		{{ edge1_pos(1, 0), edge1_pos(1, 1) }});
	auto window = contact_potential_integration::compute_quadrature_window(
		sampled_segment, projected_segment, params.alpha_t);
	std::tie(qp, weights, normal) = sample_edge<T>(edge1_pos, qord, window);
	const T scale = window[1] - window[0];
	T acc(0.0);
	for (size_t q=0; q<qord; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		const std::array<T, 2> point{{ p(0), p(1) }};
		const std::array<T, 2> normal_arr{{ normal(0), normal(1) }};
		acc += weights[q] * contact_potential_integration::integrate_potential_line_segment_substitution<T>(
			projected_segment, point, normal_arr, params.dhat, params.alpha_t, params.r, qord);
	}
	return scale*acc;
}
*/

// ----------------------------------------------------

template <typename T>
T potential_VV_onesided(
    const Eigen::Matrix<T, Eigen::Dynamic, 2>& vertex_stencil,
    const Eigen::Matrix<T, Eigen::Dynamic, 2>& qpoint_stencil,
    const HighOrderContactParameters& params)
{
    // The central vertex of the second stencil is our query point.
    const std::array<T, 2> query_point = {{ qpoint_stencil(0, 0), qpoint_stencil(0, 1) }};

    // The central vertex of the first stencil is the vertex for which we are computing the potential.
    const std::array<T, 2> vertex_pt = {{ vertex_stencil(0, 0), vertex_stencil(0, 1) }};

    T phi_start_next_val;
    T phi_end_prev_val;
    const T* phi_start_next = nullptr;
    const T* phi_end_prev = nullptr;

    if (vertex_stencil.rows() >= 2) {
        // The "next" edge is from the central vertex (0) to the first neighbor (1).
        const Eigen::Vector2<T> p0 = vertex_stencil.row(0);
        const Eigen::Vector2<T> p_next = vertex_stencil.row(1);
        const Eigen::Vector2<T> tangent_next = (p_next - p0).normalized();
        const Eigen::Vector2<T> normal_next(-tangent_next.y(), tangent_next.x());
        const std::array<T, 2> rel_next = {{ query_point[0] - p0(0), query_point[1] - p0(1) }};
        const T r_q_next = rel_next[0] * normal_next(0) + rel_next[1] * normal_next(1);
        const T y_q_next = rel_next[0] * tangent_next(0) + rel_next[1] * tangent_next(1);
        phi_start_next_val = smoothed_offset_potential::phi_value(r_q_next, y_q_next, T(0));
        phi_start_next = &phi_start_next_val;
    }

    if (vertex_stencil.rows() >= 3) {
        // The "previous" edge is from the second neighbor (2) to the central vertex (0).
        const Eigen::Vector2<T> p0 = vertex_stencil.row(0);
        const Eigen::Vector2<T> p_prev = vertex_stencil.row(2);
        const Eigen::Vector2<T> tangent_prev = (p0 - p_prev).normalized();
        const Eigen::Vector2<T> normal_prev(-tangent_prev.y(), tangent_prev.x());
        const std::array<T, 2> rel_prev = {{ query_point[0] - p_prev(0), query_point[1] - p_prev(1) }};
        const T r_q_prev = rel_prev[0] * normal_prev(0) + rel_prev[1] * normal_prev(1);
        const T y_q_prev = rel_prev[0] * tangent_prev(0) + rel_prev[1] * tangent_prev(1);
        phi_end_prev_val = smoothed_offset_potential::phi_value(r_q_prev, y_q_prev, (p0 - p_prev).norm());
        phi_end_prev = &phi_end_prev_val;
    }

    return smoothed_offset_potential::polyline_vertex_potential(
        query_point, vertex_pt, phi_start_next, phi_end_prev, T(params.alpha_t), params.r);
}

template <typename T>
T potential_VV(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b)
{
    Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
        slice_positions<T, Eigen::Dynamic, 2>(positions);
    Eigen::Matrix<T, Eigen::Dynamic, 2> v0_stencil_pos = all_pos.topRows(n_vertices_a);
    Eigen::Matrix<T, Eigen::Dynamic, 2> v1_stencil_pos = all_pos.bottomRows(n_vertices_b);

    return potential_VV_onesided(v0_stencil_pos, v1_stencil_pos, params)
        + potential_VV_onesided(v1_stencil_pos, v0_stencil_pos, params);
}

// ----------------------------------------------------

template <typename T>
T potential_EV_onesided(
    const Eigen::Matrix<T, 2, 2>& edge_pos,
    const Eigen::Matrix<T, Eigen::Dynamic, 2>& vertex_stencil_pos,
    const HighOrderContactParameters& params)
{
    Eigen::Matrix<T, 1, 2> vertex_pos = vertex_stencil_pos.topRows(1);
    const Eigen::Vector2<T> p0 = edge_pos.row(0);
	const Eigen::Vector2<T> p1 = edge_pos.row(1);
	const Eigen::Vector2<T> tangent_vec = p1 - p0;
	const T length = tangent_vec.norm();
	const Eigen::Vector2<T> tangent = tangent_vec / length;
	const Eigen::Vector2<T> normal_vec(-tangent.y(), tangent.x());

	const std::array<T, 2> p0_arr{{ p0(0), p0(1) }};
	const std::array<T, 2> tangent_arr{{ tangent(0), tangent(1) }};
	const std::array<T, 2> normal_arr{{ normal_vec(0), normal_vec(1) }};

    const std::array<T, 2> point{{ vertex_pos(0), vertex_pos(1) }};
    T phi_start, phi_end;
    return smoothed_offset_potential::polyline_edge_potential<T>(
        point, p0_arr, tangent_arr, normal_arr, length, params.alpha_t,
        params.r, phi_start, phi_end);
}

template <typename T>
T potential_VE_onesided(
    const Eigen::Matrix<T, Eigen::Dynamic, 2>& vertex_stencil_pos,
    const Eigen::Matrix<T, 2, 2>& edge_pos,
    const HighOrderContactParameters& params)
{
    Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
    Eigen::Vector2<T> normal;
    std::vector<double> weights;
    const int qord = params.quad_points;

    // Sample points on the edge
    std::tie(qp, weights, normal) = sample_edge<T>(edge_pos, qord);
    const T scale = (edge_pos.row(1) - edge_pos.row(0)).norm();

    T acc(0.0);
    for (size_t q = 0; q < qord; ++q) {
        Eigen::Matrix<T, 1, 2> qpoint_stencil = qp.row(q);
        acc += weights[q]
            * potential_VV_onesided<T>(vertex_stencil_pos, qpoint_stencil, params);
    }
    return scale * acc;
}

template <typename T>
T potential_EV(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>>
        positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b)
{
    Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
        slice_positions<T, Eigen::Dynamic, 2>(positions);
    Eigen::Matrix<T, 2, 2> edge_pos = all_pos.topRows(n_vertices_a);
    Eigen::Matrix<T, Eigen::Dynamic, 2> vertex_stencil_pos =
        all_pos.bottomRows(n_vertices_b);

    return potential_EV_onesided(edge_pos, vertex_stencil_pos, params)
        + potential_VE_onesided(vertex_stencil_pos, edge_pos, params);
}

// ----------------------------------------------------

template <typename T>
T potential_EE_onesided(
	Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge0_pos,
	Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge1_pos,
    const HighOrderContactParameters& params
) {
	Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
	Eigen::Vector2<T> normal;
	std::vector<double> weights;
	const int qord = params.quad_points;

	// "segment" is the segment we are computing the potential for (edge0)
	const Eigen::Vector2<T> p0 = edge0_pos.row(0);
	const Eigen::Vector2<T> p1 = edge0_pos.row(1);
	const Eigen::Vector2<T> tangent_vec = p1 - p0;
	const T length = tangent_vec.norm();
	const Eigen::Vector2<T> tangent = tangent_vec / length;
	const Eigen::Vector2<T> normal_vec(-tangent.y(), tangent.x());

	const std::array<T, 2> p0_arr{{ p0(0), p0(1) }};
	const std::array<T, 2> tangent_arr{{ tangent(0), tangent(1) }};
	const std::array<T, 2> normal_arr{{ normal_vec(0), normal_vec(1) }};

	// "sampled_segment" is the segment we integrate over (edge1)
	std::tie(qp, weights, normal) = sample_edge<T>(edge1_pos, qord);
	const T scale = (edge1_pos.row(1) - edge1_pos.row(0)).norm();
	T acc(0.0);
	for (size_t q=0; q<qord; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		const std::array<T, 2> point{{ p(0), p(1) }};

		T phi_start, phi_end;
		acc += weights[q]
			* smoothed_offset_potential::polyline_edge_potential<T>(
				point,
				p0_arr,
				tangent_arr,
				normal_arr,
				length,
				params.alpha_t,
				params.r,
				phi_start,
				phi_end);
	}
	return scale*acc;
}

template <typename T>
T potential_EE(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params
) {
	Eigen::Matrix<T, 4, 2> all_pos = slice_positions<T, 4, 2>(positions);
    Eigen::Matrix<T, 2, 2> edge0_pos = all_pos.topRows(2);
	Eigen::Matrix<T, 2, 2> edge1_pos = all_pos.bottomRows(2);
    //TODO do we need /2?
    return (potential_EE_onesided<T>(edge0_pos, edge1_pos, params)
		+ potential_EE_onesided<T>(edge1_pos, edge0_pos, params));
}

// ----------------------------------------------------

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
{
    return 0;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
    return Vector<double, -1, ELEMENT_SIZE>::Zero(n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
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
    return 0;
}

template <>
double HighOrderCollisionTemplate<Edge2P1, Edge2P1>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
{
    return potential_EE<double>(positions, params);
}

template <>
double HighOrderCollisionTemplate<Edge2P1, Vertex2>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
{
    return potential_EV<double>(
        positions, params, primitive_a->n_vertices(), primitive_b->n_vertices());
}

template <>
double HighOrderCollisionTemplate<Vertex2, Vertex2>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
{
    return potential_VV<double>(
        positions, params, primitive_a->n_vertices(), primitive_b->n_vertices());
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const -> Vector<double, -1, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_EV<ADGrad<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices())
        .grad;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Vertex2>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const -> Vector<double, -1, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_VV<ADGrad<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices())
        .grad;
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	return potential_EE<ADGrad<N_CORE_DOFS>>(positions, params).grad;
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

template <>
auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	return potential_EE<ADHessian<N_CORE_DOFS>>(positions, params).Hess;
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_EV<ADHessian<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices())
        .Hess;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Vertex2>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_VV<ADHessian<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices())
        .Hess;
}

// Note: Primitive pair order cannot change
template class HighOrderCollisionTemplate<Edge2P1, Vertex2>;
template class HighOrderCollisionTemplate<Vertex2, Vertex2>;
template class HighOrderCollisionTemplate<Edge2P1, Edge2P1>;

} // namespace ipc