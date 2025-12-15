#include "high_order_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "line_segment_int_substitution_impl.h"

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
    Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge_positions, int quad_order, std::array<T, 2> window={0.0, 1.0}
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

template <typename T>
T potential_onesided(
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
	const T window_width = window[1] - window[0];
	T acc(0.0);
	for (size_t q=0; q<qord; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		const contact_potential_integration::LineSegment<T> segment(
			{{ edge0_pos(0, 0), edge0_pos(0, 1) }},
			{{ edge0_pos(1, 0), edge0_pos(1, 1) }});
		const std::array<T, 2> point{{ p(0), p(1) }};
		const std::array<T, 2> normal_arr{{ normal(0), normal(1) }};
		acc += weights[q] * contact_potential_integration::integrate_potential_line_segment_substitution<T>(
			segment, point, normal_arr, params.dhat, params.alpha_t, params.r, qord);
	}
	return window_width*acc;
}

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    return 0;
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
    return Vector<double, -1, ELEMENT_SIZE>::Zero(n_dofs());
}

template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
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
    const HighOrderContactParameters& params) const
{
	Eigen::Matrix<double, 4, 2> positions_ad = slice_positions<double, 4, 2>(positions);
	Eigen::Matrix<double, 2, 2> edge0_pos = positions_ad.topRows(2);
	Eigen::Matrix<double, 2, 2> edge1_pos = positions_ad.bottomRows(2);
	return potential_onesided<double>(edge0_pos, edge1_pos, params)
		+ potential_onesided<double>(edge1_pos, edge0_pos, params);
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	using T = ADGrad<N_CORE_DOFS>;
	Eigen::Matrix<T, 4, 2> positions_ad = slice_positions<T, 4, 2>(positions);
	Eigen::Matrix<T, 2, 2> edge0_pos = positions_ad.topRows(2);
	Eigen::Matrix<T, 2, 2> edge1_pos = positions_ad.bottomRows(2);
	T acc = potential_onesided<T>(edge0_pos, edge1_pos, params)
		+ potential_onesided<T>(edge1_pos, edge0_pos, params);
	return acc.grad;
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
    const HighOrderContactParameters& params) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	using T = ADHessian<N_CORE_DOFS>;
	Eigen::Matrix<T, 4, 2> positions_ad = slice_positions<T, 4, 2>(positions);
	Eigen::Matrix<T, 2, 2> edge0_pos = positions_ad.topRows(2);
	Eigen::Matrix<T, 2, 2> edge1_pos = positions_ad.bottomRows(2);
	T acc = potential_onesided<T>(edge0_pos, edge1_pos, params)
		+ potential_onesided<T>(edge1_pos, edge0_pos, params);
	return acc.Hess;
}

// Note: Primitive pair order cannot change
template class HighOrderCollisionTemplate<Edge2P1, Vertex2>;
template class HighOrderCollisionTemplate<Vertex2, Vertex2>;
template class HighOrderCollisionTemplate<Edge2P1, Edge2P1>;

} // namespace ipc