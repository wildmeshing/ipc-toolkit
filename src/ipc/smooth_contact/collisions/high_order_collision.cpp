#include "high_order_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/edge_edge.hpp>
#include "line_segment_int_substitution_impl.h"
//#include <math.h>

constexpr int QORD = 16;

namespace ipc {

HighOrderCollision::HighOrderCollision(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh,
    const HighOrderContactParameters& params,
    const double _dhat,
    const Eigen::MatrixXd& V
)
{
	primitive0 = _primitive0;
	primitive1 = _primitive1;
	m_dhat = _dhat;
	m_is_active = true;
	m_vertex_ids.resize(4);
	m_vertex_ids[0] = mesh.edges()(_primitive0, 0);
	m_vertex_ids[1] = mesh.edges()(_primitive0, 1);
	m_vertex_ids[2] = mesh.edges()(_primitive1, 0);
	m_vertex_ids[3] = mesh.edges()(_primitive1, 1);
	/*
    const Eigen::Vector3d ea0 = to_3D(V.row(m_vertex_ids[0]));
    const Eigen::Vector3d ea1 = to_3D(V.row(m_vertex_ids[1]));
    const Eigen::Vector3d eb0 = to_3D(V.row(m_vertex_ids[2]));
    const Eigen::Vector3d eb1 = to_3D(V.row(m_vertex_ids[3]));
	const auto dt = edge_edge_distance_type(ea0, ea1, eb0, eb1);
    const double dist_sq = edge_edge_sqr_distance(Eigen::ConstRef<Eigen::Vector3d>(ea0), Eigen::ConstRef<Eigen::Vector3d>(ea1), Eigen::ConstRef<Eigen::Vector3d>(eb0), Eigen::ConstRef<Eigen::Vector3d>(eb1), dt);
    m_is_active = dist_sq < _dhat * _dhat;
    if (dist_sq < 1e-12) {
        logger().warn("edge-edge pair distance is very small: {}", dist_sq);
    }*/
}

template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, 2>, std::vector<double>, Eigen::Vector2<T>> SampleEdge(
    Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge_positions, int quad_order
){
	const Eigen::Vector2<T> p0 = edge_positions.row(0);
	const Eigen::Vector2<T> p1 = edge_positions.row(1);

	std::vector<double> nodes;
	std::vector<double> weights;
	contact_potential_integration::gauss_legendre(quad_order, nodes, weights);

	Eigen::Matrix<T, Eigen::Dynamic, 2> M(quad_order, 2);
	for (size_t i = 0; i<quad_order; ++i) {
		const double t = (nodes.at(i)+1)/2;
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
	ScalarBase::setVariableCount(HighOrderCollision::N_CORE_DOFS);
	Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
	Eigen::Vector2<T> normal;
	std::vector<double> weights;
	const int qord = params.quad_points;
	std::tie(qp, weights, normal) = SampleEdge<T>(edge1_pos, qord);
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
	return acc;
}

double HighOrderCollision::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
	Eigen::Matrix<double, 4, 2> positions_ad = slice_positions<double, 4, 2>(positions);
	Eigen::Matrix<double, 2, 2> edge0_pos = positions_ad.topRows(2);
	Eigen::Matrix<double, 2, 2> edge1_pos = positions_ad.bottomRows(2);
	return potential_onesided<double>(edge0_pos, edge1_pos, params)
		+ potential_onesided<double>(edge1_pos, edge0_pos, params);
}

auto HighOrderCollision::gradient(
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

auto HighOrderCollision::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	using T = ADHessian<N_CORE_DOFS>;
	Eigen::Matrix<T, 4, 2> positions_ad = slice_positions<T, 4, 2>(positions);
	Eigen::Matrix<T, 2, 2> edge0_pos = positions_ad.topRows(2);
	Eigen::Matrix<T, 2, 2> edge1_pos = positions_ad.bottomRows(2);
	T acc = potential_onesided<T>(edge0_pos, edge1_pos, params)
		+ potential_onesided<T>(edge1_pos, edge0_pos, params);
	return acc.Hess;
}

double HighOrderCollision::compute_distance(Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    const Eigen::Vector3d ea0 = to_3D(vertices.row(m_vertex_ids[0]));
    const Eigen::Vector3d ea1 = to_3D(vertices.row(m_vertex_ids[1]));
    const Eigen::Vector3d eb0 = to_3D(vertices.row(m_vertex_ids[2]));
    const Eigen::Vector3d eb1 = to_3D(vertices.row(m_vertex_ids[3]));
    return edge_edge_distance(ea0, ea1, eb0, eb1);
}

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

} // namespace ipc
