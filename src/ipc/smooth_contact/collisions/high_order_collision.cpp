#include "high_order_collision.hpp"

#include <ipc/config.hpp>
#include <math.h>

namespace ipc {
HighOrderCollision::HighOrderCollision(
    index_t _primitive0,
    index_t _primitive1,
    const CollisionMesh& mesh,
    const SmoothContactParameters& params,
    const double _dhat,
    const Eigen::MatrixXd& V
) : SmoothCollision(_primitive0, _primitive1, _dhat, mesh)
{
	m_is_active = true;
	m_vertex_ids.resize(4);
	m_vertex_ids[0] = mesh.edges()(_primitive0, 0);
	m_vertex_ids[1] = mesh.edges()(_primitive0, 1);
	m_vertex_ids[2] = mesh.edges()(_primitive1, 0);
	m_vertex_ids[3] = mesh.edges()(_primitive1, 1);
	/*
    m_is_active = (d.norm() < m_dhat) && primitive_a->is_active() && primitive_b->is_active();
    if (d.norm() < 1e-12) {
        logger().warn(
            "pair distance {}, id {} and {}, dtype {}, active {}", d.norm(),
            _primitive0, _primitive1,
            PrimitiveDistType<PrimitiveA, PrimitiveB>::NAME, m_is_active);

        logger().warn("value {}", (*this)(this->dof(V), params));
    }*/
}

constexpr size_t QSIZE = 7;
constexpr double QUADPOINTS[QSIZE] = {0.0, 0.08488805186071646, 0.2655756032646428, 0.5, 0.7344243967353572, 0.9151119481392833, 1.0};
constexpr double QUADWEIGHTS[QSIZE] = {0.04761905, 0.27682605, 0.43174538, 0.48761905, 0.43174538, 0.27682605, 0.04761905};

template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, 2>, Eigen::Vector2<T>, T> sampleEE2Aligned(
    Eigen::ConstRef<Vector<T, -1, 8>> positions
){
	// we align A with the x axis and sample B
	const Vector<T, 4> A = positions.template head<4>();
	const Eigen::Vector2<T> A0 = A.template head<2>();
	const Eigen::Vector2<T> A1 = A.template tail<2>();
	const Eigen::Vector2<T> AV = A1 - A0;
	// new reference frame
	const T L = AV.norm();
	const Eigen::Vector2<T> Anorm = AV / L;
	const Eigen::Vector2<T> Aorth(-Anorm.y(),Anorm.x());
	Eigen::Matrix<T, 2, 2> rot;
	rot.col(0) = Anorm;
	rot.col(1) = Aorth;
	const Vector<T, 4> B = positions.template tail<4>();
	const Eigen::Vector2<T> B0 = B.template head<2>() - A0;
	const Eigen::Vector2<T> B1 = B.template tail<2>() - A0;
	const Eigen::Vector2<T> B0r = rot.transpose()*B0; // Rotate B0 to the new frame
	const Eigen::Vector2<T> B1r = rot.transpose()*B1; // Rotate B1 to the new frame
	Eigen::Matrix<T, Eigen::Dynamic, 2> M(QSIZE, 2);
	for (size_t i = 0; i<QSIZE; ++i) {
		const double t = QUADPOINTS[i];
		const Eigen::Vector2<T> P = (1-t)*B0r + t*B1r;
		M.row(i) = P.transpose();
	}
	const Eigen::Vector2<T> Br_vec = B1r - B0r;
	Eigen::Vector2<T> Br_normal(-Br_vec.y(), Br_vec.x());
	Br_normal.normalize();
	return {M, Br_normal, L};
}

template <typename T>
T delta_alpha(const T& z, const double alpha) {
	const T a2 = 2. / alpha;
    return a2 * Math<T>::cubic_spline(a2 * z);
}

template <typename T>
T h_eps(const T& z, const double eps) {
    return 3 * Math<T>::cubic_spline(2*z/eps) / 2;
}

template <typename T>
T potentialVE2(
	const Eigen::Vector2<T>& q, const Eigen::Vector2<T>& n,
	const T& L, const SmoothContactParameters& params
) {
	const double alpha = params.alpha_t;
	const double eps = params.dhat;
	if (n.y()>0) return 0;
	const T &q0 = q.x();
	const T &q1 = q.y();
	const double phi = std::asin(alpha);
	constexpr double HPI = 1.5707963267948966;
	const T theta = acos(-n.x());
	// range limited by the segment length
	const T psi_min_segment = atan(fmin(-q0/q1,(L-q0)/q1));
	const T psi_max_segment = atan(fmax(-q0/q1,(L-q0)/q1));
	// range limited by the normal angle
	const T psi_min_angle = fmax(-phi,-phi-theta-HPI);
	const T psi_max_angle = fmin(phi,phi-theta-HPI);
	// range limited by the distance
	const T psi_min_dist = -acos(q1/eps);
	const T psi_max_dist = acos(q1/eps);
	// final range
	const T psi_min = fmax(fmax(psi_min_segment, psi_min_angle), psi_min_dist);
	const T psi_max = fmin(fmin(psi_max_segment, psi_max_angle), psi_max_dist);
	if (psi_min>=psi_max) return 0;
	T potential = 0;
	for (size_t qp=0; qp<QSIZE; ++qp) {
		const T psi = psi_min + (psi_max-psi_min)*QUADPOINTS[qp];
		const T theta1 = HPI + theta + psi;
		const T cosPsi = cos(psi);
		const T q1DcosPsi = q1/cosPsi;
		const T J = q1DcosPsi / (L*cosPsi);
		potential += cosPsi*cosPsi*J*
			h_eps(q1DcosPsi, eps)*
			delta_alpha(abs(sin(theta1)), alpha)*
			Math<T>::smooth_heaviside(cos(theta1), alpha)*
			delta_alpha(abs(sin(psi)), alpha)*
			Math<T>::smooth_heaviside(cosPsi, alpha);
	}
	return potential/(q1*q1);
}

double HighOrderCollision::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
{
	Eigen::MatrixX2d qp;
	Eigen::Vector2d normal;
	double L;
	std::tie(qp, normal, L) = sampleEE2Aligned<double>(positions);
	double acc = 0;
	for (size_t q=0; q<QSIZE; ++q) {
		const Eigen::Vector2d p = qp.row(q);
		acc += QUADWEIGHTS[q] * potentialVE2<double>(p, normal, L, params);
	}
    logger().debug("HighOrderCollision::operator() -> {}", acc);

    return acc;
}

auto HighOrderCollision::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	ScalarBase::setVariableCount(N_CORE_DOFS);
	using T = ADGrad<N_CORE_DOFS>;
	Vector<T, N_CORE_DOFS> positions_ad = slice_positions<T, N_CORE_DOFS, 1>(positions);
	Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
	Eigen::Vector2<T> normal;
	T L;
	std::tie(qp, normal, L) = sampleEE2Aligned<T>(positions_ad);
	T acc(0.0);
	for (size_t q=0; q<QSIZE; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		acc += QUADWEIGHTS[q] * potentialVE2<T>(p, normal, L, params);
	}
    const auto grad = acc.grad;
    logger().debug("HighOrderCollision::gradient -> norm={}", grad.norm());
	return grad;
}

auto HighOrderCollision::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	ScalarBase::setVariableCount(N_CORE_DOFS);
	using T = ADHessian<N_CORE_DOFS>;
	Vector<T, N_CORE_DOFS> positions_ad = slice_positions<T, N_CORE_DOFS, 1>(positions);
	Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
	Eigen::Vector2<T> normal;
	T L;
	std::tie(qp, normal, L) = sampleEE2Aligned<T>(positions_ad);
	T acc(0.0);
	for (size_t q=0; q<QSIZE; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		acc += QUADWEIGHTS[q] * potentialVE2<T>(p, normal, L, params);
	}
    const auto hess = acc.Hess;
    logger().debug("HighOrderCollision::hessian -> norm={}", hess.norm());
	return hess;
}

} // namespace ipc
