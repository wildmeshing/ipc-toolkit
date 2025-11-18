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
constexpr double QUADWEIGHTS[QSIZE] {0.04761905, 0.27682605, 0.43174538, 0.48761905, 0.43174538, 0.27682605, 0.04761905};
std::pair<Eigen::MatrixX2d, double> sampleEE2Aligned(
    Eigen::ConstRef<Vector<double, -1, 8>> positions
){
	// we align A with the x axis and sample B
	const auto A = positions.head<4>();
	const Eigen::Vector2d A0 = A.head<2>();
	const Eigen::Vector2d A1 = A.tail<2>();
	const Eigen::Vector2d AV = A1 - A0;
	// new reference frame
	const double L = AV.norm();
	const Eigen::Vector2d Anorm = AV / L;
	const Eigen::Vector2d Aorth(-Anorm.y(),Anorm.x());
	Eigen::Matrix2d rot;
	rot.col(0) = Anorm;
	rot.col(1) = Aorth;
	const auto B = positions.tail<4>();
	const Eigen::Vector2d B0 = B.head<2>() - A0;
	const Eigen::Vector2d B1 = B.tail<2>() - A0;
	const Eigen::Vector2d B0r = rot*B0;
	const Eigen::Vector2d B1r = rot*B1;
	Eigen::Matrix<double, Eigen::Dynamic, 2> M(QSIZE, 2);
	for (size_t i; i<QSIZE; ++i) {
		double t = QUADPOINTS[i];
		const Eigen::Vector2d P = (1-t)*B0r + t*B1r;
		M.row(i) = P.transpose();
	}
	return {M, L};
}

double potentialVE2(Eigen::Vector2d xy, double L) {
	const double x = xy.x();
	const double y = xy.y();
	const double iy = 1/y;
	return (std::atan(x*iy) * std::atan((L-x)*iy))*iy;
}
Eigen::Vector2d gradientVE2(Eigen::Vector2d xy, double L) {
	const double x = xy.x();
	const double y = xy.y();
	const double x0 = x*x;
	const double x1 = L - x;
	const double x2 = x1*x1;
	const double x3 = y*y;
	const double x4 = x0 + x3;
	const double x5 = x2 + x3;
	const double x6 = 1/(x4*x5);
	const double x7 = 1/y;
	return {
		x6*(-x0 + x2),
		x6*(-x*x5*y - x1*x4*y - x4*x5*(std::atan(x*x7) + std::atan(x1*x7)))/x3
	};
}
Eigen::Matrix2d hessianVE2(Eigen::Vector2d xy, double L) {
	const double x = xy.x();
	const double y = xy.y();
	const double x0 = y*y;
	const double x1 = x*x + x0;
	const double x2 = x1*x1;
	const double x3 = 1/x2;
	const double x4 = 2*x3;
	const double x5 = L - x;
	const double x6 = x5*x5;
	const double x7 = x0 + x6;
	const double x8 = x7*x7;
	const double x9 = 1/x8;
	const double x10 = 2*x9;
	const double x11 = x4*y;
	const double x12 = y*y*y;
	const double x13 = x6*y;
	const double x14 = 2/(x7*x7*x7);
	const double x15 = 1/y;
	const double x16 = x8*y;
	const double x17 = x2*y;
	const double x18 = x14*x15*x5;
	const double x19 = x17*x5;
	Eigen::Matrix2d res {{
		-L*x10 - x*x4 + 2*x*x9,
		-x11 - x13*x14 - x18*x18*x18*x18 + 2/(x12 + x13),
		},{
		x10*y - x11,
		x10*x3*(-x*x*x*x16 + 2*x*x1*x16 - x19*x19*x19 + 2*x17*x5*x7 + x2*x8*(std::atan(x*x15) + std::atan(x15*x5)))/x12
	}};
	return res;
}

double HighOrderCollision::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
{
	Eigen::MatrixX2d qp;
	double L;
	std::tie(qp, L) = sampleEE2Aligned(positions);
	double acc = 0;
	for (size_t q=0; q<QSIZE; ++q) {
		const auto &p = qp.row(q);
		acc += QUADWEIGHTS[q] * potentialVE2(p, L);
	}
	return acc;
}

auto HighOrderCollision::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	Eigen::MatrixX2d qp;
	double L;
	std::tie(qp, L) = sampleEE2Aligned(positions);
	Eigen::Vector2d acc;
	acc.setZero();
	for (size_t q=0; q<QSIZE; ++q) {
		const auto &p = qp.row(q);
		acc += QUADWEIGHTS[q] * gradientVE2(p, L);
	}
	return acc;
}

auto HighOrderCollision::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	Eigen::MatrixX2d qp;
	double L;
	std::tie(qp, L) = sampleEE2Aligned(positions);
	Eigen::Matrix2d acc;
	acc.setZero();
	for (size_t q=0; q<QSIZE; ++q) {
		const auto &p = qp.row(q);
		acc += QUADWEIGHTS[q] * hessianVE2(p, L);
	}
	return acc;
}

} // namespace ipc
