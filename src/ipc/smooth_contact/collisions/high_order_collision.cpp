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


namespace {
	constexpr double PI = 3.141592653589793238462643383279502884;
	constexpr int QORD = 3;

	template <typename T>
	T cubic_bspline(T v) {
		T abs_v = ipc::Math<T>::abs(v);
		if (abs_v < 1.0) {
			return (2.0 / 3.0) - abs_v * abs_v + 0.5 * abs_v * abs_v * abs_v;
		} else if (abs_v < 2.0) {
			T t = 2.0 - abs_v;
			return (1.0 / 6.0) * t * t * t;
		}
		return 0.0;
	}

	template <typename T>
	T H_kernel(T z) {
		if (z < -3.0) {
			return 0.0;
		} else if (z < -2.0) {
			T t = 3.0 + z;
			return (1.0 / 6.0) * t * t * t;
		} else if (z < -1.0) {
			return (1.0 / 6.0) * (3.0 - 9.0 * z - 9.0 * z * z - 2.0 * z * z * z);
		} else if (z < 0.0) {
			return 1.0 + (z * z * z) / 6.0;
		}
		return 1.0;
	}

	template <typename T>
	T norm2(T x, T y) { return sqrt(x * x + y * y); }

	template <typename T>
	T cross2(T ax, T ay, T bx, T by) { return ax * by - ay * bx; }

	template <typename T>
	T dot2(T ax, T ay, T bx, T by) { return ax * bx + ay * by; }

	template <typename T>
	T directional_factor(T dx, T dy, T nx, T ny, double alpha) {
		T denom = norm2(dx, dy);
		if (denom == 0.0) {
			return 0.0;
		}
		T phi_m = ipc::Math<T>::abs(cross2(dx, dy, nx, ny)) / denom;
		T phi_e = -dot2(dx, dy, nx, ny) / denom;
		T s = (2.0 / alpha) * phi_m;
		T t = phi_e / alpha;
		T b_val = cubic_bspline(s);
		T h_val = H_kernel(t);
		return (2.0 / alpha) * b_val * h_val;
	}

	template <typename T>
	T integrand_value(T x_param,
						T point_x,
						T point_y,
						T normal_x,
						T normal_y,
						double alpha,
						double epsilon,
						double power) {
		T f0x = x_param;
		T f0y(0.0);
		T dx = point_x - f0x;
		T dy = point_y - f0y;
		T distance = norm2(dx, dy);
		if (distance == 0.0) {
			return 0.0;
		}

		T tangent0_x(1.0);
		T tangent0_y(0.0);
		T normal0_x(0.0);
		T normal0_y(1.0);

		T g_xy = directional_factor(dx, dy, normal_x, normal_y, alpha);
		T g_yx = directional_factor(-dx, -dy, normal0_x, normal0_y, alpha);
		T gamma = g_xy * g_yx;
		if (gamma == 0.0) {
			return 0.0;
		}

		T weight = 1.5 * cubic_bspline((2.0 / epsilon) * distance);
		T numerator = gamma * weight;
		return numerator / pow(distance, power);
	}

	template <typename T>
	bool compute_window(T q0,
						T q1,
						double alpha,
						double theta,
						T& psi_lower,
						T& psi_upper) {
		if (q1 <= 0.0) {
			return false;
		}

		double phi = std::asin(std::min(0.999999, std::max(0.0, alpha)));
		T lower_geom = std::max(0., -theta - (PI / 2.0)) - phi;
		T upper_geom = std::min(0., -theta - (PI / 2.0)) + phi;
		if (lower_geom >= upper_geom) {
			return false;
		}

		T lower_x = atan((q0 - 1.0) / q1);
		T upper_x = atan(q0 / q1);

		psi_lower = (lower_geom > lower_x) ? lower_geom : lower_x;
		psi_upper = (upper_geom < upper_x) ? upper_geom : upper_x;
		if (psi_lower >= psi_upper) {
			return false;
		}
		return true;
	}

	void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights) {
		nodes.resize(n);
		weights.resize(n);
		int m = (n + 1) / 2;
		for (int i = 0; i < m; ++i) {
			double z = std::cos(PI * (i + 0.75) / (n + 0.5));
			double z1;
			double p1, p2;
			do {
				p1 = 1.0;
				p2 = 0.0;
				for (int j = 1; j <= n; ++j) {
					double p3 = p2;
					p2 = p1;
					p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
				}
				double pp = n * (z * p1 - p2) / (z * z - 1.0);
				z1 = z;
				z = z1 - p1 / pp;
			} while (std::fabs(z - z1) > 1e-14);

			nodes[i] = -z;
			nodes[n - 1 - i] = z;
			double pp = n * (z * p1 - p2) / (z * z - 1.0);
			double w = 2.0 / ((1.0 - z * z) * pp * pp);
			weights[i] = weights[n - 1 - i] = w;
		}
	}

	template <typename T>
	T integrate_substitution(T q0,
								T q1,
								int quad_order,
								double epsilon,
								double alpha,
								double power) {
		double theta = -PI / 2.0;
		T psi_lower, psi_upper;
		if (!compute_window(q0, q1, alpha, theta, psi_lower, psi_upper)) {
			return 0.0;
		}

		std::vector<double> nodes;
		std::vector<double> weights;
		gauss_legendre(quad_order, nodes, weights);

		T half = 0.5 * (psi_upper - psi_lower);
		T center = 0.5 * (psi_upper + psi_lower);
		T scaled_sum(0.0);
		for (int i = 0; i < quad_order; ++i) {
			T psi = center + half * nodes[i];
			T cos_psi = cos(psi);
			if (ipc::Math<T>::abs(cos_psi) < 1e-12) {
				continue;
			}
			T w = tan(psi);
			T x_param = (q0 - w * q1);
			if (x_param < 0.0 || x_param > 1.0) {
				continue;
			}
			T value = integrand_value(x_param, q0, q1, T(0.0), T(-1.0), alpha, epsilon, power);
			T scaled_value = (q1 * q1) * value;
			T jac = (q1 / 1.0) / (cos_psi * cos_psi);
			scaled_sum += weights[i] * scaled_value * jac;
		}

		T scaled_integral = half * scaled_sum;
		return scaled_integral / (q1 * q1);
	}
}  // namespace


template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, 2>, std::vector<double>, Eigen::Vector2<T>, T> sampleEE2Aligned(
    Eigen::ConstRef<Vector<T, -1, 8>> positions, int quad_order
){
	// we align A with the x axis and sample B
	const Eigen::Vector<T, 4> A = positions.template head<4>();
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
	const Eigen::Vector<T, 4> B = positions.template tail<4>();
	const Eigen::Vector2<T> B0 = B.template head<2>() - A0;
	const Eigen::Vector2<T> B1 = B.template tail<2>() - A0;
	const Eigen::Vector2<T> B0r = rot.transpose()*B0;
	const Eigen::Vector2<T> B1r = rot.transpose()*B1;
	std::vector<double> nodes;
	std::vector<double> weights;
	gauss_legendre(quad_order, nodes, weights);
	Eigen::Matrix<T, Eigen::Dynamic, 2> M(quad_order, 2);
	for (size_t i = 0; i<quad_order; ++i) {
		const double t = nodes.at(i);
		const Eigen::Vector2<T> P = (1-t)*B0r + t*B1r;
		M.row(i) = P.transpose();
	}
	const Eigen::Vector2<T> Br_vec = B1r - B0r;
	Eigen::Vector2<T> Br_normal(-Br_vec.y(), Br_vec.x());
	Br_normal.normalize();
	return {M, weights, Br_normal, L};
}

double HighOrderCollision::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
{
	Eigen::MatrixX2d qp;
	Eigen::Vector2d normal;
	double L;
	std::vector<double> weights;
	std::tie(qp, weights, normal, L) = sampleEE2Aligned<double>(positions, QORD);
	double acc(0.0);
	for (size_t q=0; q<QORD; ++q) {
		const Eigen::Vector2<double> p = qp.row(q);
		acc += weights[q] * integrate_substitution<double>(p[0], p[1], QORD, params.dhat, params.alpha_t, params.r);
	}
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
	std::vector<double> weights;
	std::tie(qp, weights, normal, L) = sampleEE2Aligned<T>(positions_ad, QORD);
	T acc(0.0);
	for (size_t q=0; q<QORD; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		acc += weights[q] * integrate_substitution<T>(p[0], p[1], QORD, params.dhat, params.alpha_t, params.r);
	}
	return acc.grad;
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
	std::vector<double> weights;
	std::tie(qp, weights, normal, L) = sampleEE2Aligned<T>(positions_ad, QORD);
	T acc(0.0);
	for (size_t q=0; q<QORD; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		acc += weights[q] * integrate_substitution<T>(p[0], p[1], QORD, params.dhat, params.alpha_t, params.r);
	}
	return acc.Hess;
}

} // namespace ipc
