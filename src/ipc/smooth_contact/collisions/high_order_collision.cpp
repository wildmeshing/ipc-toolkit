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


namespace {
	constexpr double PI = 3.141592653589793238462643383279502884;
	constexpr int QORD = 11;

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
						T L,
						double alpha,
						T theta,
						T& psi_lower,
						T& psi_upper) {
		if (q1 <= 0.0) {
			std::cout << "q1 <= 0.0" << std::endl;
			return false;
		}

		double phi = std::asin(std::min(0.999999, std::max(0.0, alpha)));
		const T th = -theta - (PI / 2.0);
		T lower_geom = (th > 0 ? th : 0) - phi;
		T upper_geom = (th < 0 ? th : 0) + phi;
		//T lower_geom = std::max(-phi, -theta - (PI / 2.0) - phi);
		//T upper_geom = std::min(+phi, -theta - (PI / 2.0) + phi);
		std::cout << "psi range (geom): [" << lower_geom << ", " << upper_geom << "]" << std::endl;
		if (lower_geom >= upper_geom) {
			std::cout << "lower_geom >= upper_geom" << std::endl;
			return false;
		}

		// @federico added L
		// T lower_x = atan((q0 - 1.0) / q1);
		T lower_x = atan((q0 - L) / q1);
		T upper_x = atan(q0 / q1);
		std::cout << "psi range (x): [" << lower_x << ", " << upper_x << "]" << std::endl;
		if (lower_x >= upper_x) {
			std::cout << "lower_x >= upper_x" << std::endl;
			return false;
		}

		psi_lower = (lower_geom > lower_x) ? lower_geom : lower_x; //max
		psi_upper = (upper_geom < upper_x) ? upper_geom : upper_x; //min

		std::cout << "psi range (final): [" << psi_lower << ", " << psi_upper << "]" << std::endl;
		if (psi_lower >= psi_upper) {
			std::cout << "psi_lower >= psi_upper" << std::endl;
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
	T integrate_substitution(
		T q0,
		T q1,
		T n0,
		T n1,
		T L,
		int quad_order,
		double epsilon,
		double alpha,
		double power) {
		const T theta = atan2(n1, n0);
		T psi_lower, psi_upper;
		if (!compute_window(q0, q1, L, alpha, theta, psi_lower, psi_upper)) {
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
			// @federico added normal
			//T value = integrand_value(x_param, q0, q1, T(0.0), T(-1.0), alpha, epsilon, power);
			T value = integrand_value(x_param, q0, q1, n0, n1, alpha, epsilon, power);
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
    Eigen::ConstRef<Eigen::Matrix<T, 4, 2>> positions, int quad_order
){
	// we align A with the x axis and sample B
	const Eigen::Vector2<T> A0 = positions.row(0);
	const Eigen::Vector2<T> A1 = positions.row(1);
	const Eigen::Vector2<T> AV = A1 - A0;
	// new reference frame
	const T L = AV.norm();
	if (L<=0.0) {
		std::stringstream ss;
		ss << "norm is wrong " << A0(0) << "," << A0(1) << " " << A1(0) << "," << A1(1) << " " << AV(0) << "," << AV(1) << " " << L;
		throw std::logic_error(ss.str());
	}
	const Eigen::Vector2<T> AT = AV / L;
	const Eigen::Vector2<T> AN(-AT.y(),AT.x());
	Eigen::Matrix<T, 2, 2> rot;
	rot.row(0) = AT;
	rot.row(1) = AN;
	const Eigen::Vector2<T> b0 = positions.row(2);
	const Eigen::Vector2<T> b1 = positions.row(3);
	const Eigen::Vector2<T> B0 = b0 - A0;
	const Eigen::Vector2<T> B1 = b1 - A0;
	if(abs(rot.determinant() - 1.0) > 1e-5) throw std::logic_error("rotation is wrong (Det)");
	if(((rot * AV) - Eigen::Vector2<T>(L, 0)).norm() > 1e-5) throw std::logic_error("rotation is wrong (orient)");
	const Eigen::Vector2<T> B0r = rot*B0;
	const Eigen::Vector2<T> B1r = rot*B1;
	std::vector<double> nodes;
	std::vector<double> weights;
	gauss_legendre(quad_order, nodes, weights);
	Eigen::Matrix<T, Eigen::Dynamic, 2> M(quad_order, 2);
	for (size_t i = 0; i<quad_order; ++i) {
		const double t = (nodes.at(i)+1)/2;
		const Eigen::Vector2<T> P = ((1-t) * B0r + t * B1r);
		M.row(i) = P.transpose();
	}
	Eigen::Vector2<T> Br_vec = B1r - B0r;
	Br_vec.normalize();
	const Eigen::Vector2<T> Br_normal(-Br_vec.y(), Br_vec.x());
	/*std::stringstream ss;
	if constexpr (std::is_same<T, double>::value) {}
	else{
		ss <<
			"A0 " << A0(0).val << ' ' << A0(1).val << '\n' <<
			"A1 " << A1(0).val << ' ' << A1(1).val << '\n' <<
			"AV " << AV(0).val << ' ' << AV(1).val << '\n' <<
			"B0 " << B0(0).val << ' ' << B0(1).val << '\n' <<
			"B1 " << B1(0).val << ' ' << B1(1).val << '\n' <<
			"B0r " << B0r(0).val << ' ' << B0r(1).val << '\n' <<
			"B1r " << B1r(0).val << ' ' << B1r(1).val << '\n' <<
			"BrN " << Br_normal(0).val << ' ' << Br_normal(1).val << '\n' << '\n';
	}
	std::cout << ss.str();*/
	return {M, weights, Br_normal, L};
}

template <typename T>
T potential_onesided(
	Eigen::ConstRef<Eigen::Matrix<T, 4, 2>> positions,
    const SmoothContactParameters& params
) {
	ScalarBase::setVariableCount(HighOrderCollision::N_CORE_DOFS);
	Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
	Eigen::Vector2<T> normal;
	T L;
	std::vector<double> weights;
	std::tie(qp, weights, normal, L) = sampleEE2Aligned<T>(positions, QORD);
	T acc(0.0);
	std::cout << "Positions:\n" << positions << std::endl;
	std::cout << "Normal: " << normal << std::endl;
	for (size_t q=0; q<QORD; ++q) {
		const Eigen::Vector2<T> p = qp.row(q);
		acc += weights[q] * integrate_substitution<T>(p(0), p(1), normal(0), normal(1), L, QORD, params.dhat, params.alpha_t, params.r);
		std::cout << q << " (" << p(0) << ", " << p(1) << ")" << " acc:" << acc << std::endl;
	}
	return acc;
}

double HighOrderCollision::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
{
	Eigen::Matrix<double, 4, 2> positions_ad = slice_positions<double, 4, 2>(positions);
	Eigen::Matrix<double, 4, 2> positions_ad_rev;
	positions_ad_rev.row(0) = positions_ad.row(2);
	positions_ad_rev.row(1) = positions_ad.row(3);
	positions_ad_rev.row(2) = positions_ad.row(0);
	positions_ad_rev.row(3) = positions_ad.row(1);
	double acc = potential_onesided<double>(positions_ad, params) + potential_onesided<double>(positions_ad_rev, params);
	return acc;
}

auto HighOrderCollision::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	using T = ADGrad<N_CORE_DOFS>;
	Eigen::Matrix<T, 4, 2> positions_ad = slice_positions<T, 4, 2>(positions);
	Eigen::Matrix<T, 4, 2> positions_ad_rev;
	positions_ad_rev.row(0) = positions_ad.row(2);
	positions_ad_rev.row(1) = positions_ad.row(3);
	positions_ad_rev.row(2) = positions_ad.row(0);
	positions_ad_rev.row(3) = positions_ad.row(1);
	T acc = potential_onesided<T>(positions_ad, params)
		+ potential_onesided<T>(positions_ad_rev, params);
	return acc.grad;
}

auto HighOrderCollision::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	using T = ADHessian<N_CORE_DOFS>;
	Eigen::Matrix<T, 4, 2> positions_ad = slice_positions<T, 4, 2>(positions);
	Eigen::Matrix<T, 4, 2> positions_ad_rev;
	positions_ad_rev.row(0) = positions_ad.row(2);
	positions_ad_rev.row(1) = positions_ad.row(3);
	positions_ad_rev.row(2) = positions_ad.row(0);
	positions_ad_rev.row(3) = positions_ad.row(1);
	T acc = potential_onesided<T>(positions_ad, params)
		+ potential_onesided<T>(positions_ad_rev, params);
	return acc.Hess;
}

} // namespace ipc
