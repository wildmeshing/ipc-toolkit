#include "ho_smooth_collision.hpp"

#include <ipc/config.hpp>
#include <math.h>

namespace ipc {

// clang-format off
template <> CollisionType HighOrderCollisionTemplate<Point2, Point2>::type() const { return CollisionType::VERTEX_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Point3, Point3>::type() const { return CollisionType::VERTEX_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Edge2, Point2>::type() const { return CollisionType::EDGE_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Edge3, Point3>::type() const { return CollisionType::EDGE_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Face, Point3>::type() const { return CollisionType::FACE_VERTEX; }
template <> CollisionType HighOrderCollisionTemplate<Edge3, Edge3>::type() const { return CollisionType::EDGE_EDGE; }

template <> CollisionType HighOrderCollisionTemplate<Edge2, Edge2>::type() const { return CollisionType::EDGE_EDGE; }
// clang-format on

// clang-format off
template <> std::string HighOrderCollisionTemplate<Point2, Point2>::name() const { return "vert-vert"; }
template <> std::string HighOrderCollisionTemplate<Point3, Point3>::name() const { return "vert-vert"; }
template <> std::string HighOrderCollisionTemplate<Edge2, Point2>::name() const { return "edge-vert"; }
template <> std::string HighOrderCollisionTemplate<Edge3, Point3>::name() const { return "edge-vert"; }
template <> std::string HighOrderCollisionTemplate<Face, Point3>::name() const { return "face-vert"; }
template <> std::string HighOrderCollisionTemplate<Edge3, Edge3>::name() const { return "edge-edge"; }

template <> std::string HighOrderCollisionTemplate<Edge2, Edge2>::name() const { return "edge-edge"; }
// clang-format on

Eigen::VectorXd SmoothCollision::dof(Eigen::ConstRef<Eigen::MatrixXd> X) const
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
    HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::DTYPE dtype,
    const CollisionMesh& mesh,
    const SmoothContactParameters& params,
    const double _dhat,
    const Eigen::MatrixXd& V)
    : SmoothCollision(_primitive0, _primitive1, _dhat, mesh)
{
	// don't need this probably
    VectorMax3d d =
        PrimitiveDistance<PrimitiveA, PrimitiveB>::compute_closest_direction(
            mesh, V, _primitive0, _primitive1, dtype);
    primitive_a = std::make_unique<PrimitiveA>(_primitive0, mesh, V, d, params);
    primitive_b = std::make_unique<PrimitiveB>(_primitive1, mesh, V, -d, params);

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
    m_is_active = (d.norm() < m_dhat) && primitive_a->is_active()
        && primitive_b->is_active();

    if (d.norm() < 1e-12) {
        logger().warn(
            "pair distance {}, id {} and {}, dtype {}, active {}", d.norm(),
            _primitive0, _primitive1,
            PrimitiveDistType<PrimitiveA, PrimitiveB>::NAME, m_is_active);

        logger().warn("value {}", (*this)(this->dof(V), params));
    }
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

template <>
double HighOrderCollisionTemplate<Edge2, Edge2>::operator()(
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

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
{
	return 0;
}

template <>
auto HighOrderCollisionTemplate<Edge2, Edge2>::gradient(
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
template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	return {};
}

template <>
auto HighOrderCollisionTemplate<Edge2, Point2>::hessian(
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
template <typename PrimitiveA, typename PrimitiveB>
auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const SmoothContactParameters& params) const
    -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
	return {};
}

// ---- distance ----

template <typename PrimitiveA, typename PrimitiveB>
double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::compute_distance(
    Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    Vector<double, -1, ELEMENT_SIZE> positions = dof(vertices);

    Vector<double, N_CORE_POINTS * DIM> x;
    x << positions.head(PrimitiveA::N_CORE_POINTS * DIM),
        positions.segment(
            primitive_a->n_dofs(), PrimitiveB::N_CORE_POINTS * DIM);

    return PrimitiveDistanceTemplate<
        PrimitiveA, PrimitiveB, double>::compute_distance(x, DTYPE::AUTO);
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
template class HighOrderCollisionTemplate<Edge2, Point2>;
template class HighOrderCollisionTemplate<Point2, Point2>;

template class HighOrderCollisionTemplate<Edge3, Point3>;
template class HighOrderCollisionTemplate<Edge3, Edge3>;
template class HighOrderCollisionTemplate<Point3, Point3>;
template class HighOrderCollisionTemplate<Face, Point3>;
} // namespace ipc
