#include "high_order_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#include "smoothed_offset_potential_linear.h"
#include "high_order_quadrature.hpp"
#include "ipc/smooth_contact/distance/point_edge.hpp"

namespace ipc {

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

    if constexpr (std::is_same_v<PrimitiveA, Edge2P1>) {
        m_area_a = mesh.edge_length(_primitive0);
    }

    if constexpr (std::is_same_v<PrimitiveB, Edge2P1>) {
        m_area_b = mesh.edge_length(_primitive1);
    }
        
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


template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, 2>, std::vector<double>, Eigen::Vector2<T>> sample_edge(
    Eigen::ConstRef<Eigen::Matrix<T, 2, 2>> edge_positions, int quad_order, std::array<T, 2> window={{0.0, 1.0}}
){
    const Eigen::Vector2<T> p0 = edge_positions.row(0);
	const Eigen::Vector2<T> p1 = edge_positions.row(1);
	Eigen::Vector2<T> edge_vec = p1 - p0;
	edge_vec.normalize();
	const Eigen::Vector2<T> edge_normal(-edge_vec.y(), edge_vec.x());

	Eigen::Matrix<T, Eigen::Dynamic, 2> M(quad_order, 2);
	if (window[0] < 0.0 || window[1] > 1.0 || window[1] < window[0]) {
        std::stringstream ss;
        ss << "Invalid window: " << window[0] << ',' << window[1] << "!";
        throw std::runtime_error(ss.str());
	}

    std::vector<double> nodes, weights;
    std::tie(nodes, weights) = GaussLobatto::get_rule(quad_order);

    const T center = (window[0] + window[1]) / 2;
    const T halfw = (window[1] - window[0]) / 2;
	for (size_t i = 0; i<quad_order; ++i) {
        const T t = center + halfw * nodes.at(i);
		const Eigen::Vector2<T> P = ((1-t) * p0 + t * p1);
		M.row(i) = P.transpose();
	}

	return {M, weights, edge_normal};
}

// ----------------------------------------------------

template <typename T>
T potential_VV_onesided(
    Eigen::ConstRef<Eigen::Matrix<T, Eigen::Dynamic, 2>> v_a,
    Eigen::ConstRef<Eigen::Matrix<T, Eigen::Dynamic, 2>> v_b,
    const HighOrderContactParameters& params)
{
    const std::array<T, 2> query_point = {{ v_b(0, 0), v_b(0, 1) }}; // Query point (Vertex B)
    const std::array<T, 2> vertex_pt = {{ v_a(0, 0), v_a(0, 1) }}; // Source vertex (Vertex A)

    T phi_start_next_val;
    T phi_end_prev_val;
    const T* phi_start_next = nullptr;
    const T* phi_end_prev = nullptr;

    Eigen::Vector2<T> tangent_next, normal_next = Eigen::Vector2<T>::Zero();
    std::array<T, 2> v1_arr;

    const Eigen::Vector2<T> p0 = v_a.row(0);
    if (v_a.rows() >= 2) {
        const Eigen::Vector2<T> p_next = v_a.row(1);
        tangent_next = (p_next - p0).normalized();
        normal_next << tangent_next.y(), -tangent_next.x();
        v1_arr = {{-tangent_next.x(), -tangent_next.y()}};
    }

    Eigen::Vector2<T> tangent_prev, normal_prev = Eigen::Vector2<T>::Zero();
    T p_prev_norm = 0;
    std::array<T, 2> v2_arr;

    if (v_a.rows() >= 3) {
        const Eigen::Vector2<T> p_prev = v_a.row(2);
        const Eigen::Vector2<T> edge_prev = p0 - p_prev;
        p_prev_norm = edge_prev.norm();
        tangent_prev = edge_prev / p_prev_norm;
        normal_prev << tangent_prev.y(), -tangent_prev.x();
        v2_arr = {{tangent_prev.x(), tangent_prev.y()}};
    }

    if (v_a.rows() >= 2) {
        const std::array<T, 2> rel_next = {{ query_point[0] - v_a(0, 0), query_point[1] - v_a(0, 1) }};
        const T r_q_next = rel_next[0] * normal_next(0) + rel_next[1] * normal_next(1);
        const T y_q_next = rel_next[0] * tangent_next(0) + rel_next[1] * tangent_next(1);
        phi_start_next_val = smoothed_offset_potential::phi_value(r_q_next, y_q_next, T(0));
        phi_start_next = &phi_start_next_val;
    }
    if (v_a.rows() >= 3) {
        const std::array<T, 2> rel_prev = {{ query_point[0] - v_a(2, 0), query_point[1] - v_a(2, 1) }};
        const T r_q_prev = rel_prev[0] * normal_prev(0) + rel_prev[1] * normal_prev(1);
        const T y_q_prev = rel_prev[0] * tangent_prev(0) + rel_prev[1] * tangent_prev(1);
        phi_end_prev_val = smoothed_offset_potential::phi_value(r_q_prev, y_q_prev, p_prev_norm);
        phi_end_prev = &phi_end_prev_val;
    }
    return smoothed_offset_potential::polyline_vertex_potential(
        query_point, vertex_pt, phi_start_next, phi_end_prev, params.alpha, params.r, params.dhat);
}

template <typename T>
T potential_VV(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>>
        positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b)
{
    if (params.quad_points != 0) throw std::logic_error("Quad points > 0 in potential_VV");
    Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
        slice_positions<T, Eigen::Dynamic, 2>(positions);
    const Eigen::Matrix<T, Eigen::Dynamic, 2> v_a = all_pos.topRows(n_vertices_a);
    const Eigen::Matrix<T, Eigen::Dynamic, 2> v_b = all_pos.bottomRows(n_vertices_b);

    return potential_VV_onesided<T>(v_a, v_b, params)
         + potential_VV_onesided<T>(v_b, v_a, params);
}

// ----------------------------------------------------

template <typename T>
T potential_VE(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>>
        positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b)
{
    if (params.quad_points != 0) throw std::logic_error("Quad points > 0 in potential_VE");
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
    return smoothed_offset_potential::polyline_edge_potential(
        vertex_pt, p0_arr, t_arr, n_arr, len,
        params.alpha, params.r, params.dhat,
        phi_start, phi_end);
}

// ----------------------------------------------------
namespace alternating_contact_potential {
    template <typename T>
    T distance_VE(
        const Eigen::Vector2<T> &e0,
        const Eigen::Vector2<T> &e1,
        const Eigen::Vector2<T> &v0
    ) {
        const Eigen::Vector2<T> edge = e1 - e0;
        const T length = edge.norm();
        const Eigen::Vector2<T> tangent = edge / length;
        const Eigen::Vector2<T> vec = v0 - e0;
        const T proj = vec.dot(tangent);

        if (proj < 0) return vec.norm();
        if (proj > length) return (v0 - e1).norm();

        const Eigen::Vector2<T> normal(-tangent.y(), tangent.x());
        using namespace std;
        using namespace TinyAD;
        return abs(vec.dot(normal));
    }

    template <typename T>
    T barrier_func(
        const T d,
        const HighOrderContactParameters& params
    ) {
        const T denom = (abs(pow(d, params.r)));
        if (denom <= 1e-12) return T(0);
        return smoothed_offset_potential::h_epsilon(abs(d), params.dhat) / denom;
    }

    template <typename T>
    T potential_EV(
        Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params,
        const double integration_area = -1.0
    )
    {
        Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
            slice_positions<T, Eigen::Dynamic, 2>(positions);
        const Eigen::Vector2<T> e0 = all_pos.row(0);
        const Eigen::Vector2<T> e1 = all_pos.row(1);
        const Eigen::Vector2<T> v0 = all_pos.row(2);

        int qord = params.quad_points;
        if (qord < 2) {
            qord = 2;
        }

        std::vector<double> nodes, weights;
        std::tie(nodes, weights) = GaussLobatto::get_rule(qord);

        T integral = 0.0;
        for (int i = 0; i < qord; ++i) {
            const double t_d = (nodes[i] + 1.0) / 2.0;
            const T t(t_d);
            const Eigen::Vector2<T> p = (1.0 - t) * e0 + t * e1;
            integral += weights[i] * barrier_func((p - v0).norm(), params);
        }

        const T length = (integration_area < 0) ? ((e0 - e1).norm()) : integration_area;
        return -0.5 * length * integral;
    }

    template <typename T>
    T potential_EE_onesided(
        const Eigen::Vector2<T>& e0,
        const Eigen::Vector2<T>& e1,
        const Eigen::Vector2<T>& other0,
        const Eigen::Vector2<T>& other1,
        const HighOrderContactParameters& params,
        const double integration_area
    )
    {
        int qord = params.quad_points;
        if (qord < 2) {
            qord = 2;
        }

        std::vector<double> nodes, weights;
        std::tie(nodes, weights) = GaussLobatto::get_rule(qord);

        T integral = 0.0;
        for (int i = 0; i < qord; ++i) {
            const double t_d = (nodes[i] + 1.0) / 2.0;
            const T t(t_d);
            const Eigen::Vector2<T> p = (1.0 - t) * e0 + t * e1;
            integral += weights[i] * barrier_func(distance_VE(other0, other1, p), params);
        }

        const T length = (integration_area < 0) ? ((e0 - e1).norm()) : integration_area;
        return 0.5 * length * integral;
    }
    template <typename T>
    T potential_EE(
        Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params,
        const bool is_obstacleA,
        const bool is_obstacleB,
        const double integration_areaA = -1.0,
        const double integration_areaB = -1.0
    ) {
        const Eigen::Matrix<T, 4, 2> all_pos = slice_positions<T, 4, 2>(positions);
        const Eigen::Vector2<T> ea0 = all_pos.row(0);
        const Eigen::Vector2<T> ea1 = all_pos.row(1);
        const Eigen::Vector2<T> eb0 = all_pos.row(2);
        const Eigen::Vector2<T> eb1 = all_pos.row(3);

        T pot = 0.0;
        if (!is_obstacleA) { // integrate on primitive A
            pot += potential_EE_onesided(ea0, ea1, eb0, eb1, params, integration_areaA);
        }
        if (!is_obstacleB) { // integrate on primitive B
            pot += potential_EE_onesided(eb0, eb1, ea0, ea1, params, integration_areaB);
        }
        return pot;
    }
} // namespace alternating_contact_potential
// ----------------------------------------------------

template <typename T>
T potential_EV(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>>
        positions,
    const HighOrderContactParameters& params,
    const size_t n_vertices_a,
    const size_t n_vertices_b,
    const double integration_area = -1.0
)
{
    if (params.alpha == 0) return alternating_contact_potential::potential_EV<T>(positions, params, integration_area);
    // No integration
    if (params.quad_points == 0) return potential_VE<T>(positions, params, n_vertices_a, n_vertices_b);

    Eigen::Matrix<T, Eigen::Dynamic, 2> all_pos =
        slice_positions<T, Eigen::Dynamic, 2>(positions);
    const Eigen::Matrix<T, 2, 2> edge_pos = all_pos.topRows(n_vertices_a);
    const Eigen::Matrix<T, Eigen::Dynamic, 2> vertex_stencil = all_pos.bottomRows(n_vertices_b);

    const std::array<T, 2> vertex_pt = {{ vertex_stencil(0, 0), vertex_stencil(0, 1) }};

    T phi_start_next_val;
    T phi_end_prev_val;
    const T* phi_start_next = nullptr;
    const T* phi_end_prev = nullptr;

    if (vertex_stencil.rows() == 2) {
        throw std::logic_error("Open 2D polylines are not supported yet. Make sure that every vertex has two neighbors.");
    }

    Eigen::Vector2<T> tangent_next, normal_next = Eigen::Vector2<T>::Zero();
    std::array<T, 2> v1_arr;
    const std::array<T, 2>* v1_ptr = nullptr;

    const Eigen::Vector2<T> p0 = vertex_stencil.row(0);
    if (vertex_stencil.rows() >= 2) {
        const Eigen::Vector2<T> p_next = vertex_stencil.row(1);
        tangent_next = (p_next - p0).normalized();
        normal_next << tangent_next.y(), -tangent_next.x();
        v1_arr = {{-tangent_next.x(), -tangent_next.y()}};
        v1_ptr = &v1_arr;
    }

    Eigen::Vector2<T> tangent_prev, normal_prev = Eigen::Vector2<T>::Zero();
    T p_prev_norm = 0;
    std::array<T, 2> v2_arr;
    const std::array<T, 2>* v2_ptr = nullptr;

    if (vertex_stencil.rows() >= 3) {
        const Eigen::Vector2<T> p_prev = vertex_stencil.row(2);
        const Eigen::Vector2<T> edge_prev = p0 - p_prev;
        p_prev_norm = edge_prev.norm();
        tangent_prev = edge_prev / p_prev_norm;
        normal_prev << tangent_prev.y(), -tangent_prev.x();
        v2_arr = {{tangent_prev.x(), tangent_prev.y()}};
        v2_ptr = &v2_arr;
    }

    const std::array<T, 2> edge_p0 = {{edge_pos(0, 0), edge_pos(0, 1)}};
    const std::array<T, 2> edge_p1 = {{edge_pos(1, 0), edge_pos(1, 1)}};

    std::array<T, 2> window = smoothed_offset_potential::compute_vertex_window(
        edge_p0, edge_p1, vertex_pt, v1_ptr, v2_ptr, params.alpha, &params.dhat);
    if (window[0] == 1.0 && window[1] == 0.0) return T(0);

    Eigen::Matrix<T, Eigen::Dynamic, 2> qp;
    std::vector<double> weights;
    Eigen::Vector2<T> normal;
    const int qord = params.quad_points;

    // Sample points on the edge
    std::tie(qp, weights, normal) = sample_edge<T>(edge_pos, qord, window);
    const T scale = .5 * (edge_pos.row(1) - edge_pos.row(0)).norm() * (window[1] - window[0]);
    T acc(0.0);
    for (size_t q = 0; q < qord; ++q) {
        const std::array<T, 2> query_point = {{ qp(q, 0), qp(q, 1) }};

        if (vertex_stencil.rows() >= 2) {
            const std::array<T, 2> rel_next = {{ query_point[0] - vertex_stencil(0, 0), query_point[1] - vertex_stencil(0, 1) }};
            const T r_q_next = rel_next[0] * normal_next(0) + rel_next[1] * normal_next(1);
            const T y_q_next = rel_next[0] * tangent_next(0) + rel_next[1] * tangent_next(1);
            phi_start_next_val = smoothed_offset_potential::phi_value(r_q_next, y_q_next, T(0));
            phi_start_next = &phi_start_next_val;
        }
        if (vertex_stencil.rows() >= 3) {
            const std::array<T, 2> rel_prev = {{ query_point[0] - vertex_stencil(2, 0), query_point[1] - vertex_stencil(2, 1) }};
            const T r_q_prev = rel_prev[0] * normal_prev(0) + rel_prev[1] * normal_prev(1);
            const T y_q_prev = rel_prev[0] * tangent_prev(0) + rel_prev[1] * tangent_prev(1);
            phi_end_prev_val = smoothed_offset_potential::phi_value(r_q_prev, y_q_prev, p_prev_norm);
            phi_end_prev = &phi_end_prev_val;
        }
        acc += weights[q] * smoothed_offset_potential::polyline_vertex_potential(
            query_point, vertex_pt, phi_start_next, phi_end_prev, params.alpha, params.r, params.dhat);
    }
    return scale * acc;
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
	const Eigen::Vector2<T> p0 = edge0_pos.row(1);
	const Eigen::Vector2<T> p1 = edge0_pos.row(0);
	const Eigen::Vector2<T> tangent_vec = p1 - p0;
	const T length = tangent_vec.norm();
	const Eigen::Vector2<T> tangent = tangent_vec / length;
	const Eigen::Vector2<T> normal_vec(-tangent.y(), tangent.x());

	const std::array<T, 2> p0_arr{{ p0(0), p0(1) }};
    const std::array<T, 2> p1_arr{{ p1(0), p1(1) }};
	const std::array<T, 2> tangent_arr{{ tangent(0), tangent(1) }};
	const std::array<T, 2> normal_arr{{ normal_vec(0), normal_vec(1) }};

    const Eigen::Vector2<T> ep0 = edge1_pos.row(0);
    const Eigen::Vector2<T> ep1 = edge1_pos.row(1);
    const std::array<T, 2> ep0_arr{{ ep0(0), ep0(1) }};
    const std::array<T, 2> ep1_arr{{ ep1(0), ep1(1) }};

    std::array<T, 2> window = smoothed_offset_potential::compute_edge_window(
        ep0_arr, ep1_arr, p0_arr, p1_arr, params.alpha, &params.dhat);
    if (window[0] == 1.0 && window[1] == 0.0) return T(0);

	// "sampled_segment" is the segment we integrate over (edge1)
	std::tie(qp, weights, normal) = sample_edge<T>(edge1_pos, qord, window);
	const T scale = .5 * (edge1_pos.row(1) - edge1_pos.row(0)).norm() * (window[1] - window[0]);
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
				params.alpha,
				params.r,
				params.dhat,
				phi_start,
				phi_end);
	}
	return scale*acc;
}

template <typename T>
T potential_EE(
    Eigen::ConstRef<Vector<double, -1, HighOrderCollision::ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params,
    const bool is_obstacleA,
    const bool is_obstacleB,
    const double integration_areaA = -1.0,
    const double integration_areaB = -1.0
) {
    if (params.alpha == 0) return alternating_contact_potential::potential_EE<T>(
        positions, params, is_obstacleA, is_obstacleB, integration_areaA, integration_areaB);
    if (params.quad_points == 0) throw std::logic_error("Quad points = 0 in potential_EE");
	Eigen::Matrix<T, 4, 2> all_pos = slice_positions<T, 4, 2>(positions);
    Eigen::Matrix<T, 2, 2> edge0_pos = all_pos.topRows(2);
	Eigen::Matrix<T, 2, 2> edge1_pos = all_pos.bottomRows(2);
    return (potential_EE_onesided<T>(edge0_pos, edge1_pos, params)
		+ potential_EE_onesided<T>(edge1_pos, edge0_pos, params));
}

// ----------------------------------------------------

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
    const HighOrderContactParameters& params) const
{
    return potential_EE<double>(positions, params,
        is_obstacle_a(), is_obstacle_b(),
        area_a(), area_b()
    );
}

template <>
double HighOrderCollisionTemplate<Edge2P1, Vertex2>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    return is_obstacle_a() ? 0.0 : potential_EV<double>(
        positions, params, primitive_a->n_vertices(), primitive_b->n_vertices(), area_a());
}

template <>
double HighOrderCollisionTemplate<Vertex2, Vertex2>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    return potential_VV<double>(
        positions, params, primitive_a->n_vertices(), primitive_b->n_vertices());
}

template <>
double HighOrderCollisionTemplate<Vertex3, Vertex3>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = (positions.template head<3>() - positions.template segment<3>(3)).norm();
    return Math<double>::inv_barrier(dist / params.dhat, params.r);
}

template <>
double HighOrderCollisionTemplate<Edge3P1, Vertex3>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = point_edge_distance(
        positions.template segment<3>(6),
        positions.template head<3>(),
        positions.template segment<3>(3));
    return Math<double>::inv_barrier(sqrt(dist) / params.dhat, params.r);
}

template <>
double HighOrderCollisionTemplate<Face3P1, Vertex3>::operator()(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
{
    const double dist = point_triangle_distance(
        positions.template segment<3>(9),
        positions.template head<3>(),
        positions.template segment<3>(3),
        positions.template segment<3>(6));
    return Math<double>::inv_barrier(sqrt(dist) / params.dhat, params.r);
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
	return potential_EE<ADGrad<N_CORE_DOFS>>(positions, params,
        is_obstacle_a(), is_obstacle_b(),
        area_a(), area_b()
    ).grad;
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const -> Vector<double, -1, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    if (is_obstacle_a()) return Vector<double, -1, ELEMENT_SIZE>::Zero(n_dofs());
    return potential_EV<ADGrad<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices(), area_a())
        .grad;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Vertex2>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const -> Vector<double, -1, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_VV<ADGrad<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices()).grad;
}

template <>
auto HighOrderCollisionTemplate<Vertex3, Vertex3>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
{
    assert(positions.size() == 6);
    const double dist = (positions.template head<3>() - positions.template tail<3>()).norm();
    double deriv = Math<double>::inv_barrier_grad(dist / params.dhat, params.r);
    deriv *= 1. / params.dhat / dist / 2.;

    Vector6d grad = deriv * point_point_distance_gradient(positions.template head<3>(), positions.template tail<3>());

    return grad;
}

template <>
auto HighOrderCollisionTemplate<Edge3P1, Vertex3>::gradient(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
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

    double deriv = Math<double>::inv_barrier_grad(dist / params.dhat, params.r);
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
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const
    -> Vector<double, -1, ELEMENT_SIZE>
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

    double deriv = Math<double>::inv_barrier_grad(dist / params.dhat, params.r);
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
	return potential_EE<ADHessian<N_CORE_DOFS>>(positions, params,
        is_obstacle_a(), is_obstacle_b(),
        area_a(), area_b()
    ).Hess;
}

template <>
auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    if (is_obstacle_a()) return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
    return potential_EV<ADHessian<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices(), area_a())
        .Hess;
}

template <>
auto HighOrderCollisionTemplate<Vertex2, Vertex2>::hessian(
    Eigen::ConstRef<Vector<double, -1, ELEMENT_SIZE>> positions,
    const HighOrderContactParameters& params) const -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
{
    ScalarBase::setVariableCount(positions.rows());
    return potential_VV<ADHessian<-1>>(
               positions, params, primitive_a->n_vertices(), primitive_b->n_vertices()).Hess;
}

// Note: Primitive pair order cannot change
template class HighOrderCollisionTemplate<Edge2P1, Vertex2>;
template class HighOrderCollisionTemplate<Vertex2, Vertex2>;
template class HighOrderCollisionTemplate<Edge2P1, Edge2P1>;

template class HighOrderCollisionTemplate<Vertex3, Vertex3>;
template class HighOrderCollisionTemplate<Edge3P1, Vertex3>;
template class HighOrderCollisionTemplate<Face3P1, Vertex3>;

} // namespace ipc