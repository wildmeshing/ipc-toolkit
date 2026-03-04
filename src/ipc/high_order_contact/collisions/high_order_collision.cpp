#include "high_order_collision.hpp"
#include <ipc/config.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include "alternating_potential_2D.hpp"
#include "ipc/smooth_contact/distance/point_edge.hpp"

namespace ipc
{
    namespace acp = alternating_contact_potential;

// clang-format off
template <> HighOrderCollisionType HighOrderCollisionTemplate<Vertex2, Vertex2>::type() const { return HighOrderCollisionType::VERTEX_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Edge2P1, Vertex2>::type() const { return HighOrderCollisionType::EDGE_VERTEX; }
template <> HighOrderCollisionType HighOrderCollisionTemplate<Edge2P1, Edge2P1>::type() const { return HighOrderCollisionType::EDGE_EDGE; }
    // clang-format on

// clang-format off
template <> std::string HighOrderCollisionTemplate<Vertex2, Vertex2>::name() const { return "vv_2d"; }
template <> std::string HighOrderCollisionTemplate<Edge2P1, Vertex2>::name() const { return "ve_2d"; }
template <> std::string HighOrderCollisionTemplate<Edge2P1, Edge2P1>::name() const { return "ee_2d"; }
    // clang-format on

    std::vector<index_t> HighOrderCollision::vertex_ids() const
    {
        std::vector<index_t> ids;
        ids.reserve(num_vertices());
        for (int i = 0; i < num_vertices(); ++i) {
            ids.push_back(vertex_id(i));
        }
        return ids;
    }

    Eigen::VectorXd HighOrderCollision::dof(Eigen::ConstRef<Eigen::MatrixXd> X) const
    {
        const int DIM = X.cols();
        Eigen::VectorXd x(num_vertices() * DIM);
        if (DIM == 2) {
            for (int i = 0; i < num_vertices(); i++) {
                x.segment<2>(i * 2) = X.row(vertex_id(i));
            }
        } else if (DIM == 3) {
            for (int i = 0; i < num_vertices(); i++) {
                x.segment<3>(i * 3) = X.row(vertex_id(i));
            }
        } else {
            throw std::runtime_error("Invalid dimension!");
        }
        return x;
    }

    Eigen::VectorXd HighOrderCollision::dof(ConcatMatrixView<3> X_extended) const
    {
        Eigen::VectorXd x(num_vertices() * 3);
        for (int i = 0; i < num_vertices(); i++) {
            assert(vertex_id(i) < X_extended.rows());
            x.segment<3>(i * 3) = X_extended(vertex_id(i));
        }
        return x;
    }

    template <typename PrimitiveA, typename PrimitiveB>
    index_t HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::vertex_id(index_t i) const
    {
        if (i < primitive_a.n_vertices()) {
            return primitive_a.vertex_ids()[i];
        }
        else {
            i -= primitive_a.n_vertices();
            assert(primitive_b.n_vertices() > i);
            return primitive_b.vertex_ids()[i];
        }
    }

    template <typename PrimitiveA, typename PrimitiveB>
    HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::HighOrderCollisionTemplate(
        index_t _primitive0,
        index_t _primitive1,
        const CollisionMesh& mesh,
        const HighOrderContactParameters& params,
        const double _dhat,
        const Eigen::MatrixXd& V)
        : primitive_a(_primitive0, mesh, V),
          primitive_b(_primitive1, mesh, V)
    {
        if constexpr (std::is_same_v<PrimitiveA, Edge2P1>) {
            m_area_a = mesh.edge_length(_primitive0);
        }

        if constexpr (std::is_same_v<PrimitiveB, Edge2P1>) {
            m_area_b = mesh.edge_length(_primitive1);
        }

        auto is_obstacle = [&](const auto& primitive) {
            bool any_obstacle = false;
            bool all_obstacle = true;
            for (const index_t vid : primitive.vertex_ids()) {
                if (mesh.is_obstacle_vertex(vid)) {
                    any_obstacle = true;
                }
                else {
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
    }

    template <>
    double HighOrderCollisionTemplate<Vertex2, Vertex2>::compute_distance(
        Eigen::ConstRef<Eigen::MatrixXd> vertices) const
    {
        return point_point_distance(
            vertices.row(vertex_id(0)), vertices.row(vertex_id(n_vertices_a())));
    }

    template <>
    double HighOrderCollisionTemplate<Edge2P1, Vertex2>::compute_distance(
        Eigen::ConstRef<Eigen::MatrixXd> vertices) const
    {
        return point_edge_distance(
            vertices.row(vertex_id(n_vertices_a())), vertices.row(vertex_id(0)),
            vertices.row(vertex_id(1)));
    }

    template <>
    double HighOrderCollisionTemplate<Edge2P1, Edge2P1>::compute_distance(
        Eigen::ConstRef<Eigen::MatrixXd> vertices) const
    {
        const auto& ea0 = vertices.row(vertex_id(0));
        const auto& ea1 = vertices.row(vertex_id(1));
        const auto& eb0 = vertices.row(vertex_id(n_vertices_a()));
        const auto& eb1 = vertices.row(vertex_id(n_vertices_a() + 1));
        return std::min({
            point_edge_distance(ea0, eb0, eb1),
            point_edge_distance(ea1, eb0, eb1),
            point_edge_distance(eb0, ea0, ea1),
            point_edge_distance(eb1, ea0, ea1)
        });
    }

    namespace acp = alternating_contact_potential;


    // ----------------------------------------------------

    template <typename PrimitiveA, typename PrimitiveB>
    double HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::operator()(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        return 0;
    }

    template <typename PrimitiveA, typename PrimitiveB>
    auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::gradient(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
        -> VectorMax<double, ELEMENT_SIZE>
    {
        return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
    }

    template <typename PrimitiveA, typename PrimitiveB>
    auto HighOrderCollisionTemplate<PrimitiveA, PrimitiveB>::hessian(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
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
        log_and_throw_error("Not implemented");
        return 0;
    }

    // ----------------------------------------------------

    template <>
    double HighOrderCollisionTemplate<Edge2P1, Edge2P1>::operator()(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        if (is_obstacle_a()) return 0.0;
        return acp::potential_EE(positions, params, area_a());
    }

    template <>
    double HighOrderCollisionTemplate<Edge2P1, Vertex2>::operator()(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
    {
        if (is_obstacle_a()) return 0.0;
        return acp::potential_EV(positions, params, area_a());
    }


    template <>
    auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::gradient(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
        -> VectorMax<double, ELEMENT_SIZE>
    {
        if (is_obstacle_a()) return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
        return acp::gradient_EE(positions, params, area_a());
    }

    template <>
    auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::gradient(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
        -> VectorMax<double, ELEMENT_SIZE>
    {
        if (is_obstacle_a()) return VectorMax<double, ELEMENT_SIZE>::Zero(n_dofs());
        return acp::gradient_EV(positions, params, area_a());
    }


    template <>
    auto HighOrderCollisionTemplate<Edge2P1, Edge2P1>::hessian(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
        -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
    {
        if (is_obstacle_a()) return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
        return acp::hessian_EE(positions, params, area_a());
    }

    template <>
    auto HighOrderCollisionTemplate<Edge2P1, Vertex2>::hessian(
        Eigen::ConstRef<VectorMax<double, ELEMENT_SIZE>> positions,
        const HighOrderContactParameters& params) const
        -> MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>
    {
        if (is_obstacle_a()) return MatrixMax<double, ELEMENT_SIZE, ELEMENT_SIZE>::Zero(n_dofs(), n_dofs());
        return acp::hessian_EV(positions, params, area_a());
    }

    // ----------------------------------------------------

    // Note: Primitive pair order cannot change
    template class HighOrderCollisionTemplate<Edge2P1, Vertex2>;
    template class HighOrderCollisionTemplate<Vertex2, Vertex2>;
    template class HighOrderCollisionTemplate<Edge2P1, Edge2P1>;
} // namespace ipc
