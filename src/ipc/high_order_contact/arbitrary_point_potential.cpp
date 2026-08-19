#include "arbitrary_point_potential.hpp"

#include <ipc/candidates/edge_vertex.hpp>
#include <ipc/candidates/face_vertex.hpp>
#include <ipc/high_order_contact/collisions/high_order_collision_template.hpp>
#include <ipc/high_order_contact/collisions/vertex_matrix_view.hpp>
#include <ipc/high_order_contact/high_order_collisions_builder.hpp>
#include <ipc/utils/unordered_map_and_set.hpp>

namespace ipc {

namespace {

    // Mirrors the local insert_pair() helper in quadrature_potential.cpp:
    // merges collisions that resolve to the same underlying feature (same
    // typed hash) by accumulating their weight, and drops the entry
    // entirely if the accumulated weight is exactly zero. This is the
    // symbolic-cancellation step: redundant +1/-1 contributions from
    // different primitives converging on the same feature (e.g. two
    // adjacent faces and their shared edge) cancel here, as integers,
    // before any barrier value is ever computed -- never as a
    // floating-point subtraction of two independently-rounded nearly-equal
    // barrier values (which is numerically unstable near dhat -> 0, where
    // the barrier and its derivatives blow up).
    template <typename KeyType, typename ValueType>
    void
    insert_pair(unordered_map<KeyType, ValueType>& map, ValueType&& collision)
    {
        if (auto iter = map.find(collision->get_typed_hash());
            iter != map.end()) {
            iter->second->weight += collision->weight;
            if (iter->second->weight == 0) {
                map.erase(iter);
            }
        } else {
            map[collision->get_typed_hash()] = std::move(collision);
        }
    }

} // namespace

ArbitraryPointPotential::ArbitraryPointPotential(
    const CollisionMesh& _mesh, HighOrderContactParameters _params)
    : mesh(_mesh)
    , params(std::move(_params))
{
}

void ArbitraryPointPotential::update(Eigen::ConstRef<Eigen::MatrixXd> V)
{
    point_bvh.update(V, mesh);
}

std::unique_ptr<HighOrderCollisionDict<PointType::VERTEX>>
ArbitraryPointPotential::build_collisions_at_point(
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::RowVector3d> q) const
{
    const index_t vid = static_cast<index_t>(V.rows()); // virtual vertex id
    const VertexMatrixView<3> V_view(V, q);

    std::vector<index_t> vertex_ids, edge_ids, face_ids;
    point_bvh.query_point(q, params.dhat, vertex_ids, edge_ids, face_ids);

    unordered_map<std::array<index_t, 3>, std::shared_ptr<HighOrderCollision>>
        pairs;

    // Face candidates: reduce_point_triangle_collision classifies the exact
    // closest feature (interior / edge / corner) via exact distance-type
    // predicates and returns the correspondingly-typed collision -- never
    // just "this face's interior" regardless of where q's closest point on
    // it actually falls. Weight stays at the class default (+1).
    for (index_t fi : face_ids) {
        if (std::shared_ptr<HighOrderCollision> pair =
                HighOrderCollisionsBuilder<3>::reduce_point_triangle_collision(
                    FaceVertexCandidate(fi, vid), params, mesh, V_view)) {
            insert_pair(pairs, std::move(pair));
        }
    }
    // Edge candidates: weight -1, matching the alternating-sum sign
    // convention. May merge (and symbolically cancel) with an edge-typed
    // collision produced by the face loop above when both resolve to the
    // same edge.
    for (index_t ei : edge_ids) {
        if (std::shared_ptr<HighOrderCollision> pair =
                HighOrderCollisionsBuilder<3>::reduce_point_edge_collision(
                    EdgeVertexCandidate(ei, vid), params, mesh, V_view)) {
            pair->weight = -1;
            insert_pair(pairs, std::move(pair));
        }
    }
    // Vertex candidates: weight +1, direct point-point pairs.
    for (index_t vi : vertex_ids) {
        if ((V.row(vi) - q).squaredNorm() >= params.dhat * params.dhat) {
            continue;
        }
        std::shared_ptr<HighOrderCollision> pair =
            std::make_shared<HighOrderCollisionTemplate<Vertex3, Vertex3>>(
                vid, vi, mesh);
        insert_pair(pairs, std::move(pair));
    }

    auto collisions =
        std::make_unique<HighOrderCollisionDict<PointType::VERTEX>>();
    collisions->initialize(
        std::vector<index_t> { vid }, std::vector<index_t> { vid }, pairs);
    return collisions;
}

double ArbitraryPointPotential::operator()(
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::RowVector3d> q) const
{
    const auto collisions = build_collisions_at_point(V, q);
    const VertexMatrixView<3> V_view(V, q);

    double value = 0.0;
    for (int ci = 0; ci < collisions->size(); ci++) {
        const auto& cc = (*collisions)[ci];
        value += cc.weight * cc(cc.dof(V_view), params, /*adaptive=*/nullptr);
    }
    return value;
}

Eigen::Vector3d ArbitraryPointPotential::gradient(
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::RowVector3d> q) const
{
    const auto collisions = build_collisions_at_point(V, q);
    const VertexMatrixView<3> V_view(V, q);
    const index_t vid = static_cast<index_t>(V.rows());

    Eigen::VectorXd grad =
        Eigen::VectorXd::Zero(collisions->vertex_ids().size() * 3);
    for (int ci = 0; ci < collisions->size(); ci++) {
        const auto& cc = (*collisions)[ci];
        const Eigen::VectorXd g = cc.weight
            * cc.gradient(cc.dof(V_view), params, /*adaptive=*/nullptr);
        for (int j = 0; j < cc.num_vertices(); j++) {
            grad.segment<3>(
                3 * collisions->vertex_ids_inverse(cc.vertex_id(j))) +=
                g.segment<3>(3 * j);
        }
    }

    const index_t q_local = collisions->vertex_ids_inverse(vid);
    return grad.segment<3>(3 * q_local);
}

Eigen::Matrix3d ArbitraryPointPotential::hessian(
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::RowVector3d> q) const
{
    const auto collisions = build_collisions_at_point(V, q);
    const VertexMatrixView<3> V_view(V, q);
    const index_t vid = static_cast<index_t>(V.rows());

    const int m = static_cast<int>(collisions->vertex_ids().size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m * 3, m * 3);
    for (int ci = 0; ci < collisions->size(); ci++) {
        const auto& cc = (*collisions)[ci];
        const Eigen::MatrixXd h =
            cc.hessian(cc.dof(V_view), params, /*adaptive=*/nullptr)
            * cc.weight;
        for (int i = 0; i < cc.num_vertices(); i++) {
            for (int j = 0; j < cc.num_vertices(); j++) {
                H.block<3, 3>(
                    3 * collisions->vertex_ids_inverse(cc.vertex_id(i)),
                    3 * collisions->vertex_ids_inverse(cc.vertex_id(j))) +=
                    h.block<3, 3>(3 * i, 3 * j);
            }
        }
    }

    const index_t q_local = collisions->vertex_ids_inverse(vid);
    return H.block<3, 3>(3 * q_local, 3 * q_local);
}

std::tuple<double, Eigen::Vector3d, Eigen::Matrix3d>
ArbitraryPointPotential::evaluate(
    Eigen::ConstRef<Eigen::MatrixXd> V,
    Eigen::ConstRef<Eigen::RowVector3d> q) const
{
    const auto collisions = build_collisions_at_point(V, q);
    const VertexMatrixView<3> V_view(V, q);
    const index_t vid = static_cast<index_t>(V.rows());

    double value = 0.0;
    const int m = static_cast<int>(collisions->vertex_ids().size());
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(m * 3);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m * 3, m * 3);

    for (int ci = 0; ci < collisions->size(); ci++) {
        const auto& cc = (*collisions)[ci];
        const auto dof = cc.dof(V_view);

        value += cc.weight * cc(dof, params, /*adaptive=*/nullptr);

        const Eigen::VectorXd g =
            cc.weight * cc.gradient(dof, params, /*adaptive=*/nullptr);
        const Eigen::MatrixXd h =
            cc.weight * cc.hessian(dof, params, /*adaptive=*/nullptr);

        for (int i = 0; i < cc.num_vertices(); i++) {
            const index_t gi = collisions->vertex_ids_inverse(cc.vertex_id(i));
            grad.segment<3>(3 * gi) += g.segment<3>(3 * i);
            for (int j = 0; j < cc.num_vertices(); j++) {
                const index_t gj =
                    collisions->vertex_ids_inverse(cc.vertex_id(j));
                H.block<3, 3>(3 * gi, 3 * gj) += h.block<3, 3>(3 * i, 3 * j);
            }
        }
    }

    const index_t q_local = collisions->vertex_ids_inverse(vid);
    return { value, grad.segment<3>(3 * q_local),
             H.block<3, 3>(3 * q_local, 3 * q_local) };
}

} // namespace ipc
