#include "high_order_contact_potential.hpp"

#include <ipc/utils/local_to_global.hpp>
#include <ipc/utils/maybe_parallel_for.hpp>

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "ipc/distance/edge_edge.hpp"
#include "ipc/smooth_contact/distance/point_face.hpp"
#include "ipc/smooth_contact/distance/mollifier.hpp"
#include "ipc/high_order_contact/quadrature_potential.hpp"

namespace ipc {

double HighOrderContactPotential::operator()(
    const HighOrderCollisions& collisions,
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> X) const
{
    assert(X.rows() == mesh.num_vertices());

    if (collisions.empty()) {
        return 0;
    }

    tbb::enumerable_thread_specific<double> storage(0);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), collisions.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_potential = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                // Quadrature weight is premultiplied by local potential
                local_potential += (*this)(collisions[i], collisions[i].dof(X));
            }
        });

    if (mesh.dim() == 3) {
        if (!collisions.use_quadrature) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(size_t(0), collisions.triple_collisions.size()),
                [&](const tbb::blocked_range<size_t>& r) {
                    auto& local_potential = storage.local();
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        // Quadrature weight is premultiplied by local potential
                        local_potential += (*this)(*collisions.triple_collisions[i], collisions.triple_collisions[i]->dof(X));
                    }
                });
        }
        else {
            double total = 0;
            for (index_t f = 0; f < mesh.num_faces(); f++) {
                const double area = mesh.face_areas()(f);

                Eigen::MatrixXd V_extended(X.rows() + 1, 3);
                V_extended.topRows(X.rows()) = X;
                V_extended.row(X.rows()) = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                for (index_t le = 0; le < 3; le++) {
                    const index_t edge_id = mesh.faces_to_edges()(f, le);
                    const index_t ea = mesh.edges()(edge_id, 0);
                    const index_t eb = mesh.edges()(edge_id, 1);

                    const std::set<index_t> close_edges = collisions.m_candidates.ee_set(edge_id);

                    double local_potential = 0;
                    for (index_t other_edge_id : close_edges) {
                        const index_t ec = mesh.edges()(other_edge_id, 0);
                        const index_t ed = mesh.edges()(other_edge_id, 1);

                        // Skip adjacent edges
                        if (ea == ec || ea == ed || eb == ec || eb == ed) {
                            continue;
                        }

                        if (auto iter = collisions.edge_edge_collisions.find(std::make_pair(edge_id, other_edge_id));
                            iter != collisions.edge_edge_collisions.end()) {

                            const double dist = sqrt(edge_edge_distance(
                                X.row(ea), X.row(eb),
                                X.row(ec), X.row(ed), EdgeEdgeDistanceType::EA_EB));

                            std::array<HeavisideType, 4> mtypes{{HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT}};
                            double mollifier = Math<double>::cubic_spline(dist / params.dhat) * 1.5;
                            mollifier *= edge_edge_mollifier<double>(
                                    X.row(ea), X.row(eb),
                                    X.row(ec), X.row(ed),
                                    mtypes, dist * dist);

                            local_potential += mollifier * PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(X, iter->second, params);
                        }
                        else {
                            /* P(q) = 0 */
                        }
                    }

                    // face center
                    if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                        local_potential += PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
                            V_extended, iter->second, params);
                    }

                    // vertex ea
                    if (auto iter = collisions.vertex_collisions.find(ea); iter != collisions.vertex_collisions.end()) {
                        local_potential += PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
                            X, iter->second, params);
                    }

                    // vertex eb
                    if (auto iter = collisions.vertex_collisions.find(eb); iter != collisions.vertex_collisions.end()) {
                        local_potential += PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
                            X, iter->second, params);
                    }

                    total += local_potential * area / 9.;
                }
            }

            return total;
        }
    }
    return storage.combine([](double a, double b) { return a + b; });
}

Eigen::VectorXd HighOrderContactPotential::gradient(
    const HighOrderCollisions& collisions,
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> X) const
{
    assert(X.rows() == mesh.num_vertices());

    if (collisions.empty()) {
        return Eigen::VectorXd::Zero(X.size());
    }

    const int dim = X.cols();

    auto storage =
        create_thread_storage<Eigen::VectorXd>(Eigen::VectorXd::Zero(X.size()));
    maybe_parallel_for(
        collisions.size(), [&](int start, int end, int thread_id) {
            auto& global_grad = get_local_thread_storage(storage, thread_id);

            for (size_t i = start; i < end; i++) {
                const HighOrderCollision& collision = collisions[i];

                const Eigen::VectorXd local_grad =
                    this->gradient(collision, collision.dof(X));

                const std::vector<index_t> vids = collision.vertex_ids();

                local_gradient_to_global_gradient(
                    local_grad, vids, dim, global_grad);
            }
        });

    if (mesh.dim() == 3) {
        if (!collisions.use_quadrature) {
            maybe_parallel_for(
                collisions.triple_collisions.size(), [&](int start, int end, int thread_id) {
                    auto& global_grad = get_local_thread_storage(storage, thread_id);

                    for (size_t i = start; i < end; i++) {
                        const TriplePairCollision& collision = *collisions.triple_collisions[i];

                        const Eigen::VectorXd local_grad =
                            this->gradient(collision, collision.dof(X));

                        const std::vector<index_t> vids = collision.vertex_ids();

                        local_gradient_to_global_gradient(
                            local_grad, vids, dim, global_grad);
                    }
                });
        }
        else {
            Eigen::VectorXd grad;
            grad.setZero(X.size());

            using T = ADGrad<12>;

            for (index_t f = 0; f < mesh.num_faces(); f++) {
                const double area = mesh.face_areas()(f);

                Eigen::MatrixXd V_extended(X.rows() + 1, 3);
                V_extended.topRows(X.rows()) = X;
                V_extended.row(X.rows()) = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                for (index_t le = 0; le < 3; le++) {
                    const index_t edge_id = mesh.faces_to_edges()(f, le);
                    const index_t ea = mesh.edges()(edge_id, 0);
                    const index_t eb = mesh.edges()(edge_id, 1);

                    const std::set<index_t> close_edges = collisions.m_candidates.ee_set(edge_id);

                    Eigen::SparseMatrix<double> local_grad(X.size(), 1);
                    for (index_t other_edge_id : close_edges) {
                        const index_t ec = mesh.edges()(other_edge_id, 0);
                        const index_t ed = mesh.edges()(other_edge_id, 1);

                        // Skip adjacent edges
                        if (ea == ec || ea == ed || eb == ec || eb == ed) {
                            continue;
                        }

                        if (auto iter = collisions.edge_edge_collisions.find(std::make_pair(edge_id, other_edge_id));
                            iter != collisions.edge_edge_collisions.end()) {

                            // collisions.edge_edge_collisions only contain EA_EB type collision
                            // other types are ignored because the mollifier makes them vanish

                            Eigen::Vector<double, 12> positions;
                            positions << X.row(ea).transpose(), X.row(eb).transpose(), X.row(ec).transpose(), X.row(ed).transpose();

                            Eigen::Matrix<T, 4, 3> positionsT = slice_positions<T, 4, 3>(positions);

                            const T dist = sqrt(line_line_sqr_distance<T>(
                                positionsT.row(0), positionsT.row(1),
                                positionsT.row(2), positionsT.row(3)));

                            std::array<HeavisideType, 4> mtypes{{HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT}};
                            T mollifier = Math<T>::cubic_spline(dist / params.dhat) * 1.5;
                            mollifier *= edge_edge_mollifier<T>(
                                positionsT.row(0), positionsT.row(1),
                                positionsT.row(2), positionsT.row(3),
                                    mtypes, dist * dist);

                            const double local_potential_1 = PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(X, iter->second, params);
                            const Eigen::SparseMatrix<double> local_grad_1 = PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(X, iter->second, params);
                            
                            local_grad += mollifier.val * local_grad_1;
                            const Vector12d local_grad_2 = local_potential_1 * mollifier.grad;
                            for (int d = 0; d < 3; d++) {
                                local_grad.coeffRef(ea * 3 + d, 0) += local_grad_2(d + 0);
                                local_grad.coeffRef(eb * 3 + d, 0) += local_grad_2(d + 3);

                                local_grad.coeffRef(ec * 3 + d, 0) += local_grad_2(d + 6);
                                local_grad.coeffRef(ed * 3 + d, 0) += local_grad_2(d + 9);
                            }
                        }
                        else {
                            /* P(q) = 0 */
                        }
                    }

                    // face center
                    if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                        local_grad += PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
                            V_extended, mesh.faces().row(f), iter->second, params);
                    }

                    // vertex ea
                    if (auto iter = collisions.vertex_collisions.find(ea); iter != collisions.vertex_collisions.end()) {
                        local_grad += PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                            X, iter->second, params);
                    }

                    // vertex eb
                    if (auto iter = collisions.vertex_collisions.find(eb); iter != collisions.vertex_collisions.end()) {
                        local_grad += PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                            X, iter->second, params);
                    }

                    grad += local_grad * (area / 9.);
                }
            }

            return grad;
        }
    }

    Eigen::VectorXd grad;
    grad.setZero(X.size());
    for (const auto& local_storage : storage) {
        grad += local_storage;
    }
    return grad;
}

Eigen::SparseMatrix<double> HighOrderContactPotential::hessian(
    const HighOrderCollisions& collisions,
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> X,
    const PSDProjectionMethod project_hessian_to_psd) const
{
    assert(X.rows() == mesh.num_vertices());

    if (collisions.empty()) {
        return Eigen::SparseMatrix<double>(X.size(), X.size());
    }

    const int dim = X.cols();
    const int ndof = X.size();

    const int max_triplets_size = int(1e7);
    const int buffer_size = std::min(max_triplets_size, ndof);
    auto storage =
        create_thread_storage(LocalThreadMatStorage(buffer_size, ndof, ndof));
    maybe_parallel_for(
        collisions.size(), [&](int start, int end, int thread_id) {
            auto& hess_triplets = get_local_thread_storage(storage, thread_id);

            for (size_t i = start; i < end; i++) {
                const HighOrderCollision& collision = collisions[i];

                const Eigen::MatrixXd local_hess = this->hessian(
                    collisions[i], collisions[i].dof(X),
                    project_hessian_to_psd);

                local_hessian_to_global_triplets(
                    local_hess, collision.vertex_ids(), dim,
                    *(hess_triplets.cache));
            }
        });

    if (mesh.dim() == 3) {
        if (!collisions.use_quadrature) {
            maybe_parallel_for(
                collisions.triple_collisions.size(), [&](int start, int end, int thread_id) {
                    auto& hess_triplets = get_local_thread_storage(storage, thread_id);

                    for (size_t i = start; i < end; i++) {
                        const TriplePairCollision& collision = *collisions.triple_collisions[i];

                        const Eigen::MatrixXd local_hess = this->hessian(
                            collisions[i], collisions[i].dof(X),
                            project_hessian_to_psd);

                        local_hessian_to_global_triplets(
                            local_hess, collision.vertex_ids(), dim,
                            *(hess_triplets.cache));
                    }
                });
        }
        else {
            using T = ADHessian<12>;

            // TODO: Implement project PSD

            Eigen::SparseMatrix<double> hess(ndof, ndof);

            std::vector<Eigen::Triplet<double>> triplets;

            for (index_t f = 0; f < mesh.num_faces(); f++) {
                const double area = mesh.face_areas()(f);

                Eigen::MatrixXd V_extended(X.rows() + 1, 3);
                V_extended.topRows(X.rows()) = X;
                V_extended.row(X.rows()) = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                for (index_t le = 0; le < 3; le++) {
                    const index_t edge_id = mesh.faces_to_edges()(f, le);
                    const index_t ea = mesh.edges()(edge_id, 0);
                    const index_t eb = mesh.edges()(edge_id, 1);

                    const std::set<index_t> close_edges = collisions.m_candidates.ee_set(edge_id);

                    Eigen::SparseMatrix<double> local_hess(X.size(), X.size());
                    for (index_t other_edge_id : close_edges) {
                        const index_t ec = mesh.edges()(other_edge_id, 0);
                        const index_t ed = mesh.edges()(other_edge_id, 1);

                        // Skip adjacent edges
                        if (ea == ec || ea == ed || eb == ec || eb == ed) {
                            continue;
                        }

                        if (auto iter = collisions.edge_edge_collisions.find(std::make_pair(edge_id, other_edge_id));
                            iter != collisions.edge_edge_collisions.end()) {

                            // collisions.edge_edge_collisions only contain EA_EB type collision
                            // other types are ignored because the mollifier makes them vanish

                            Eigen::Vector<double, 12> positions;
                            positions << X.row(ea).transpose(), X.row(eb).transpose(), X.row(ec).transpose(), X.row(ed).transpose();

                            Eigen::Matrix<T, 4, 3> positionsT = slice_positions<T, 4, 3>(positions);

                            const T dist = sqrt(line_line_sqr_distance<T>(
                                positionsT.row(0), positionsT.row(1),
                                positionsT.row(2), positionsT.row(3)));

                            std::array<HeavisideType, 4> mtypes{{HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT, HeavisideType::VARIANT}};
                            T mollifier = Math<T>::cubic_spline(dist / params.dhat) * 1.5;
                            mollifier *= edge_edge_mollifier<T>(
                                positionsT.row(0), positionsT.row(1),
                                positionsT.row(2), positionsT.row(3),
                                    mtypes, dist * dist);

                            const double local_potential_1 = PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(X, iter->second, params);
                            const Eigen::SparseMatrix<double> local_grad_1 = PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions(X, iter->second, params);
                            const Eigen::SparseMatrix<double> local_hess_1 = PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(X, iter->second, params);

                            local_hess += local_hess_1 * mollifier.val;

                            std::array<index_t, 4> ee_indices = {{ea, eb, ec, ed}};
                            const Matrix12d local_hess_2 = local_potential_1 * mollifier.Hess;
                            for (index_t i = 0; i < 4; i++) {
                                for (index_t j = 0; j < 4; j++) {
                                    for (index_t di = 0; di < 3; di++) {
                                        for (index_t dj = 0; dj < 3; dj++) {
                                            triplets.emplace_back(ee_indices[i] * 3 + di, ee_indices[j] * 3 + dj, local_hess_2(i * 3 + di, j * 3 + dj) * (area / 9.));
                                        }
                                    }
                                }
                            }

                            for (index_t k = 0; k < local_grad_1.outerSize(); ++k) {
                                for (Eigen::SparseMatrix<double>::InnerIterator it(local_grad_1, k); it; ++it) {
                                    for (index_t i = 0; i < 12; i++) {
                                        assert(it.col() == 0);
                                        index_t id = ee_indices[i / 3] * 3 + i % 3;
                                        triplets.emplace_back(id, it.row(), mollifier.grad(i) * it.value() * (area / 9.));
                                        triplets.emplace_back(it.row(), id, mollifier.grad(i) * it.value() * (area / 9.));
                                    }
                                }
                            }
                        }
                        else {
                            /* P(q) = 0 */
                        }
                    }

                    // face center
                    if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                        local_hess += PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(
                            V_extended, mesh.faces().row(f), iter->second, params);
                    }

                    // vertex ea
                    if (auto iter = collisions.vertex_collisions.find(ea); iter != collisions.vertex_collisions.end()) {
                        local_hess += PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
                            X, iter->second, params);
                    }

                    // vertex eb
                    if (auto iter = collisions.vertex_collisions.find(eb); iter != collisions.vertex_collisions.end()) {
                        local_hess += PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
                            X, iter->second, params);
                    }

                    hess += local_hess * (area / 9.);
                }
            }

            Eigen::SparseMatrix<double> hess2(ndof, ndof);
            hess2.setFromTriplets(triplets.begin(), triplets.end());

            return hess + hess2;
        }
    }

    Eigen::SparseMatrix<double> hess(ndof, ndof);

    // Assemble the stiffness matrix by concatenating the tuples in each local
    // storage

    // Collect thread storages
    std::vector<LocalThreadMatStorage*> storages(storage.size());
    int index = 0;
    for (auto& local_storage : storage) {
        storages[index++] = &local_storage;
    }

    maybe_parallel_for(
        storages.size(), [&](int i) { storages[i]->cache->prune(); });

    if (storage.empty()) {
        return Eigen::SparseMatrix<double>();
    }

    // Prepares for parallel concatenation
    std::vector<int> offsets(storage.size());

    index = 0;
    int triplet_count = 0;
    for (auto& local_storage : storage) {
        offsets[index++] = triplet_count;
        triplet_count += local_storage.cache->triplet_count();
    }

    std::vector<Eigen::Triplet<double>> triplets;

    assert(!storages.empty());
    if (triplet_count >= triplets.max_size()) {
        // Serial fallback version in case the vector of triplets cannot be
        // allocated

        logger().warn(
            "Cannot allocate space for triplets, switching to serial assembly.");

        // Serially merge local storages
        for (LocalThreadMatStorage& local_storage : storage) {
            hess += local_storage.cache->get_matrix(false); // will also prune
        }
        hess.makeCompressed();
    } else {
        triplets.resize(triplet_count);

        // Parallel copy into triplets
        maybe_parallel_for(storages.size(), [&](int i) {
            const SparseMatrixCache& cache =
                dynamic_cast<const SparseMatrixCache&>(*storages[i]->cache);
            int offset = offsets[i];

            std::copy(
                cache.entries().begin(), cache.entries().end(),
                triplets.begin() + offset);
            offset += cache.entries().size();

            if (cache.mat().nonZeros() > 0) {
                int count = 0;
                for (int k = 0; k < cache.mat().outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(
                             cache.mat(), k);
                         it; ++it) {
                        assert(count < cache.mat().nonZeros());
                        triplets[offset + count++] = Eigen::Triplet<double>(
                            it.row(), it.col(), it.value());
                    }
                }
            }
        });

        // Sort and assemble
        hess.setFromTriplets(triplets.begin(), triplets.end());
    }

    return hess;
}

double HighOrderContactPotential::operator()(
    const HighOrderCollision& collision,
    Eigen::ConstRef<Eigen::VectorXd> positions) const
{
    return collision.weight * collision(positions, params);
}

Eigen::VectorXd HighOrderContactPotential::gradient(
    const HighOrderCollision& collision,
    Eigen::ConstRef<Eigen::VectorXd> positions) const
{
    return collision.weight * collision.gradient(positions, params);
}

Eigen::MatrixXd HighOrderContactPotential::hessian(
    const HighOrderCollision& collision,
    Eigen::ConstRef<Eigen::VectorXd> positions,
    const PSDProjectionMethod project_hessian_to_psd) const
{
    Eigen::MatrixXd hess = collision.weight * collision.hessian(positions, params);
    return project_to_psd(hess, project_hessian_to_psd);
}

double HighOrderContactPotential::operator()(
        const TriplePairCollision& collision,
        Eigen::ConstRef<Eigen::VectorXd> positions) const
{
    return collision.weight * collision(positions, params);
}

    Eigen::VectorXd HighOrderContactPotential::gradient(
        const TriplePairCollision& collision,
        Eigen::ConstRef<Eigen::VectorXd> positions) const
{
    return collision.weight * collision.gradient(positions, params);
}

Eigen::MatrixXd HighOrderContactPotential::hessian(
        const TriplePairCollision& collision,
        Eigen::ConstRef<Eigen::VectorXd> positions,
        const PSDProjectionMethod project_hessian_to_psd) const
{
    Eigen::MatrixXd hess = collision.weight * collision.hessian(positions, params);
    return project_to_psd(hess, project_hessian_to_psd);
}
} // namespace ipc