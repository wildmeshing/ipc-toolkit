#include "high_order_contact_potential.hpp"

#include "igl/write_triangle_mesh.h"

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
            throw std::runtime_error("Not implemented!");
        }
        else {
            auto potential_storage = create_thread_storage(0.0);

            auto loop_body = [&](int start, int end, int thread_id) {
                double& total = get_local_thread_storage(potential_storage, thread_id);
                for (index_t f = start; f < end; f++) {
                    const double area = mesh.face_areas()(f);

                    const Eigen::RowVector3d face_center = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

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

                            if (auto iter = collisions.edge_edge_collisions_advanced.find(std::make_pair(edge_id, other_edge_id));
                                iter != collisions.edge_edge_collisions_advanced.end()) {

                                auto dtype = edge_edge_distance_type(
                                    X.row(ea), X.row(eb),
                                    X.row(ec), X.row(ed));

                                const double dist = sqrt(edge_edge_distance(
                                    X.row(ea), X.row(eb),
                                    X.row(ec), X.row(ed), dtype));

                                const double uv = closest_point_uv<double>(
                                    X.row(ea), X.row(eb),
                                    X.row(ec), X.row(ed), dtype);

                                const Eigen::RowVector3d ee_closest_point = uv * (X.row(eb) - X.row(ea)) + X.row(ea);

                                double mollifier = Math<double>::cubic_spline(dist / params.dhat) * 1.5;
                                mollifier *= half_edge_edge_mollifier<double>(
                                        X.row(ea), X.row(eb),
                                        X.row(ec), X.row(ed),
                                        dtype);

                                local_potential += mollifier * PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                                    ConcatMatrixView<3>(X, ee_closest_point), iter->second, params, dtype);
                            }
                            else {
                                /* P(q) = 0 */
                            }
                        }

                        // face center
                        if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                            local_potential += PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
                                ConcatMatrixView<3>(X, face_center), iter->second, params);
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
            };

            if constexpr (use_parallel_eval) {
                maybe_parallel_for(mesh.num_faces(), loop_body);
            } else {
                loop_body(0, mesh.num_faces(), 0);
            }

            double total_potential = 0;
            for (const auto& local_potential : potential_storage) {
                total_potential += local_potential;
            }
            return total_potential;
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
            throw std::runtime_error("Not implemented!");
        }
        else {
            auto grad_storage = create_thread_storage<Eigen::VectorXd>(Eigen::VectorXd::Zero(X.size()));

            using T = ADGrad<12>;

            auto loop_body = [&](int start, int end, int thread_id) {
                Eigen::VectorXd& grad = get_local_thread_storage(grad_storage, thread_id);
                for (index_t f = start; f < end; f++) {
                    const double area = mesh.face_areas()(f);

                    const Eigen::RowVector3d face_center = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                    for (index_t le = 0; le < 3; le++) {
                        const index_t edge_id = mesh.faces_to_edges()(f, le);
                        const index_t ea = mesh.edges()(edge_id, 0);
                        const index_t eb = mesh.edges()(edge_id, 1);

                        const std::set<index_t> close_edges = collisions.m_candidates.ee_set(edge_id);

                        // TODO: Collect local DoFs instead of directly using SparseMatrix
                        Eigen::SparseMatrix<double> local_grad(X.size(), 1);
                        for (index_t other_edge_id : close_edges) {
                            const index_t ec = mesh.edges()(other_edge_id, 0);
                            const index_t ed = mesh.edges()(other_edge_id, 1);

                            // Skip adjacent edges
                            if (ea == ec || ea == ed || eb == ec || eb == ed) {
                                continue;
                            }

                            if (auto iter = collisions.edge_edge_collisions_advanced.find(std::make_pair(edge_id, other_edge_id));
                                iter != collisions.edge_edge_collisions_advanced.end()) {

                                // collisions.edge_edge_collisions_advanced only contain EA_EB* type collision
                                // other types are ignored because the mollifier makes them vanish

                                Eigen::Vector<double, 12> positions;
                                positions << X.row(ea).transpose(), X.row(eb).transpose(), X.row(ec).transpose(), X.row(ed).transpose();

                                const auto dtype = edge_edge_distance_type(
                                    X.row(ea), X.row(eb),
                                    X.row(ec), X.row(ed));

                                Eigen::Matrix<T, 4, 3> positionsT = slice_positions<T, 4, 3>(positions);

                                const T dist = sqrt(edge_edge_sqr_distance<T>(
                                    positionsT.row(0), positionsT.row(1),
                                    positionsT.row(2), positionsT.row(3), dtype));

                                const T uv = closest_point_uv<T>(
                                    positionsT.row(0), positionsT.row(1),
                                    positionsT.row(2), positionsT.row(3), dtype);

                                const Eigen::RowVector3<T> ee_closest_point_T = uv * (positionsT.row(1) - positionsT.row(0)) + positionsT.row(0);
                                const Eigen::RowVector3<double> ee_closest_point(ee_closest_point_T(0).val, ee_closest_point_T(1).val, ee_closest_point_T(2).val);

                                T mollifier = Math<T>::cubic_spline(dist / params.dhat) * 1.5;
                                mollifier *= half_edge_edge_mollifier<T>(
                                    positionsT.row(0), positionsT.row(1),
                                    positionsT.row(2), positionsT.row(3),
                                        dtype);

                                ConcatMatrixView<3> X_extended(X, ee_closest_point);
                                assert(X_extended.rows() == X.rows() + 1);
                                assert(X_extended.m_A == X.data() && "ConcatMatrixView has made a deepcopy!");
                                const double local_potential_1 = PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                                    X_extended, iter->second, params, dtype);
                                Eigen::SparseMatrix<double> local_grad_1;
                                {
                                    Eigen::VectorXd tmp = Eigen::VectorXd::Zero(X.size());
                                    tmp(iter->second.dofs()) = PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<T>(
                                    X_extended, iter->second, params, ee_closest_point_T);
                                    local_grad_1 = tmp.sparseView();
                                }

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
                            Eigen::VectorXd tmp = Eigen::VectorXd::Zero(local_grad.size());
                            tmp(iter->second.dofs()) = PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
                                ConcatMatrixView<3>(X, face_center), iter->second, params);
                            local_grad += tmp.sparseView();
                        }

                        // vertex ea
                        if (auto iter = collisions.vertex_collisions.find(ea); iter != collisions.vertex_collisions.end()) {
                            Eigen::VectorXd tmp = Eigen::VectorXd::Zero(local_grad.size());
                            tmp(iter->second.dofs()) = PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                                X, iter->second, params);
                            local_grad += tmp.sparseView();
                        }

                        // vertex eb
                        if (auto iter = collisions.vertex_collisions.find(eb); iter != collisions.vertex_collisions.end()) {
                            Eigen::VectorXd tmp = Eigen::VectorXd::Zero(local_grad.size());
                            tmp(iter->second.dofs()) = PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                                X, iter->second, params);
                            local_grad += tmp.sparseView();
                        }

                        grad += local_grad * (area / 9.);
                    }
                }
            };

            if constexpr (use_parallel_eval) {
                maybe_parallel_for(mesh.num_faces(), loop_body);
            } else {
                loop_body(0, mesh.num_faces(), 0);
            }

            Eigen::VectorXd total_grad = Eigen::VectorXd::Zero(X.size());
            for (const auto& local_grad : grad_storage) {
                total_grad += local_grad;
            }
            return total_grad;
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
            throw std::runtime_error("Not implemented!");
        }
        else {
            // TODO: Implement project PSD

            using T = ADHessian<12>;

            Eigen::SparseMatrix<double> hess(ndof, ndof);

            auto triplets_storage = create_thread_storage<std::vector<Eigen::Triplet<double>>>(
                std::vector<Eigen::Triplet<double>>());

            auto loop_body = [&](int start, int end, int thread_id) {
                auto& triplets = get_local_thread_storage(triplets_storage, thread_id);
                for (index_t f = start; f < end; f++) {
                    const double area = mesh.face_areas()(f);

                    const Eigen::RowVector3d face_center = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                    for (index_t le = 0; le < 3; le++) {
                        const index_t edge_id = mesh.faces_to_edges()(f, le);
                        const index_t ea = mesh.edges()(edge_id, 0);
                        const index_t eb = mesh.edges()(edge_id, 1);

                        const std::set<index_t> close_edges = collisions.m_candidates.ee_set(edge_id);

                        for (index_t other_edge_id : close_edges) {
                            const index_t ec = mesh.edges()(other_edge_id, 0);
                            const index_t ed = mesh.edges()(other_edge_id, 1);

                            // Skip adjacent edges
                            if (ea == ec || ea == ed || eb == ec || eb == ed) {
                                continue;
                            }

                            if (auto iter = collisions.edge_edge_collisions_advanced.find(std::make_pair(edge_id, other_edge_id));
                                iter != collisions.edge_edge_collisions_advanced.end()) {

                                // collisions.edge_edge_collisions_advanced only contain EA_EB* type collision
                                // other types are ignored because the mollifier makes them vanish

                                Eigen::Vector<double, 12> positions;
                                positions << X.row(ea).transpose(), X.row(eb).transpose(), X.row(ec).transpose(), X.row(ed).transpose();

                                const auto dtype = edge_edge_distance_type(
                                    X.row(ea), X.row(eb),
                                    X.row(ec), X.row(ed));

                                Eigen::Matrix<T, 4, 3> positionsT = slice_positions<T, 4, 3>(positions);

                                const T dist = sqrt(edge_edge_sqr_distance<T>(
                                    positionsT.row(0), positionsT.row(1),
                                    positionsT.row(2), positionsT.row(3), dtype));

                                const T uv = closest_point_uv<T>(
                                    positionsT.row(0), positionsT.row(1),
                                    positionsT.row(2), positionsT.row(3), dtype);

                                const Eigen::RowVector3<T> ee_closest_point_T = uv * (positionsT.row(1) - positionsT.row(0)) + positionsT.row(0);
                                const Eigen::RowVector3<double> ee_closest_point(ee_closest_point_T(0).val, ee_closest_point_T(1).val, ee_closest_point_T(2).val);

                                T mollifier = Math<T>::cubic_spline(dist / params.dhat) * 1.5;
                                mollifier *= half_edge_edge_mollifier<T>(
                                    positionsT.row(0), positionsT.row(1),
                                    positionsT.row(2), positionsT.row(3),
                                        dtype);

                                const HighOrderCollisionDict<PointType::EDGE>& dict = iter->second;

                                ConcatMatrixView<3> X_extended(X, ee_closest_point);
                                assert(X_extended.m_A == X.data() && "ConcatMatrixView has made a deepcopy!");
                                const double local_potential_1 = PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                                    X_extended, dict, params, dtype);
                                const Eigen::VectorXd local_grad = PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<T>(
                                    X_extended, dict, params, ee_closest_point_T);
                                Eigen::MatrixXd local_hess = PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
                                    X_extended, dict, params, ee_closest_point_T) * mollifier.val;

                                for (index_t i = 0; i < 4; i++) {
                                    for (index_t j = 0; j < 4; j++) {
                                        local_hess.block<3, 3>(dict.vertex_ids_inverse(dict.primary_vertex_ids()[i]) * 3, dict.vertex_ids_inverse(dict.primary_vertex_ids()[j]) * 3) += local_potential_1 * mollifier.Hess.block<3, 3>(i * 3, j * 3);
                                    }
                                }

                                Eigen::MatrixXd tmp;
                                for (index_t i = 0; i < 4; i++) {
                                    tmp = mollifier.grad.segment<3>(i * 3) * local_grad.transpose();
                                    local_hess.middleRows(dict.vertex_ids_inverse(dict.primary_vertex_ids()[i]) * 3, 3) += tmp;
                                    local_hess.middleCols(dict.vertex_ids_inverse(dict.primary_vertex_ids()[i]) * 3, 3) += tmp.transpose();
                                }

                                if (project_hessian_to_psd != PSDProjectionMethod::NONE) {
                                    local_hess = project_to_psd(local_hess, project_hessian_to_psd);
                                }

                                for (int i = 0; i < local_hess.rows(); i++) {
                                    for (int j = 0; j < local_hess.cols(); j++) {
                                        if (local_hess(i, j) != 0) {
                                            triplets.emplace_back(dict.dofs()[i], dict.dofs()[j], local_hess(i, j) * area / 9.);
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
                            const Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(
                                ConcatMatrixView<3>(X, face_center), iter->second, params, project_hessian_to_psd);
                            for (int i = 0; i < h.rows(); i++) {
                                index_t row = iter->second.dofs()[i];
                                for (int j = 0; j < h.cols(); j++) {
                                    index_t col = iter->second.dofs()[j];
                                    triplets.emplace_back(row, col, h(i, j) * (area / 9.));
                                }
                            }
                        }

                        // vertex ea
                        if (auto iter = collisions.vertex_collisions.find(ea); iter != collisions.vertex_collisions.end()) {
                            const Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
                                X, iter->second, params, project_hessian_to_psd);
                            for (int i = 0; i < h.rows(); i++) {
                                index_t row = iter->second.dofs()[i];
                                for (int j = 0; j < h.cols(); j++) {
                                    index_t col = iter->second.dofs()[j];
                                    triplets.emplace_back(row, col, h(i, j) * (area / 9.));
                                }
                            }
                        }

                        // vertex eb
                        if (auto iter = collisions.vertex_collisions.find(eb); iter != collisions.vertex_collisions.end()) {
                            const Eigen::MatrixXd h = PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
                                X, iter->second, params, project_hessian_to_psd);
                            for (int i = 0; i < h.rows(); i++) {
                                index_t row = iter->second.dofs()[i];
                                for (int j = 0; j < h.cols(); j++) {
                                    index_t col = iter->second.dofs()[j];
                                    triplets.emplace_back(row, col, h(i, j) * (area / 9.));
                                }
                            }
                        }
                    }
                }
            };

            if constexpr (use_parallel_eval) {
                maybe_parallel_for(mesh.num_faces(), loop_body);
            } else {
                loop_body(0, mesh.num_faces(), 0);
            }

            std::vector<Eigen::Triplet<double>> all_triplets;
            size_t num_triplets = 0;
            for (const auto& local_triplets : triplets_storage) {
                num_triplets += local_triplets.size();
            }
            all_triplets.reserve(num_triplets);
            for (const auto& local_triplets : triplets_storage) {
                all_triplets.insert(all_triplets.end(), local_triplets.begin(), local_triplets.end());
            }

            Eigen::SparseMatrix<double> hess2(ndof, ndof);
            hess2.setFromTriplets(all_triplets.begin(), all_triplets.end());

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