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

    double result = 0;

    if (mesh.dim() == 2) {
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
        result = storage.combine([](double a, double b) { return a + b; });
    }
    else if (mesh.dim() == 3) {
        {
            auto potential_storage = create_thread_storage(0.0);

            auto loop_body = [&](int start, int end, int thread_id) {
                double& total = get_local_thread_storage(potential_storage, thread_id);
                for (index_t f = start; f < end; f++) {
                    const double area = mesh.face_areas()(f);
                    const double w = area / 9.;

                    const Eigen::RowVector3d face_center = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                    double total_w = 0;
                    double total_p = 0;

                    for (index_t le = 0; le < 3; le++) {
                        const index_t edge_id = mesh.faces_to_edges()(f, le);
                        const index_t ea = mesh.edges()(edge_id, 0);
                        const index_t eb = mesh.edges()(edge_id, 1);

                        for (index_t other_edge_id : collisions.m_candidates.ee_set(edge_id)) {
                            const index_t ec = mesh.edges()(other_edge_id, 0);
                            const index_t ed = mesh.edges()(other_edge_id, 1);

                            // Skip adjacent edges
                            if (ea == ec || ea == ed || eb == ec || eb == ed) {
                                continue;
                            }

                            if (auto iter = collisions.edge_edge_collisions.find(std::make_pair(edge_id, other_edge_id));
                                iter != collisions.edge_edge_collisions.end()) {

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
                                    X.row(ec), X.row(ed), dtype);

                                total_w += mollifier;
                                total_p += mollifier * PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                                    ConcatMatrixView<3>(X, ee_closest_point), *(iter->second), params, dtype);
                            }
                        }
                    }

                    total_w += 1.;
                    if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                        total_p += PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
                            ConcatMatrixView<3>(X, face_center), *(iter->second), params);
                    }

                    for (index_t lv = 0; lv < 3; lv++) {
                        const index_t v = mesh.faces()(f, lv);
                        total_w += 1.;
                        if (auto iter = collisions.vertex_collisions.find(v); iter != collisions.vertex_collisions.end()) {
                            total_p += PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
                                X, *(iter->second), params);
                        }
                    }

                    assert(total_w > 0);
                    total += normalize_weights ? w * (total_p / total_w) : w * total_p;
                }
            };

            maybe_parallel_for(mesh.num_faces(), loop_body);

            for (const auto& local_potential : potential_storage) {
                result += local_potential;
            }
        }
    }
    return result;
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

    if (mesh.dim() == 2) {
        maybe_parallel_for(
            collisions.size(), [&](int start, int end, int thread_id) {
                auto& global_grad = get_local_thread_storage(storage, thread_id);

                for (size_t i = start; i < end; i++) {
                    const HighOrderCollision& collision = collisions[i];

                    const Eigen::VectorXd local_grad =
                        this->gradient(collision, collision.dof(X));

                    local_gradient_to_global_gradient(
                        local_grad, collision.vertex_ids(), dim, global_grad);
                }
            });
    }
    else if (mesh.dim() == 3) {
        {
            using T = ADGrad<12>;

            auto loop_body = [&](int start, int end, int thread_id) {
                Eigen::VectorXd& grad = get_local_thread_storage(storage, thread_id);
                for (index_t f = start; f < end; f++) {
                    const double area = mesh.face_areas()(f);
                    const double w = area / 9.;
                    const Eigen::RowVector3d face_center = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                    // Pass 1: collect all quadrature contributions for this face
                    struct EEGradEntry {
                        const HighOrderCollisionDict<PointType::EDGE>* dict;
                        double mol_val;
                        Eigen::Vector<double, 12> mol_grad;
                        double P;
                        Eigen::VectorXd grad_P;
                    };
                    struct ConstGradEntry {  // face center and vertex: constant weight, no mol correction
                        std::vector<int> dofs;
                        Eigen::VectorXd grad_P;
                    };
                    std::vector<EEGradEntry> ee_cache;
                    std::vector<ConstGradEntry> const_cache;
                    double total_w = 0;
                    double total_p = 0;

                    for (index_t le = 0; le < 3; le++) {
                        const index_t edge_id = mesh.faces_to_edges()(f, le);
                        const index_t ea = mesh.edges()(edge_id, 0);
                        const index_t eb = mesh.edges()(edge_id, 1);

                        for (index_t other_edge_id : collisions.m_candidates.ee_set(edge_id)) {
                            const index_t ec = mesh.edges()(other_edge_id, 0);
                            const index_t ed = mesh.edges()(other_edge_id, 1);

                            // Skip adjacent edges
                            if (ea == ec || ea == ed || eb == ec || eb == ed) {
                                continue;
                            }

                            if (auto iter = collisions.edge_edge_collisions.find(std::make_pair(edge_id, other_edge_id));
                                iter != collisions.edge_edge_collisions.end()) {

                                // collisions.edge_edge_collisions only contain EA_EB* type collision
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
                                    positionsT.row(2), positionsT.row(3), dtype);

                                const HighOrderCollisionDict<PointType::EDGE>& dict = *(iter->second);

                                ConcatMatrixView<3> X_extended(X, ee_closest_point);
                                assert(X_extended.rows() == X.rows() + 1);
                                assert(X_extended.m_A == X.data() && "ConcatMatrixView has made a deepcopy!");
                                const double P = PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                                    X_extended, dict, params, dtype);
                                const Eigen::VectorXd grad_P = PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<T>(
                                    X_extended, dict, params, ee_closest_point_T);

                                ee_cache.push_back({&dict, mollifier.val, mollifier.grad, P, grad_P});
                                total_w += mollifier.val;
                                total_p += mollifier.val * P;
                            }
                        }
                    }

                    total_w += 1.;
                    if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                        const double P = PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
                            ConcatMatrixView<3>(X, face_center), (*iter->second), params);
                        const Eigen::VectorXd grad_P = PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
                            ConcatMatrixView<3>(X, face_center), (*iter->second), params);
                        const_cache.push_back(ConstGradEntry{(*iter->second).dofs(), grad_P});
                        total_p += P;
                    }

                    for (index_t lv = 0; lv < 3; lv++) {
                        const index_t v = mesh.faces()(f, lv);
                        total_w += 1.;
                        if (auto iter = collisions.vertex_collisions.find(v); iter != collisions.vertex_collisions.end()) {
                            const double P = PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
                                X, (*iter->second), params);
                            const Eigen::VectorXd grad_P = PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                                X, (*iter->second), params);
                            const_cache.push_back(ConstGradEntry{(*iter->second).dofs(), grad_P});
                            total_p += P;
                        }
                    }

                    // Pass 2: apply gradient
                    assert(total_w > 0);
                    if (normalize_weights) {
                        const double avg_P = total_p / total_w;
                        for (const auto& e : ee_cache) {
                            grad(e.dict->dofs()) += (w / total_w * e.mol_val) * e.grad_P;
                            grad(e.dict->primary_dofs()) += (w / total_w * (e.P - avg_P)) * e.mol_grad;
                        }
                        for (const auto& e : const_cache) {
                            grad(e.dofs) += (w / total_w) * e.grad_P;
                        }
                    } else {
                        for (const auto& e : ee_cache) {
                            grad(e.dict->dofs()) += w * e.mol_val * e.grad_P;
                            grad(e.dict->primary_dofs()) += w * e.P * e.mol_grad;
                        }
                        for (const auto& e : const_cache) {
                            grad(e.dofs) += w * e.grad_P;
                        }
                    }
                }
            };

            maybe_parallel_for(mesh.num_faces(), loop_body);
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

    if (mesh.dim() == 2) {
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
    }
    else if (mesh.dim() == 3) {
        {
            using T = ADHessian<12>;

            auto loop_body = [&](int start, int end, int thread_id) {
                auto& hess_triplets = get_local_thread_storage(storage, thread_id);
                for (index_t f = start; f < end; f++) {
                    const double area = mesh.face_areas()(f);
                    const double w = area / 9.;

                    const Eigen::RowVector3d face_center = (X.row(mesh.faces()(f, 0)) + X.row(mesh.faces()(f, 1)) + X.row(mesh.faces()(f, 2))) / 3.;

                    // Pass 1: collect all quadrature contributions for this face
                    struct EEHessEntry {
                        const HighOrderCollisionDict<PointType::EDGE>* dict;
                        double mol_val;
                        Eigen::Vector<double, 12> mol_grad;        // on primary_dofs
                        Eigen::Matrix<double, 12, 12> mol_hess;    // H(mol) on primary_dofs
                        double P;
                        Eigen::VectorXd grad_P;                    // indexed by dict->dofs()
                        Eigen::MatrixXd local_hess;                // H(mol*P), PSD-projected
                    };
                    struct ConstHessEntry {
                        std::vector<index_t> vertex_ids;
                        std::vector<index_t> dofs;
                        double P;
                        Eigen::VectorXd grad_P;                    // indexed by dofs
                        Eigen::MatrixXd local_hess;                // H(P)
                    };
                    std::vector<EEHessEntry> ee_cache;
                    std::vector<ConstHessEntry> const_cache;
                    double total_w = 0;
                    double total_p = 0;

                    for (index_t le = 0; le < 3; le++) {
                        const index_t edge_id = mesh.faces_to_edges()(f, le);
                        const index_t ea = mesh.edges()(edge_id, 0);
                        const index_t eb = mesh.edges()(edge_id, 1);

                        for (index_t other_edge_id : collisions.m_candidates.ee_set(edge_id)) {
                            const index_t ec = mesh.edges()(other_edge_id, 0);
                            const index_t ed = mesh.edges()(other_edge_id, 1);

                            // Skip adjacent edges
                            if (ea == ec || ea == ed || eb == ec || eb == ed) {
                                continue;
                            }

                            if (auto iter = collisions.edge_edge_collisions.find(std::make_pair(edge_id, other_edge_id));
                                iter != collisions.edge_edge_collisions.end()) {

                                // collisions.edge_edge_collisions only contain EA_EB* type collision
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
                                    positionsT.row(2), positionsT.row(3), dtype);

                                const HighOrderCollisionDict<PointType::EDGE>& dict = *(iter->second);

                                ConcatMatrixView<3> X_extended(X, ee_closest_point);
                                assert(X_extended.m_A == X.data() && "ConcatMatrixView has made a deepcopy!");
                                const double P = PointPotentialHelper::evaluate_potential_at_edge_edge_closest_point_with_cached_collisions(
                                    X_extended, dict, params, dtype);
                                const Eigen::VectorXd grad_P = PointPotentialHelper::evaluate_potential_gradient_at_edge_edge_closest_point_with_cached_collisions<T>(
                                    X_extended, dict, params, ee_closest_point_T);
                                Eigen::MatrixXd local_hess = PointPotentialHelper::evaluate_potential_hessian_at_edge_edge_closest_point_with_cached_collisions(
                                    X_extended, dict, params, ee_closest_point_T) * mollifier.val;

                                for (index_t i = 0; i < 4; i++) {
                                    for (index_t j = 0; j < 4; j++) {
                                        local_hess.block<3, 3>(dict.vertex_ids_inverse(dict.primary_vertex_ids()[i]) * 3, dict.vertex_ids_inverse(dict.primary_vertex_ids()[j]) * 3) += P * mollifier.Hess.block<3, 3>(i * 3, j * 3);
                                    }
                                }

                                for (index_t i = 0; i < 4; i++) {
                                    const Eigen::MatrixXd tmp = mollifier.grad.segment<3>(i * 3) * grad_P.transpose();
                                    local_hess.middleRows(dict.vertex_ids_inverse(dict.primary_vertex_ids()[i]) * 3, 3) += tmp;
                                    local_hess.middleCols(dict.vertex_ids_inverse(dict.primary_vertex_ids()[i]) * 3, 3) += tmp.transpose();
                                }

                                if (project_hessian_to_psd != PSDProjectionMethod::NONE) {
                                    local_hess = project_to_psd(local_hess, project_hessian_to_psd);
                                }

                                ee_cache.push_back({
                                    &dict,
                                    mollifier.val,
                                    mollifier.grad,
                                    mollifier.Hess,
                                    P,
                                    grad_P,
                                    std::move(local_hess)});
                                total_w += mollifier.val;
                                total_p += mollifier.val * P;
                            }
                        }
                    }

                    total_w += 1.;
                    if (auto iter = collisions.face_collisions.find(f); iter != collisions.face_collisions.end()) {
                        ConcatMatrixView<3> X_face(X, face_center);
                        ConstHessEntry entry;
                        entry.vertex_ids = (*iter->second).vertex_ids();
                        entry.dofs = (*iter->second).dofs();
                        entry.P = PointPotentialHelper::evaluate_potential_at_face_center_with_cached_collisions(
                            X_face, (*iter->second), params);
                        entry.grad_P = PointPotentialHelper::evaluate_potential_gradient_at_face_center_with_cached_collisions(
                            X_face, (*iter->second), params);
                        entry.local_hess = PointPotentialHelper::evaluate_potential_hessian_at_face_center_with_cached_collisions(
                            X_face, (*iter->second), params, project_hessian_to_psd);
                        total_p += entry.P;
                        const_cache.push_back(std::move(entry));
                    }

                    for (index_t lv = 0; lv < 3; lv++) {
                        const index_t v = mesh.faces()(f, lv);
                        total_w += 1.;
                        if (auto iter = collisions.vertex_collisions.find(v); iter != collisions.vertex_collisions.end()) {
                            ConstHessEntry entry;
                            entry.vertex_ids = (*iter->second).vertex_ids();
                            entry.dofs = (*iter->second).dofs();
                            entry.P = PointPotentialHelper::evaluate_potential_at_vertex_with_cached_collisions(
                                X, (*iter->second), params);
                            entry.grad_P = PointPotentialHelper::evaluate_potential_gradient_at_vertex_with_cached_collisions(
                                X, (*iter->second), params);
                            entry.local_hess = PointPotentialHelper::evaluate_potential_hessian_at_vertex_with_cached_collisions(
                                X, (*iter->second), params, project_hessian_to_psd);
                            total_p += entry.P;
                            const_cache.push_back(std::move(entry));
                        }
                    }

                    // Pass 2: apply hessian
                    assert(total_w > 0);
                    if (normalize_weights) {
                        // Exact quotient rule: H(w*p/Z) = (w/Z)*H(p) - (w*avg_P/Z)*H(Z) - (w/Z²)*sym(G⊗∇Z)
                        // where Z = total_w, ∇Z = Σ_i mol_grad_i, G = ∇p - avg_P*∇Z
                        const double avg_P = total_p / total_w;
                        const double scale_C = -(w / (total_w * total_w));

                        // Adds scale_C * sym(outer(g_vec, gradz_vec)) to triplets.
                        auto add_sym_correction = [&](
                            const std::vector<index_t>& g_dofs,
                            const Eigen::Ref<const Eigen::VectorXd>& g_vec,
                            const std::vector<index_t>& gradz_dofs,
                            const Eigen::Ref<const Eigen::VectorXd>& gradz_vec) {
                            for (int a = 0; a < static_cast<int>(g_dofs.size()); a++) {
                                for (int b = 0; b < static_cast<int>(gradz_dofs.size()); b++) {
                                    const double v = scale_C * g_vec[a] * gradz_vec[b];
                                    hess_triplets.cache->add_value(0, g_dofs[a], gradz_dofs[b], v);
                                    hess_triplets.cache->add_value(0, gradz_dofs[b], g_dofs[a], v);
                                }
                            }
                        };

                        // Term A: (w/total_w) * H(p_sum)
                        for (const auto& e : ee_cache) {
                            local_hessian_to_global_triplets(
                                (w / total_w) * e.local_hess, e.dict->vertex_ids(), dim,
                                *(hess_triplets.cache));
                        }
                        for (const auto& e : const_cache) {
                            local_hessian_to_global_triplets(
                                (w / total_w) * e.local_hess, e.vertex_ids, dim,
                                *(hess_triplets.cache));
                        }

                        // Term B: -(w*avg_P/Z) * Σ_i H(mol_i)
                        for (const auto& e : ee_cache) {
                            local_hessian_to_global_triplets(
                                -(w * avg_P / total_w) * e.mol_hess,
                                e.dict->primary_vertex_ids(), dim,
                                *(hess_triplets.cache));
                        }

                        // Term C: -(w/Z²) * sym(G⊗∇Z)
                        for (const auto& ei : ee_cache) {
                            const std::vector<index_t> prim_dofs_i = ei.dict->primary_dofs();
                            const Eigen::Vector<double, 12> mol_grad_i = ei.mol_grad;
                            for (const auto& ek : ee_cache) {
                                add_sym_correction(
                                    ek.dict->primary_dofs(),
                                    (ek.P - avg_P) * ek.mol_grad,
                                    prim_dofs_i, mol_grad_i);
                                add_sym_correction(
                                    ek.dict->dofs(),
                                    ek.mol_val * ek.grad_P,
                                    prim_dofs_i, mol_grad_i);
                            }
                            for (const auto& ej : const_cache) {
                                add_sym_correction(ej.dofs, ej.grad_P, prim_dofs_i, mol_grad_i);
                            }
                        }
                    } else {
                        // Unnormalized: H(w*p_sum) = w * H(p_sum)
                        for (const auto& e : ee_cache) {
                            local_hessian_to_global_triplets(
                                w * e.local_hess, e.dict->vertex_ids(), dim,
                                *(hess_triplets.cache));
                        }
                        for (const auto& e : const_cache) {
                            local_hessian_to_global_triplets(
                                w * e.local_hess, e.vertex_ids, dim,
                                *(hess_triplets.cache));
                        }
                    }
                }
            };

            maybe_parallel_for(mesh.num_faces(), loop_body);
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