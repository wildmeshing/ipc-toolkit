#include "high_order_collisions.hpp"

#include "high_order_collisions_builder.hpp"

#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_line.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/utils/local_to_global.hpp>
#include <ipc/utils/maybe_parallel_for.hpp>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <stdexcept> // std::out_of_range

namespace ipc {
namespace
{
    template <typename Candidate>
    std::vector<VertexVertexCandidate>
    element_vertex_to_vertex_vertex_candidates(
        Eigen::ConstRef<Eigen::MatrixXi> elements,
        Eigen::ConstRef<Eigen::MatrixXd> vertices,
        const std::vector<Candidate>& candidates,
        const std::function<bool(double)>& is_active)
    {
        std::vector<VertexVertexCandidate> vv_candidates;
        for (const auto& [ei, vi] : candidates) {
            for (int j = 0; j < elements.cols(); j++) {
                const int vj = elements(ei, j);
                if (is_active(point_point_distance(
                    vertices.row(vi), vertices.row(vj)))) {
                    vv_candidates.emplace_back(std::min(vi, vj), std::max(vi, vj));
                }
            }
        }

        // Remove duplicates
        tbb::parallel_sort(vv_candidates.begin(), vv_candidates.end());
        vv_candidates.erase(
            std::unique(vv_candidates.begin(), vv_candidates.end()),
            vv_candidates.end());

        return vv_candidates;
    }

    std::vector<VertexVertexCandidate> face_vertex_to_vertex_vertex_candidates(
        const CollisionMesh& mesh,
        Eigen::ConstRef<Eigen::MatrixXd> vertices,
        const std::vector<FaceVertexCandidate>& fv_candidates,
        const std::function<bool(double)>& is_active)
    {
        return element_vertex_to_vertex_vertex_candidates(
            mesh.faces(), vertices, fv_candidates, is_active);
    }

    std::vector<EdgeVertexCandidate> face_vertex_to_edge_vertex_candidates(
        const CollisionMesh& mesh,
        Eigen::ConstRef<Eigen::MatrixXd> vertices,
        const std::vector<FaceVertexCandidate>& fv_candidates,
        const std::function<bool(double)>& is_active)
    {
        std::vector<EdgeVertexCandidate> ev_candidates;
        for (const auto& [fi, vi] : fv_candidates) {
            for (int j = 0; j < 3; j++) {
                const int ei = mesh.faces_to_edges()(fi, j);
                const int vj = mesh.edges()(ei, 0);
                const int vk = mesh.edges()(ei, 1);
                if (is_active(point_edge_distance(vertices.row(vi), vertices.row(vj),
                                                  vertices.row(vk)))) {
                    ev_candidates.emplace_back(ei, vi);
                }
            }
        }

        // Remove duplicates
        tbb::parallel_sort(ev_candidates.begin(), ev_candidates.end());
        ev_candidates.erase(
            std::unique(ev_candidates.begin(), ev_candidates.end()),
            ev_candidates.end());

        return ev_candidates;
    }
}

void HighOrderCollisions::compute_adaptive_dhat(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices, // set to zero for rest pose
    const HighOrderContactParameters params,
    const std::shared_ptr<BroadPhase>& broad_phase)
{
    assert(vertices.rows() == mesh.num_vertices());

    const double dhat = params.dhat;
    double inflation_radius = dhat / 2;

    // Candidates m_candidates;
    m_candidates.build(mesh, vertices, inflation_radius, broad_phase);
    this->build(
        m_candidates, mesh, vertices, params,
        false /*disable adaptive dhat to compute true pairs*/);

    vert_adaptive_dhat.setConstant(mesh.num_vertices(), dhat);
    edge_adaptive_dhat.setConstant(mesh.num_edges(), dhat);
    if (mesh.dim() == 3) {
        face_adaptive_dhat.setConstant(mesh.num_faces(), dhat);
    } else {
        face_adaptive_dhat.resize(0);
    }

    auto assign_min = [](double& a, const double b) -> void {
        a = std::min(a, b);
    };

    for (const auto& cc : collisions) {
        const double dist =
            params.adaptive_dhat_ratio() * sqrt(cc->compute_distance(vertices));
        switch (cc->type()) {
        case HighOrderCollisionType::EDGE_EDGE: {
            assign_min(edge_adaptive_dhat((*cc)[0]), dist);
            assign_min(edge_adaptive_dhat((*cc)[1]), dist);
            break;
        }
        case HighOrderCollisionType::EDGE_VERTEX: {
            assign_min(edge_adaptive_dhat((*cc)[0]), dist);
            assign_min(vert_adaptive_dhat((*cc)[1]), dist);
            break;
        }
        case HighOrderCollisionType::FACE_VERTEX: {
            assign_min(face_adaptive_dhat((*cc)[0]), dist);
            assign_min(vert_adaptive_dhat((*cc)[1]), dist);
            break;
        }
        case HighOrderCollisionType::VERTEX_VERTEX: {
            assign_min(vert_adaptive_dhat((*cc)[0]), dist);
            assign_min(vert_adaptive_dhat((*cc)[1]), dist);
            break;
        }
        default: {
            throw std::runtime_error("Invalid collision type!");
        }
        }
    }

    // face adaptive dhat should be minimum of all its adjacent vertices and
    // edges
    if (mesh.dim() == 3) {
        for (int f = 0; f < mesh.num_faces(); f++) {
            for (int lv = 0; lv < 3; lv++) {
                face_adaptive_dhat(f) = std::min(
                    face_adaptive_dhat(f),
                    vert_adaptive_dhat(mesh.faces()(f, lv)));
                face_adaptive_dhat(f) = std::min(
                    face_adaptive_dhat(f),
                    edge_adaptive_dhat(mesh.faces_to_edges()(f, lv)));
            }
        }
    }

    // edge adaptive dhat should be minimum of all its adjacent vertices
    for (int e = 0; e < mesh.num_edges(); e++) {
        for (int lv = 0; lv < 2; lv++) {
            edge_adaptive_dhat(e) = std::min(
                edge_adaptive_dhat(e), vert_adaptive_dhat(mesh.edges()(e, lv)));
        }
    }

    logger().debug(
        "Adaptive dhat: vert dhat min {:.2e}, max {:.2e}",
        vert_adaptive_dhat.minCoeff(), vert_adaptive_dhat.maxCoeff());
    logger().debug(
        "Adaptive dhat: edge dhat min {:.2e}, max {:.2e}",
        edge_adaptive_dhat.minCoeff(), edge_adaptive_dhat.maxCoeff());
    if (mesh.dim() == 3) {
        logger().debug(
            "Adaptive dhat: face dhat min {:.2e}, max {:.2e}",
            face_adaptive_dhat.minCoeff(), face_adaptive_dhat.maxCoeff());
    }
}

void HighOrderCollisions::build(
    const Candidates& candidates,
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const HighOrderContactParameters params,
    const bool use_adaptive_dhat)
{
    assert(vertices.rows() == mesh.num_vertices());

    clear();

    const double dhat = params.dhat;
    if (!use_adaptive_dhat) {
        vert_adaptive_dhat.resize(1);
        vert_adaptive_dhat(0) = dhat;
        edge_adaptive_dhat.resize(1);
        edge_adaptive_dhat(0) = dhat;
        if (mesh.dim() == 3) {
            face_adaptive_dhat.resize(1);
            face_adaptive_dhat(0) = dhat;
        } else {
            face_adaptive_dhat.resize(0);
        }
    }

    auto vert_dhat = [&](const index_t v_id) {
        return this->get_vert_dhat(v_id);
    };
    auto edge_dhat = [&](const index_t e_id) {
        return this->get_edge_dhat(e_id);
    };

    if (mesh.dim() == 2) {
        auto storage = create_thread_storage<HighOrderCollisionsBuilder<2>>(
            HighOrderCollisionsBuilder<2>());
        maybe_parallel_for(
            candidates.ev_candidates.size(),
            [&](int start, int end, int thread_id) {
                HighOrderCollisionsBuilder<2>& local_storage =
                    get_local_thread_storage(storage, thread_id);
                local_storage.add_edge_vertex_collisions(
                    mesh, vertices, candidates.ev_candidates, params, vert_dhat,
                    edge_dhat, start, end);
            });
        HighOrderCollisionsBuilder<2>::merge(storage, *this);
    }
        else {
            if (use_adaptive_dhat) {
                log_and_throw_error("Adaptive dhat with exact cancellation is not implemented!");
            }

            auto is_active = [offset_sqr = dhat * dhat](double distance_sqr) {
                return distance_sqr < offset_sqr;
            };

            auto storage = create_thread_storage<HighOrderCollisionsBuilder<3>>(
                HighOrderCollisionsBuilder<3>());

            // Integral over vertices

            std::vector<std::vector<index_t>> face_ids_close_to_v(mesh.num_vertices());
            for (auto candidate : candidates.fv_candidates) {
                face_ids_close_to_v[candidate.vertex_id].push_back(candidate.face_id);
            }

            std::vector<std::vector<index_t>> vertex_ids_close_to_f(mesh.num_faces());
            for (auto candidate : candidates.fv_candidates) {
                vertex_ids_close_to_f[candidate.face_id].push_back(candidate.vertex_id);
            }

            maybe_parallel_for(
                candidates.fv_candidates.size(),
                [&](int start, int end, int thread_id) {
                    HighOrderCollisionsBuilder<3>& local_storage =
                        get_local_thread_storage(storage, thread_id);
                    local_storage.add_face_vertex_collisions(
                        mesh, vertices, candidates.fv_candidates, params, start, end);
                });

            // This for loop is an inefficient hack that we should get rid of
            for (index_t hack_id = 0; hack_id < mesh.num_vertices(); hack_id++) {
                std::vector<FaceVertexCandidate> fv_candidates;
                for (auto candidate : candidates.fv_candidates)
                    if (candidate.vertex_id == hack_id)
                        fv_candidates.push_back(candidate);

                if (fv_candidates.empty())
                    continue;

                // Convert face-vertex to edge-vertex
                const std::vector<EdgeVertexCandidate> ev_candidates =
                    face_vertex_to_edge_vertex_candidates(
                        mesh, vertices, fv_candidates, is_active);

                maybe_parallel_for(
                    ev_candidates.size(),
                    [&](int start, int end, int thread_id) {
                        HighOrderCollisionsBuilder<3>& local_storage =
                            get_local_thread_storage(storage, thread_id);
                        local_storage.add_face_vertex_negative_edge_vertex_collisions(
                            mesh, vertices, ev_candidates, params, start, end);
                    });

                // Convert face-vertex to vertex-vertex
                const std::vector<VertexVertexCandidate> vv_candidates =
                    face_vertex_to_vertex_vertex_candidates(
                        mesh, vertices, fv_candidates, is_active);

                maybe_parallel_for(
                    vv_candidates.size(),
                    [&](int start, int end, int thread_id) {
                        HighOrderCollisionsBuilder<3>& local_storage =
                            get_local_thread_storage(storage, thread_id);
                        local_storage.add_face_vertex_positive_vertex_vertex_collisions(
                            mesh, vertices, vv_candidates, params, start, end);
                    });
            }

        // Integral over edge-edge pairs

        // This for loop is an inefficient hack that we should get rid of
        for (index_t hack_id = 0; hack_id < mesh.num_edges(); hack_id++) {
            std::vector<EdgeEdgeCandidate> ee_candidates;
            for (auto candidate : candidates.ee_candidates) {
                if (candidate.edge0_id == hack_id || candidate.edge1_id == hack_id) {
                    ee_candidates.push_back(candidate);
                }
            }

            if (ee_candidates.empty()) {
                continue;
            }

            // Find V, E, F that are close to hack_id
            std::set<index_t> vids, eids, fids;
            {
                for (auto candidate : candidates.ef_candidates) {
                    if (candidate.edge_id == hack_id) {
                        fids.insert(candidate.face_id);
                    }
                }

                for (int i = 0; i < 2; i++) {
                    for (int fid : mesh.vertices_to_faces()[mesh.edges()(hack_id, i)]) {
                        if (mesh.faces_to_edges()(fid, 0) != hack_id &&
                            mesh.faces_to_edges()(fid, 1) != hack_id &&
                            mesh.faces_to_edges()(fid, 2) != hack_id) {
                            fids.insert(fid);
                        }
                    }
                }

                for (int i = 0; i < 2; i++) {
                    for (int ei : mesh.vertices_to_edges()[mesh.edges()(hack_id, i)]) {
                        if (ei != hack_id) {
                            eids.insert(ei);
                        }
                    }
                }

                for (auto candidate1 : ee_candidates) {
                    const index_t ei = candidate1.edge0_id == hack_id ? candidate1.edge1_id : candidate1.edge0_id;
                    eids.insert(ei);

                    for (int i = 0; i < 2; i++) {
                        const index_t vi = mesh.edges()(ei, i);
                        vids.insert(vi);
                    }
                }

                for (int i = 0; i < 2; i++) {
                    if (int fi = mesh.edges_to_faces()(hack_id, i); fi >= 0) {
                        for (int vi : vertex_ids_close_to_f[fi]) {
                            vids.insert(vi);
                        }
                    }
                }

                for (int vi : mesh.edge_vertex_adjacencies()[hack_id]) {
                    vids.insert(vi);
                }

                vids.insert(mesh.edges()(hack_id, 0));
                vids.insert(mesh.edges()(hack_id, 1));
            }

            // debug starts -----

            // for (int i = 0; i < mesh.num_vertices(); i++) {
            //     vids.insert(i);
            // }
            // for (int i = 0; i < mesh.num_edges(); i++) {
            //     if (i != hack_id) {
            //         eids.insert(i);
            //     }
            // }
            // for (int i = 0; i < mesh.num_faces(); i++) {
            //     if (mesh.faces_to_edges()(i, 0) != hack_id &&
            //         mesh.faces_to_edges()(i, 1) != hack_id &&
            //         mesh.faces_to_edges()(i, 2) != hack_id) {
            //         fids.insert(i);
            //     }
            // }

            // debug ends -----

            // EE candidates become three types of terms: EEV, EEE, EEF
            std::vector<std::array<index_t, 3>> triplets_eev, triplets_eee, triplets_eef;
            for (auto candidate1 : ee_candidates) {
                const index_t other_e = candidate1.edge0_id == hack_id ? candidate1.edge1_id : candidate1.edge0_id;

                for (index_t vi : vids) {
                    triplets_eev.push_back(std::array<index_t, 3>{{hack_id, other_e, vi}});
                }

                for (index_t ei : eids) {
                    triplets_eee.push_back(std::array<index_t, 3>{{hack_id, other_e, ei}});
                }

                for (index_t fi : fids) {
                    triplets_eef.push_back(std::array<index_t, 3>{{hack_id, other_e, fi}});
                }
            }

            maybe_parallel_for(
                triplets_eef.size(),
                [&](int start, int end, int thread_id)
                {
                    HighOrderCollisionsBuilder<3>& local_storage =
                        get_local_thread_storage(storage, thread_id);
                    local_storage.add_edge_edge_face_collisions(
                        mesh, vertices, triplets_eef, params, dhat, start, end);
                });

            maybe_parallel_for(
                triplets_eee.size(),
                [&](int start, int end, int thread_id)
                {
                    HighOrderCollisionsBuilder<3>& local_storage =
                        get_local_thread_storage(storage, thread_id);
                    local_storage.add_negative_edge_edge_edge_collisions(
                        mesh, vertices, triplets_eee, params, dhat, start, end);
                });

            maybe_parallel_for(
                triplets_eev.size(),
                [&](int start, int end, int thread_id)
                {
                    HighOrderCollisionsBuilder<3>& local_storage =
                        get_local_thread_storage(storage, thread_id);
                    local_storage.add_edge_edge_vertex_collisions(
                        mesh, vertices, triplets_eev, params, dhat, start, end);
                });
        }

        HighOrderCollisionsBuilder<3>::merge(storage, *this);
    }
    m_candidates = candidates;
}

void HighOrderCollisions::build(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const HighOrderContactParameters params,
    const bool use_adaptive_dhat,
    const std::shared_ptr<BroadPhase>& broad_phase)
{
    assert(vertices.rows() == mesh.num_vertices());

    double inflation_radius = params.dhat / 2;

    // Candidates m_candidates;
    m_candidates.build(mesh, vertices, inflation_radius, broad_phase);
    this->build(m_candidates, mesh, vertices, params, use_adaptive_dhat);
}

// ============================================================================
size_t HighOrderCollisions::size() const { return collisions.size(); }
bool HighOrderCollisions::empty() const { return collisions.empty() && triple_collisions.empty(); }
void HighOrderCollisions::clear() { collisions.clear(); }

HighOrderCollision& HighOrderCollisions::operator[](size_t i)
{
    if (i < collisions.size()) {
        return *collisions[i];
    }
    throw std::out_of_range("Collision index is out of range!");
}

const HighOrderCollision& HighOrderCollisions::operator[](size_t i) const
{
    if (i < collisions.size()) {
        return *collisions[i];
    }
    throw std::out_of_range("Collision index is out of range!");
}

std::string HighOrderCollisions::to_string(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const HighOrderContactParameters& params) const
{
    std::stringstream ss;
    for (const auto& cc : collisions) {
        ss << "\n";
        {
            ss << fmt::format(
                "[{}]: ({} {}) weight {} dist {} potential {} grad {}", cc->name(),
                (*cc)[0], (*cc)[1], cc->weight, cc->compute_distance(vertices),
                (*cc)(cc->dof(vertices), params),
                (*cc).gradient(cc->dof(vertices), params).norm());
        }
    }
    for (const auto& cc : triple_collisions) {
        ss << "\n";
        {
            ss << fmt::format(
                "[{}]: ({} {} {}) weight {} potential {} grad {}", cc->name(),
                (*cc)[0], (*cc)[1], (*cc)[2], cc->weight,
                (*cc)(cc->dof(vertices), params),
                (*cc).gradient(cc->dof(vertices), params).norm());
        }
    }
    return ss.str();
}

// NOTE: Actually distance squared
double HighOrderCollisions::compute_minimum_distance(
    const CollisionMesh& mesh, Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    assert(vertices.rows() == mesh.num_vertices());

    if (m_candidates.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    const Eigen::MatrixXi& edges = mesh.edges();
    const Eigen::MatrixXi& faces = mesh.faces();

    tbb::enumerable_thread_specific<double> storage(
        std::numeric_limits<double>::infinity());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, m_candidates.size()),
        [&](tbb::blocked_range<size_t> r) {
            double& local_min_dist = storage.local();

            for (size_t i = r.begin(); i < r.end(); i++) {
                const double dist = m_candidates[i].compute_distance(
                    m_candidates[i].dof(vertices, edges, faces));

                local_min_dist = std::min(dist, local_min_dist);
            }
        });

    return storage.combine([](double a, double b) { return std::min(a, b); });
}

double HighOrderCollisions::compute_active_minimum_distance(
    const CollisionMesh& mesh, Eigen::ConstRef<Eigen::MatrixXd> vertices) const
{
    assert(vertices.rows() == mesh.num_vertices());

    if (collisions.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    tbb::enumerable_thread_specific<double> storage(
        std::numeric_limits<double>::infinity());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, collisions.size()),
        [&](tbb::blocked_range<size_t> r) {
            double& local_min_dist = storage.local();

            for (size_t i = r.begin(); i < r.end(); i++) {
                const double dist = collisions[i]->compute_distance(vertices);

                if (collisions[i]->is_active() && dist < local_min_dist) {
                    local_min_dist = dist;
                }
            }
        });

    return storage.combine([](double a, double b) { return std::min(a, b); });
}

} // namespace ipc
