#include "high_order_collisions.hpp"

#include "high_order_collisions_builder.hpp"

#include <algorithm>
#include <numeric>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_line.hpp>
#include <ipc/distance/point_point.hpp>
#include <ipc/utils/local_to_global.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>
#include <ipc/utils/maybe_parallel_for.hpp>
#include <ipc/high_order_contact/quadrature_potential.hpp>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <stdexcept> // std::out_of_range
#include <utility>

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
    BroadPhase* broad_phase)
{
    throw std::logic_error("Please don't use adaptive dhat right now"); //TODO enable
    assert(vertices.rows() == mesh.num_vertices());

    const double dhat = params.dhat;
    double inflation_radius = dhat / 2;

    // Candidates m_candidates;
    m_candidates.build(mesh, vertices, inflation_radius, broad_phase, true);
    m_candidates.convert_candidates_to_sets();
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
        if (use_adaptive_dhat) {
            log_and_throw_error("Adaptive dhat with exact cancellation is not implemented!");
        }
        auto storage = create_thread_storage<HighOrderCollisionsBuilder<2>>(
            HighOrderCollisionsBuilder<2>());
        // add all EV collision pairs for adjacent vertices
        std::vector<EdgeVertexCandidate> ev_candidates;
        ev_candidates.reserve(candidates.ev_candidates.size() + mesh.num_edges()*2);
        std::copy(candidates.ev_candidates.begin(), candidates.ev_candidates.end(), std::back_inserter(ev_candidates));
        for (index_t ei = 0; ei < mesh.num_edges(); ei++) {
            for (int j = 0; j < 2; j++) {
                ev_candidates.emplace_back(ei, mesh.edges()(ei, j));
            }
        }
        if (candidates.ev_candidates.size() + mesh.num_edges()*2 != ev_candidates.size()) throw std::logic_error("unexpected size of ev_candidates" + std::to_string(ev_candidates.size()) + " != " + std::to_string(candidates.ev_candidates.size() + mesh.num_edges()*2));
        maybe_parallel_for(
            ev_candidates.size(),
            [&](int start, int end, int thread_id) {
                HighOrderCollisionsBuilder<2>& local_storage =
                    get_local_thread_storage(storage, thread_id);
                local_storage.add_edge_vertex_collisions(
                    mesh, vertices, ev_candidates, params, vert_dhat,
                    edge_dhat, start, end);
            });
        // build set of EE candidates from EV candidates
        // start with sets to filter duplicates
        std::vector<std::set<index_t>> ee_candidates_set;
        ee_candidates_set.resize(mesh.num_edges());
        const auto &ve_adj = mesh.vertex_edge_adjacencies();
        for (const auto& [ei, vi] : ev_candidates) {
            for (const auto &ej : ve_adj[vi]) {
                if (ei != ej) {
                    ee_candidates_set[ei].insert(ej);
                    ee_candidates_set[ej].insert(ei);
                }
            }
        }
        std::vector<EdgeEdgeCandidate> ee_candidates;
        //each edge gets at least its two neighbors, potentially more
        ee_candidates.reserve(mesh.num_edges()*3);
        for (index_t ei=0; ei<mesh.num_edges(); ++ei) {
            for (const index_t ej : ee_candidates_set[ei]) {
                ee_candidates.emplace_back(ei, ej);
            }
        }
        maybe_parallel_for(
            ee_candidates.size(),
            [&](int start, int end, int thread_id) {
                HighOrderCollisionsBuilder<2>& local_storage =
                    get_local_thread_storage(storage, thread_id);
                local_storage.add_edge_edge_collisions(
                    mesh, vertices, ee_candidates, params, vert_dhat,
                    edge_dhat, start, end);
            });
        HighOrderCollisionsBuilder<2>::merge(storage, *this);
    }
    else {
        /*auto is_active = [offset_sqr = dhat * dhat](double distance_sqr) {
            return distance_sqr < offset_sqr;
        };*/

/* prepare collision sets to compute each P(q) */
        // compute masks
        std::vector<bool> vertex_mask(mesh.num_vertices(), false);
        for (const auto& candidate : candidates.fv_candidates) {
            vertex_mask[candidate.vertex_id] = true;
        }
        std::vector<index_t> vertices_to_process;
        vertices_to_process.reserve(mesh.num_vertices());
        for (int i = 0; i < mesh.num_vertices(); ++i) {
            if (vertex_mask[i]) {
                vertices_to_process.push_back(i);
            }
        }

        std::vector<index_t> faces_to_process;
        if (params.quad_order > 0) {
            faces_to_process.resize(mesh.num_faces());
            std::iota(faces_to_process.begin(), faces_to_process.end(), 0);
        }

        // create builder and parallel loops
        auto storage = create_thread_storage<QuadratureCollisionsBuilder>(
            QuadratureCollisionsBuilder(mesh, candidates, params));

        maybe_parallel_for(
            vertices_to_process.size(),
            [&](int start, int end, int thread_id) {
                QuadratureCollisionsBuilder& local_storage =
                    get_local_thread_storage(storage, thread_id);
                local_storage.build_vertex_collisions(
                    vertices, vertices_to_process, start, end);
            });

        if (params.quad_order > 0) {
            maybe_parallel_for(
                faces_to_process.size(),
                [&](int start, int end, int thread_id) {
                    QuadratureCollisionsBuilder& local_storage =
                        get_local_thread_storage(storage, thread_id);
                    local_storage.build_face_collisions(
                        vertices, faces_to_process, start, end);
                });
        }

        maybe_parallel_for(
            candidates.ee_candidates.size(),
            [&](int start, int end, int thread_id) {
                QuadratureCollisionsBuilder& local_storage =
                    get_local_thread_storage(storage, thread_id);
                local_storage.build_edge_edge_collisions(
                    vertices, candidates.ee_candidates, start, end);
            });

        QuadratureCollisionsBuilder::merge(storage, *this);
    }
    m_candidates = candidates;
}

void HighOrderCollisions::build(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const HighOrderContactParameters params,
    const bool use_adaptive_dhat,
    BroadPhase* broad_phase)
{
    assert(vertices.rows() == mesh.num_vertices());

    double inflation_radius = params.dhat / 2;  //TODO use dbar for EE collisions broad phase

    // Candidates m_candidates;
    m_candidates.build(mesh, vertices, inflation_radius, broad_phase, true);
    m_candidates.convert_candidates_to_sets();
    this->build(m_candidates, mesh, vertices, params, use_adaptive_dhat);
}

// ============================================================================
size_t HighOrderCollisions::size() const
{
    if (collisions.size() > 0) {
        return collisions.size();
    }
    else {
        size_t size = 0;
        for (const auto& cc : vertex_collisions) {
            size += cc.second->size();
        }
        for (const auto& cc : edge_edge_collisions) {
            size += cc.second->size();
        }
        for (const auto& cc : face_collisions) {
            for (const auto& dict_ptr : cc.second) {
                size += dict_ptr->size();
            }
        }
        return size;
    }
}
bool HighOrderCollisions::empty() const { return collisions.empty() && vertex_collisions.empty() && edge_edge_collisions.empty() && face_collisions.empty(); }
void HighOrderCollisions::clear()
{
    collisions.clear();

    vertex_collisions.clear();
    edge_edge_collisions.clear();
    face_collisions.clear();
}

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

    for (const auto& ccs : vertex_collisions) {
        for (int i = 0; i < (*ccs.second).size(); i++) {
            const auto& cc = (*ccs.second)[i];
            ss << "\n";
            {
                ss << fmt::format(
                    "vert [{}]: ({} {}) weight {} dist sqr {} potential {} grad {}", cc.name(),
                    cc[0], cc[1], cc.weight, cc.compute_distance(vertices),
                    cc(cc.dof(vertices), params),
                    cc.gradient(cc.dof(vertices), params).norm());
            }
        }
    }
    for (const auto& ccs : edge_edge_collisions) {
        for (int i = 0; i < (*ccs.second).size(); i++) {
            const auto& cc = (*ccs.second)[i];
            ss << "\n";
            {
                ss << fmt::format(
                    "edge [{}]: ({} {}) ({} {}) weight {}", cc.name(),
                    ccs.first.first, ccs.first.second,
                    cc[0], cc[1], cc.weight);
            }
        }
    }
    for (const auto& ccs : face_collisions) {
        for (const auto& dict_ptr : ccs.second) {
            for (int i = 0; i < dict_ptr->size(); i++) {
                const auto& cc = (*dict_ptr)[i];
                ss << "\n";
                {
                    ss << fmt::format(
                        "face [{}]: ({} {}) weight {} dist sqr {} potential {} grad {}", cc.name(),
                        cc[0], cc[1], cc.weight, cc.compute_distance(vertices),
                        cc(cc.dof(vertices), params),
                        cc.gradient(cc.dof(vertices), params).norm());
                }
            }
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

    if (empty()) {
        return std::numeric_limits<double>::infinity();
    }

    tbb::enumerable_thread_specific<double> storage(
        std::numeric_limits<double>::infinity());

    if (mesh.dim() == 2) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, collisions.size()),
            [&](tbb::blocked_range<size_t> r) {
                double& local_min_dist = storage.local();

                for (size_t i = r.begin(); i < r.end(); i++) {
                    const double dist = collisions[i]->compute_distance(vertices);
                    local_min_dist = std::min(dist, local_min_dist);
                }
            });
    }
    else {
        const double bbox_diag = world_bbox_diagonal_length(vertices);
        double min_dist = bbox_diag * bbox_diag;
        for (const auto& map : vertex_collisions) {
            for (int i = 0; i < (*map.second).size(); i++) {
                const auto& cc = (*map.second)[i];
                min_dist = std::min(min_dist, cc.compute_distance(vertices));
            }
        }

        // TODO

        // for (const auto& map : face_collisions) {
        //     for (const auto& cc : (*map.second)) {
        //         min_dist = std::min(min_dist, cc.second->compute_distance(vertices));
        //     }
        // }

        // for (const auto& map : edge_edge_collisions) {
        //     for (const auto& cc : (*map.second)) {
        //         min_dist = std::min(min_dist, cc.second->compute_distance(vertices));
        //     }
        // }

        return min_dist;
    }

    return storage.combine([](double a, double b) { return std::min(a, b); });
}

std::map<size_t, size_t> HighOrderCollisions::edge_id_count_distribution() const
{
    unordered_map<index_t, size_t> counts;
    for (const auto& [key, _] : edge_edge_collisions) {
        counts[key.first]++;
    }

    std::map<size_t, size_t> distribution;
    for (const auto& [_, count] : counts) {
        distribution[count]++;
    }
    return distribution;
}

Eigen::VectorXd HighOrderCollisions::edge_collision_counts(size_t num_edges) const
{
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(num_edges);
    for (const auto& [key, _] : edge_edge_collisions) {
        counts(key.first)++;
    }
    return counts;
}

} // namespace ipc
