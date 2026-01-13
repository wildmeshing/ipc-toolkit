#include "candidates.hpp"

#include <ipc/config.hpp>
#include <ipc/ipc.hpp>
#include <ipc/broad_phase/default_broad_phase.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/utils/save_obj.hpp>

#include <igl/remove_unreferenced.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <fstream>
#include <shared_mutex>

namespace ipc {

namespace {
    // Pad codim_edges because remove_unreferenced requires a N×3 matrix.
    Eigen::MatrixXi pad_edges(Eigen::ConstRef<Eigen::MatrixXi> E)
    {
        assert(E.cols() == 2);
        Eigen::MatrixXi E_padded(E.rows(), 3);
        E_padded.leftCols(2) = E;
        E_padded.col(2) = E.col(1);
        return E_padded;
    }

    Eigen::MatrixXi unpad_edges(Eigen::ConstRef<Eigen::MatrixXi> E_padded)
    {
        return E_padded.leftCols(2);
    }
} // namespace

void Candidates::build(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const double inflation_radius,
    const std::shared_ptr<BroadPhase>& broad_phase,
    const bool all_types)
{
    assert(broad_phase != nullptr);

    const int dim = vertices.cols();
    mesh_ = mesh;

    clear();

    broad_phase->can_vertices_collide = mesh.can_collide;
    broad_phase->build(vertices, mesh.edges(), mesh.faces(), inflation_radius);
    broad_phase->detect_collision_candidates(dim, *this, all_types);

    // Codim. vertices to codim. vertices:
    if (mesh.num_codim_vertices()) {
        broad_phase->clear();
        broad_phase->build(
            vertices(mesh.codim_vertices(), Eigen::all), //
            Eigen::MatrixXi(), Eigen::MatrixXi(), inflation_radius);

        broad_phase->detect_vertex_vertex_candidates(vv_candidates);
        for (auto& [vi, vj] : vv_candidates) {
            vi = mesh.codim_vertices()[vi];
            vj = mesh.codim_vertices()[vj];
        }
    }

    // Codim. edges to codim. vertices:
    // Only need this in 3D because in 2D, the codim. edges are the same as the
    // edges of the boundary. Only need codim. edge to codim. vertex because
    // codim. edge to non-codim. vertex is the same as edge-edge or face-vertex.
    if (dim == 3 && mesh.num_codim_vertices() && mesh.num_codim_edges()) {
        // Extract the vertices of the codim. edges
        Eigen::MatrixXd CE_V; // vertices of codim. edges
        Eigen::MatrixXi CE;   // codim. edges (indices into CEV)
        {
            Eigen::VectorXi _I, _J; // unused mappings
            igl::remove_unreferenced(
                vertices,
                pad_edges(mesh.edges()(mesh.codim_edges(), Eigen::all)), CE_V,
                CE, _I, _J);
            CE = unpad_edges(CE);
        }

        const size_t nCV = mesh.num_codim_vertices();
        Eigen::MatrixXd V(nCV + CE_V.rows(), dim);
        V.topRows(nCV) = vertices(mesh.codim_vertices(), Eigen::all);
        V.bottomRows(CE_V.rows()) = CE_V;

        CE.array() += nCV; // Offset indices to account for codim. vertices

        // TODO: Can we reuse the broad phase from above?
        broad_phase->clear();
        broad_phase->can_vertices_collide = [&](size_t vi, size_t vj) {
            // Ignore c-edge to c-edge and c-vertex to c-vertex
            return ((vi < nCV) ^ (vj < nCV)) && mesh.can_collide(vi, vj);
        };
        broad_phase->build(V, CE, Eigen::MatrixXi(), inflation_radius);

        broad_phase->detect_edge_vertex_candidates(ev_candidates);
        for (auto& [ei, vi] : ev_candidates) {
            assert(vi < mesh.codim_vertices().size());
            ei = mesh.codim_edges()[ei];    // Map back to mesh.edges
            vi = mesh.codim_vertices()[vi]; // Map back to vertices
        }
    }
}

void Candidates::build(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t0,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t1,
    const double inflation_radius,
    const std::shared_ptr<BroadPhase>& broad_phase,
    const bool all_types)
{
    assert(broad_phase != nullptr);

    const int dim = vertices_t0.cols();
    mesh_ = mesh;

    clear();

    broad_phase->can_vertices_collide = mesh.can_collide;
    broad_phase->build(
        vertices_t0, vertices_t1, mesh.edges(), mesh.faces(), inflation_radius);
    broad_phase->detect_collision_candidates(dim, *this);

    // Codim. vertices to codim. vertices:
    if (mesh.num_codim_vertices()) {
        broad_phase->clear();
        broad_phase->build(
            vertices_t0(mesh.codim_vertices(), Eigen::all),
            vertices_t1(mesh.codim_vertices(), Eigen::all), //
            Eigen::MatrixXi(), Eigen::MatrixXi(), inflation_radius);

        broad_phase->detect_vertex_vertex_candidates(vv_candidates);
        for (auto& [vi, vj] : vv_candidates) {
            vi = mesh.codim_vertices()[vi];
            vj = mesh.codim_vertices()[vj];
        }
    }

    // Codim. edges to codim. vertices:
    // Only need this in 3D because in 2D, the codim. edges are the same as the
    // edges of the boundary. Only need codim. edge to codim. vertex because
    // codim. edge to non-codim. vertex is the same as edge-edge or face-vertex.
    if (dim == 3 && mesh.num_codim_vertices() && mesh.num_codim_edges()) {
        // Extract the vertices of the codim. edges
        Eigen::MatrixXd CE_V_t0, CE_V_t1; // vertices of codim. edges
        Eigen::MatrixXi CE;               // codim. edges (indices into CEV)
        {
            Eigen::VectorXi _I, J;
            igl::remove_unreferenced(
                vertices_t0,
                pad_edges(mesh.edges()(mesh.codim_edges(), Eigen::all)),
                CE_V_t0, CE, _I, J);
            CE_V_t1 = vertices_t1(J, Eigen::all);
            CE = unpad_edges(CE);
        }

        const size_t nCV = mesh.num_codim_vertices();

        Eigen::MatrixXd V_t0(nCV + CE_V_t0.rows(), dim);
        V_t0.topRows(nCV) = vertices_t0(mesh.codim_vertices(), Eigen::all);
        V_t0.bottomRows(CE_V_t0.rows()) = CE_V_t0;

        Eigen::MatrixXd V_t1(nCV + CE_V_t1.rows(), dim);
        V_t1.topRows(nCV) = vertices_t1(mesh.codim_vertices(), Eigen::all);
        V_t1.bottomRows(CE_V_t1.rows()) = CE_V_t1;

        CE.array() += nCV; // Offset indices to account for codim. vertices

        // TODO: Can we reuse the broad phase from above?
        broad_phase->clear();
        broad_phase->can_vertices_collide = [&](size_t vi, size_t vj) {
            // Ignore c-edge to c-edge and c-vertex to c-vertex
            return ((vi < nCV) ^ (vj < nCV)) && mesh.can_collide(vi, vj);
        };
        broad_phase->build(V_t0, V_t1, CE, Eigen::MatrixXi(), inflation_radius);

        broad_phase->detect_edge_vertex_candidates(ev_candidates);
        for (auto& [ei, vi] : ev_candidates) {
            assert(vi < mesh.codim_vertices().size());
            ei = mesh.codim_edges()[ei];    // Map back to mesh.edges
            vi = mesh.codim_vertices()[vi]; // Map back to vertices
        }
    }
}

bool Candidates::is_step_collision_free(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t0,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t1,
    const double min_distance,
    const NarrowPhaseCCD& narrow_phase_ccd) const
{
    assert(vertices_t0.rows() == mesh.num_vertices());
    assert(vertices_t1.rows() == mesh.num_vertices());

    // Narrow phase
    for (size_t i = 0; i < size(); i++) {
        const CollisionStencil& candidate = (*this)[i];

        double toi;
        bool is_collision = candidate.ccd(
            candidate.dof(vertices_t0, mesh.edges(), mesh.faces()),
            candidate.dof(vertices_t1, mesh.edges(), mesh.faces()), //
            toi, min_distance, /*tmax=*/1.0, narrow_phase_ccd);

        if (is_collision) {
            return false;
        }
    }

    return true;
}

double Candidates::compute_collision_free_stepsize(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t0,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t1,
    const double min_distance,
    const NarrowPhaseCCD& narrow_phase_ccd) const
{
    assert(vertices_t0.rows() == mesh.num_vertices());
    assert(vertices_t1.rows() == mesh.num_vertices());

    if (empty()) {
        return 1; // No possible collisions, so can take full step.
    }

    double earliest_toi = 1;
    std::shared_mutex earliest_toi_mutex;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, size()),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                // Use the mutex to read as well in case writing double takes
                // more than one clock cycle.
                double tmax;
                {
                    std::shared_lock lock(earliest_toi_mutex);
                    tmax = earliest_toi;
                }

                const CollisionStencil& candidate = (*this)[i];

                double toi = std::numeric_limits<double>::infinity(); // output
                const bool are_colliding = candidate.ccd(
                    candidate.dof(vertices_t0, mesh.edges(), mesh.faces()),
                    candidate.dof(vertices_t1, mesh.edges(), mesh.faces()), //
                    toi, min_distance, tmax, narrow_phase_ccd);

                if (are_colliding) {
                    std::unique_lock lock(earliest_toi_mutex);
                    earliest_toi = std::min(earliest_toi, toi);
                }
            }
        });

    assert(earliest_toi >= 0 && earliest_toi <= 1.0);
    return earliest_toi;
}

double Candidates::compute_noncandidate_conservative_stepsize(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> displacements,
    const double dhat) const
{
    assert(displacements.rows() == mesh.num_vertices());

    if (empty()) {
        return 1; // No possible collisions, so can take full step.
    }

    const Eigen::MatrixXi& E = mesh.edges();
    const Eigen::MatrixXi& F = mesh.faces();

    std::vector<bool> is_vertex_a_candidates(mesh.num_vertices(), false);
    for (size_t i = 0; i < size(); i++) {
        for (const index_t vid : (*this)[i].vertex_ids(E, F)) {
            if (vid < 0) {
                break;
            }
            is_vertex_a_candidates[vid] = true;
        }
    }

    double max_displacement = 0;
    for (size_t i = 0; i < displacements.rows(); i++) {
        if (!is_vertex_a_candidates[i]) {
            continue;
        }
        max_displacement =
            std::max(max_displacement, displacements.row(i).norm());
    }

    return 0.5 * dhat / max_displacement;
}

double Candidates::compute_cfl_stepsize(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t0,
    Eigen::ConstRef<Eigen::MatrixXd> vertices_t1,
    const double dhat,
    const double min_distance,
    const std::shared_ptr<BroadPhase>& broad_phase,
    const NarrowPhaseCCD& narrow_phase_ccd) const
{
    assert(vertices_t0.rows() == mesh.num_vertices());
    assert(vertices_t1.rows() == mesh.num_vertices());

    const double alpha_C = this->compute_collision_free_stepsize(
        mesh, vertices_t0, vertices_t1, min_distance, narrow_phase_ccd);

    const double alpha_F = this->compute_noncandidate_conservative_stepsize(
        mesh, vertices_t1 - vertices_t0, dhat);

    // If alpha_F < 0.5 * alpha_C, then we should do full CCD.
    if (alpha_F < 0.5 * alpha_C) {
        return ipc::compute_collision_free_stepsize(
            mesh, vertices_t0, vertices_t1, min_distance, broad_phase,
            narrow_phase_ccd);
    }
    return std::min(alpha_C, alpha_F);
}

size_t Candidates::size() const
{
    return vv_candidates.size() + ev_candidates.size() + ee_candidates.size()
        + fv_candidates.size();
}

bool Candidates::empty() const
{
    return vv_candidates.empty() && ev_candidates.empty()
        && ee_candidates.empty() && fv_candidates.empty();
}

void Candidates::clear()
{
    vv_candidates.clear();
    ev_candidates.clear();
    ee_candidates.clear();
    fv_candidates.clear();
}

CollisionStencil& Candidates::operator[](size_t i)
{
    if (i < vv_candidates.size()) {
        return vv_candidates[i];
    }
    i -= vv_candidates.size();
    if (i < ev_candidates.size()) {
        return ev_candidates[i];
    }
    i -= ev_candidates.size();
    if (i < ee_candidates.size()) {
        return ee_candidates[i];
    }
    i -= ee_candidates.size();
    if (i < fv_candidates.size()) {
        return fv_candidates[i];
    }
    throw std::out_of_range("Candidate index is out of range!");
}

const CollisionStencil& Candidates::operator[](size_t i) const
{
    if (i < vv_candidates.size()) {
        return vv_candidates[i];
    }
    i -= vv_candidates.size();
    if (i < ev_candidates.size()) {
        return ev_candidates[i];
    }
    i -= ev_candidates.size();
    if (i < ee_candidates.size()) {
        return ee_candidates[i];
    }
    i -= ee_candidates.size();
    if (i < fv_candidates.size()) {
        return fv_candidates[i];
    }
    throw std::out_of_range("Candidate index is out of range!");
}

bool Candidates::save_obj(
    const std::string& filename,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    Eigen::ConstRef<Eigen::MatrixXi> edges,
    Eigen::ConstRef<Eigen::MatrixXi> faces) const
{
    std::ofstream obj(filename, std::ios::out);
    if (!obj.is_open()) {
        return false;
    }
    int v_offset = 0;
    ipc::save_obj(obj, vertices, edges, faces, vv_candidates, v_offset);
    v_offset += vv_candidates.size() * 2;
    ipc::save_obj(obj, vertices, edges, faces, ev_candidates, v_offset);
    v_offset += ev_candidates.size() * 3;
    ipc::save_obj(obj, vertices, edges, faces, ee_candidates, v_offset);
    v_offset += ee_candidates.size() * 4;
    ipc::save_obj(obj, vertices, faces, faces, fv_candidates, v_offset);
    return true;
}

void Candidates::convert_candidates_to_sets()
{
    for (const auto& vv : vv_candidates) {
        m_vv_set[vv.vertex0_id].insert(vv.vertex1_id);
        m_vv_set[vv.vertex1_id].insert(vv.vertex0_id);
    }
    for (const auto& ee : ee_candidates) {
        m_ee_set[ee.edge0_id].insert(ee.edge1_id);
        m_ee_set[ee.edge1_id].insert(ee.edge0_id);
    }
    for (const auto& ff : ff_candidates) {
        m_ff_set[ff.face0_id].insert(ff.face1_id);
        m_ff_set[ff.face1_id].insert(ff.face0_id);
    }
    for (const auto& ev : ev_candidates) {
        m_ev_set[ev.edge_id].insert(ev.vertex_id);
        m_ve_set[ev.vertex_id].insert(ev.edge_id);
    }
    for (const auto& fv : fv_candidates) {
        m_fv_set[fv.face_id].insert(fv.vertex_id);
        m_vf_set[fv.vertex_id].insert(fv.face_id);
    }
    for (const auto& ef : ef_candidates) {
        m_ef_set[ef.edge_id].insert(ef.face_id);
        m_fe_set[ef.face_id].insert(ef.edge_id);
    }
}

std::set<index_t> Candidates::vv_set(index_t id) const
{
    if (auto iter = m_vv_set.find(id); iter != m_vv_set.end()) {
        return iter->second;
    }
    return {};
}
std::set<index_t> Candidates::ve_set(index_t id) const
{
    if (auto iter = m_ve_set.find(id); iter != m_ve_set.end()) {
        return iter->second;
    }
    return {};
}
std::set<index_t> Candidates::vf_set(index_t id) const
{
    if (auto iter = m_vf_set.find(id); iter != m_vf_set.end()) {
        return iter->second;
    }
    return {};
}

std::set<index_t> Candidates::ev_set(index_t id) const
{
    assert(mesh_.num_vertices());
    std::set<index_t> out;
    if (auto iter = m_ev_set.find(id); iter != m_ev_set.end()) {
        out = iter->second;
    }
    for (index_t lv = 0; lv < 2; ++lv) {
        out.insert(mesh_.edges()(id, lv));
    }
    return out;
}
std::set<index_t> Candidates::ee_set(index_t id) const
{
    assert(mesh_.num_vertices());
    std::set<index_t> out;
    if (auto iter = m_ee_set.find(id); iter != m_ee_set.end()) {
        out = iter->second;
    }
    for (index_t lv = 0; lv < 2; ++lv) {
        for (index_t eid : mesh_.vertices_to_edges()[mesh_.edges()(id, lv)]) {
            out.insert(eid);
        }
    }
    out.erase(id);
    return out;
}
std::set<index_t> Candidates::ef_set(index_t id) const
{
    assert(mesh_.num_vertices());
    std::set<index_t> out;
    if (auto iter = m_ef_set.find(id); iter != m_ef_set.end()) {
        out = iter->second;
    }
    for (index_t lv = 0; lv < 2; ++lv) {
        const auto& faces = mesh_.vertices_to_faces()[mesh_.edges()(id, lv)];
        for (int fid : faces) {
            out.insert(fid);
        }
    }
    out.erase(mesh_.edges_to_faces()(id, 0));
    out.erase(mesh_.edges_to_faces()(id, 1));
    return out;
}

std::set<index_t> Candidates::fv_set(index_t id) const
{
    assert(mesh_.num_vertices());
    std::set<index_t> out;
    if (auto iter = m_fv_set.find(id); iter != m_fv_set.end()) {
        out = iter->second;
    }
    for (index_t lv = 0; lv < 3; ++lv) {
        out.insert(mesh_.faces()(id, lv));
    }
    return out;
}
std::set<index_t> Candidates::fe_set(index_t id) const
{
    assert(mesh_.num_vertices());
    std::set<index_t> out;
    if (auto iter = m_fe_set.find(id); iter != m_fe_set.end()) {
        out = iter->second;
    }
    for (index_t lv = 0; lv < 3; ++lv) {
        for (index_t eid : mesh_.vertices_to_edges()[mesh_.faces()(id, lv)]) {
            out.insert(eid);
        }
    }
    return out;
}
std::set<index_t> Candidates::ff_set(index_t id) const
{
    assert(mesh_.num_vertices());
    std::set<index_t> out;
    if (auto iter = m_ff_set.find(id); iter != m_ff_set.end()) {
        out = iter->second;
    }
    for (index_t lv = 0; lv < 3; ++lv) {
        const index_t vid = mesh_.faces()(id, lv);
        for (index_t fid : mesh_.vertices_to_faces()[vid]) {
            out.insert(fid);
        }
    }
    out.erase(id);
    return out;
}

} // namespace ipc
