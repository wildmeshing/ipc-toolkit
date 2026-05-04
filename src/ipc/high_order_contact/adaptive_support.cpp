#include "adaptive_support.hpp"

namespace ipc {

AdaptiveSupport::AdaptiveSupport(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> rest_positions,
    const HighOrderContactParameters& params
)
    : m_mesh(&mesh)
{
    m_values.setRandom(mesh.num_vertices()); // just for testing
    m_values = (m_values.array() + 3.0) / 4.0 * params.dhat;
}

double AdaptiveSupport::vertex(index_t vertex_id) const
{
    return m_values(vertex_id);
}

double AdaptiveSupport::edge(index_t edge_id, double t) const
{
    const int v0 = m_mesh->edges()(edge_id, 0);
    const int v1 = m_mesh->edges()(edge_id, 1);
    return (1.0 - t) * m_values(v0) + t * m_values(v1);
}

double AdaptiveSupport::face(index_t face_id, double u, double v) const
{
    const int f0 = m_mesh->faces()(face_id, 0);
    const int f1 = m_mesh->faces()(face_id, 1);
    const int f2 = m_mesh->faces()(face_id, 2);
    const double w = 1.0 - u - v;
    return w * m_values(f0) + u * m_values(f1) + v * m_values(f2);
}

} // namespace ipc
