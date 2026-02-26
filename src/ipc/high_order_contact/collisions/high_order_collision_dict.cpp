#include "high_order_collision_dict.hpp"

namespace ipc {

template <int N, int dim>
std::vector<index_t> HighOrderCollisionDict<N, dim>::vertex_ids() const {
  std::set<index_t> vids;
  for (const auto& [key, val] : map) {
    for (const index_t vid : val->vertex_ids()) {
      vids.insert(vid);
    }
  }

  std::vector<index_t> out(vids.size());
  out.assign(vids.begin(), vids.end());
  assert(std::is_sorted(out.begin(), out.end()));
  return out;
}

template <int N, int dim>
Eigen::VectorXd HighOrderCollisionDict<N, dim>::dof(Eigen::ConstRef<Eigen::MatrixXd> X) const
{
  const std::vector<index_t> vids = vertex_ids();
  Eigen::VectorXd out(vids.size() * dim);
  for (index_t i = 0; i < vids.size(); ++i) {
      assert(vids[i] < X.rows());
      assert(X.cols() == dim);
      out.segment<dim>(i * dim) = X.row(vids[i]);
  }
  return out;
}

template <int N, int dim>
void HighOrderCollisionDict<N, dim>::insert_pair(ValueType&& collision)
{
  if (auto iter = map.find(collision->get_typed_hash()); iter != map.end()) {
    iter->second->weight += collision->weight;
    if (iter->second->weight == 0) {
      map.erase(iter);
    }
  }
  else {
    map[collision->get_typed_hash()] = std::move(collision);
  }
}

template class HighOrderCollisionDict<3, 3>;

}