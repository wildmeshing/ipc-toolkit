#pragma once
#include "high_order_collision.hpp"
#include <ipc/utils/unordered_map_and_set.hpp>

namespace ipc {

// A wrapper for unordered_map, with extra helper functions for collisions
template <int N, int dim = 3> class HighOrderCollisionDict {
public:
    using KeyType = std::array<index_t, N>;
    using ValueType = std::shared_ptr<HighOrderCollision>;
    using IterType = typename unordered_map<KeyType, ValueType>::iterator;
    using ConstIterType =
        typename unordered_map<KeyType, ValueType>::const_iterator;

    HighOrderCollisionDict() = default;
    ~HighOrderCollisionDict() = default;

    // collision-specific helper functions

    std::vector<index_t> vertex_ids() const;
    Eigen::VectorXd dof(Eigen::ConstRef<Eigen::MatrixXd> X) const;
    void insert_pair(ValueType&& collision);

    // unordered_map functions:

    IterType find(const KeyType& key) { return map.find(key); }
    ConstIterType find(const KeyType& key) const { return map.find(key); }

    IterType begin() noexcept { return map.begin(); }
    ConstIterType begin() const noexcept { return map.begin(); }
    IterType end() noexcept { return map.end(); }
    ConstIterType end() const noexcept { return map.end(); }

    ValueType& operator[](const KeyType& key) { return map[key]; }
    ValueType& operator[](KeyType&& key) { return map[std::move(key)]; }

private:
    unordered_map<KeyType, ValueType> map;
};
} // namespace ipc