#pragma once

#include <array>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ipc {

// Interior-only quadrature rules for triangular faces.
//
// Points are given in barycentric coordinates (λ0, λ1, λ2) with λ0+λ1+λ2=1
// and 0 < λi < 1 for all i (strictly interior; no vertex or edge points).
// Weights are normalised to sum to 1 over all points in the rule.
//
// Usage mirrors GaussLobatto in high_order_quadrature.hpp:
//
//   const auto& rule = TriangularQuadrature::get_rule(params.quad_order);
//   for (const auto& qp : rule) {
//       // qp.lambda[0..2] are barycentric coords, qp.weight is the weight
//   }
//
// quad_order semantics for the 3D face-interior integration:
//   0  – no interior points (face-interior contribution is skipped entirely)
//   1  – single centroid point  {1/3, 1/3, 1/3},  weight = 1
//        (equivalent to the previously hard-coded face-centre evaluation)

struct TriangularQuadraturePoint {
    std::array<double, 3> lambda; ///< Barycentric coordinates (sum = 1)
    double weight;                ///< Quadrature weight  (sum over rule = 1)
};

class TriangularQuadrature {
public:
    using Rule = std::vector<TriangularQuadraturePoint>;

    /// @brief Return the cached interior quadrature rule for order \p n.
    static const Rule& get_rule(int n)
    {
        static std::map<int, Rule> cache;
        static std::mutex mtx;

        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(n);
        if (it == cache.end()) {
            it = cache.emplace(n, make_rule(n)).first;
        }
        return it->second;
    }

private:
    static Rule make_rule(int n)
    {
        switch (n) {
        case 0:
            // No face-interior quadrature; face-centre and higher-order
            // interior contributions are skipped entirely.
            return {};

        case 1:
            // 1-point centroid rule (exact for degree-1 polynomials).
            // Equivalent to the previously hard-coded face-centre evaluation.
            return {{{{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}}, 1.0}};

        case 2: {
            // 3-point symmetric rule (exact for degree-2 polynomials).
            // Points are the three cyclic permutations of (2/3, 1/6, 1/6),
            // all strictly interior, with equal weights 1/3.
            constexpr double a = 2.0 / 3.0, b = 1.0 / 6.0, w = 1.0 / 3.0;
            return {
                {{{a, b, b}}, w},
                {{{b, a, b}}, w},
                {{{b, b, a}}, w},
            };
        }

        default:
            throw std::runtime_error(
                "TriangularQuadrature: unsupported order "
                + std::to_string(n));
        }
    }
};

} // namespace ipc
