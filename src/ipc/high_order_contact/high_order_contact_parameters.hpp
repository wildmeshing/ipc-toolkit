#pragma once
#include <ipc/utils/logger.hpp>
#include <ipc/barrier/barrier.hpp>
#include <atomic>
#include <limits>
#include <memory>

namespace ipc {

/// A single face quadrature point in barycentric coordinates with its weight.
struct FaceQuadPoint {
    std::array<double, 3> lambda; ///< Barycentric coordinates (sum = 1)
    double weight;
};
using FaceQuadRule = std::vector<FaceQuadPoint>;

struct HighOrderContactParameters {
    enum class IntegrationType {
        BRUTE_FORCE, ///< Integrate all pairs with no obstacle filtering
        NORMAL,      ///< Filter obstacle-obstacle pairs; skip primitives with only obstacle candidates
        NO_OBST      ///< Skip obstacle sources entirely, may miss collisions!
    };

    HighOrderContactParameters(
        const double _dhat,
        const double _dbar_factor = 1.0,
        const int _quad_order = 1,
        const IntegrationType _integration_type = IntegrationType::NORMAL
    ) :
        dhat(_dhat),
        dbar(dhat * _dbar_factor),
        quad_order(_quad_order),
        integration_type(_integration_type)
    {
        if (quad_order > 14) {
            throw std::invalid_argument("Quadrature order >14 is too large.");
        }
        else if (quad_order == 6 || quad_order == 8) {
            logger().error("Quadrature orders 6 and 8 has negative vertex weights.");
        }
        else if (quad_order >= 10 && quad_order <= 12) {
            logger().warn("Quadrature orders 10-12 are not implemented, and instead use order 13.");
        }
        else if (quad_order == 1) {
            logger().warn("Quadrature order 1 is equivalent to vertex quadrature.");
        }
    }

    const double dhat;
    const double dbar;

    /// When true, use OGC feasibility-region collision building instead of the
    /// standard quadrature-based alternating-sign formulation.
    bool ogc_collisions = false;

    /// Barrier function used in 3D collision evaluation.
    std::shared_ptr<Barrier> barrier = std::make_shared<NormalizedClampedLogBarrier>();
    const int quad_order;
    const IntegrationType integration_type;

    double get_dhat(bool safety_mode=false) const { return safety_mode ? dbar : dhat; }

    double adaptive_dhat_ratio() const { return m_adaptive_dhat_ratio; }

    void set_adaptive_dhat_ratio(const double adaptive_dhat_ratio)
    {
        m_adaptive_dhat_ratio = adaptive_dhat_ratio;
    }

    const FaceQuadRule& get_quad_rule() const { return face_quad_rule; }

    /// Face quadrature rule. Empty (default) skips face quadrature entirely,
    /// matching the behaviour of quad_order == 0 in the old interface.
    FaceQuadRule face_quad_rule;

    /// Record a distance passed to the barrier; tracks the running minimum
    /// across all threads. Copies of this struct share the same tracker
    /// (shared_ptr), so pass-by-value sites still update the original.
    void record_dist(double d) const {
        auto& a = *m_min_dist_seen;
        double cur = a.load(std::memory_order_relaxed);
        while (d < cur && !a.compare_exchange_weak(cur, d, std::memory_order_relaxed)) {}
    }

    double min_dist_seen() const {
        return m_min_dist_seen->load(std::memory_order_relaxed);
    }

    void reset_min_dist() const {
        m_min_dist_seen->store(std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
    }

private:
    double m_adaptive_dhat_ratio = 0.5;
    std::shared_ptr<std::atomic<double>> m_min_dist_seen =
        std::make_shared<std::atomic<double>>(std::numeric_limits<double>::infinity());
};

} // namespace ipc
