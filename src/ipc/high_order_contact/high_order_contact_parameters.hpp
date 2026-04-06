#pragma once
#include <ipc/utils/logger.hpp>

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
        const int _exponent = 2,
        const IntegrationType _integration_type = IntegrationType::NORMAL
    ) :
        dhat(_dhat),
        dbar(dhat * _dbar_factor),
        quad_order(_quad_order),
        r(_exponent),
        integration_type(_integration_type)
    {
    }

    constexpr static double alpha = 0.; // For compatibility
    const double dhat;
    const double dbar;
    const int quad_order;
    const int r = 2;
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

private:
    double m_adaptive_dhat_ratio = 0.5;
};

} // namespace ipc
