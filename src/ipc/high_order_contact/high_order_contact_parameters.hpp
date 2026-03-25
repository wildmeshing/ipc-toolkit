#pragma once
#include <ipc/utils/logger.hpp>

namespace ipc {

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

private:
    double m_adaptive_dhat_ratio = 0.5;
};

} // namespace ipc
