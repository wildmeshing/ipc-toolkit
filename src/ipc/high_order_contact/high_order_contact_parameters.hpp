#pragma once
#include <ipc/utils/logger.hpp>

namespace ipc {

struct HighOrderContactParameters {
    HighOrderContactParameters(
        const double _dhat,
        const double _dbar_factor = 1.0,
        const int _quad_order = 1,
        const int _exponent = 2,
        const bool _skip_obstacle = true
    ) :
        dhat(_dhat),
        dbar(dhat * _dbar_factor),
        quad_order(_quad_order),
        r(_exponent),
        skip_obstacle(_skip_obstacle)
    {
    }

    constexpr static double alpha = 0.; // For compatibility
    const double dhat;
    const double dbar;
    const int quad_order;
    const int r = 2;
    const bool skip_obstacle;

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