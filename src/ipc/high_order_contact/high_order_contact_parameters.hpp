#pragma once
#include <ipc/utils/logger.hpp>

namespace ipc {

struct HighOrderContactParameters {
    HighOrderContactParameters(
        const double _dhat,
        const double _alpha,
        const int _r,
        const int _quad_points,
        const bool _skip_obstacle = true
    ) :
        dhat(_dhat),
        alpha(_alpha),
        r(_r),
        quad_points(_quad_points),
        skip_obstacle(_skip_obstacle)
    {
        if (abs(alpha) > 1) {
            logger().error(
                "Parameter 'alpha' must be in [-1, 1]! alpha: {}", alpha);
        }
    }

    double dhat;
    double alpha;
    int r;
    int quad_points;
    bool skip_obstacle;


    double adaptive_dhat_ratio() const { return m_adaptive_dhat_ratio; }

    void set_adaptive_dhat_ratio(const double adaptive_dhat_ratio)
    {
        m_adaptive_dhat_ratio = adaptive_dhat_ratio;
    }

private:
    double m_adaptive_dhat_ratio = 0.5;
};

} // namespace ipc