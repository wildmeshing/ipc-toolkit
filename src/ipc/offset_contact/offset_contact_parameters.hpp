#pragma once
#include <ipc/utils/logger.hpp>

namespace ipc {

struct OffsetContactParameters {
    OffsetContactParameters(
        const double _dhat,
        const double _alpha_t,
        const double _alpha_n,
        const int _r,
        const int _quad_points) :
        dhat(_dhat),
        alpha_t(_alpha_t),
        alpha_n(_alpha_n),
        r(_r),
        quad_points(_quad_points)
    {
        if (abs(alpha_t) > 1) {
            logger().error(
                "Parameter 'alpha_t' must be in [-1, 1]! alpha_t: {}", alpha_t);
        }
        if (abs(alpha_n) > 1) {
            logger().error(
                "Parameter 'alpha_n' must be in [-1, 1]! alpha_n: {}", alpha_n);
        }
    }

    double dhat = 1;
    double alpha_t = 1;
    double alpha_n = 0.1;
    int r = 2;
    int quad_points = 4;


    double adaptive_dhat_ratio() const { return m_adaptive_dhat_ratio; }

    void set_adaptive_dhat_ratio(const double adaptive_dhat_ratio)
    {
        m_adaptive_dhat_ratio = adaptive_dhat_ratio;
    }

private:
    double m_adaptive_dhat_ratio = 0.5;
};

} // namespace ipc