#pragma once
#include <ipc/utils/logger.hpp>

namespace ipc {

struct OffsetContactParameters {
    double dhat = 1;
    int r = 1;

    OffsetContactParameters(
        const double _dhat) :
        dhat(_dhat)
    {}


    double adaptive_dhat_ratio() const { return m_adaptive_dhat_ratio; }

    void set_adaptive_dhat_ratio(const double adaptive_dhat_ratio)
    {
        m_adaptive_dhat_ratio = adaptive_dhat_ratio;
    }

private:
    double m_adaptive_dhat_ratio = 0.5;
};

} // namespace ipc