// Barrier functions that grow to infinity as x -> 0+. Includes gradient and
// hessian functions, too. These barrier functions can be used to impose
// inequality constraints on a function.
#include "barrier.hpp"
#include <ipc/math/math.hpp>

#include <cmath>
#include <limits>

namespace ipc {

double barrier(const double d, const double dhat)
{
    if (d <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (d >= dhat) {
        return 0;
    }
    // b(d) = -(d-d̂)²ln(d / d̂)
    const double d_minus_dhat = (d - dhat);
    return -d_minus_dhat * d_minus_dhat * log(d / dhat);
}

double barrier_first_derivative(const double d, const double dhat)
{
    if (d <= 0.0 || d >= dhat) {
        return 0.0;
    }
    // b(d) = -(d - d̂)²ln(d / d̂)
    // b'(d) = -2(d - d̂)ln(d / d̂) - (d-d̂)²(1 / d)
    //       = (d - d̂) * (-2ln(d/d̂) - (d - d̂) / d)
    //       = (d̂ - d) * (2ln(d/d̂) - d̂/d + 1)
    return (dhat - d) * (2 * log(d / dhat) - dhat / d + 1);
}

double barrier_second_derivative(const double d, const double dhat)
{
    if (d <= 0.0 || d >= dhat) {
        return 0.0;
    }
    const double dhat_d = dhat / d;
    return (dhat_d + 2) * dhat_d - 2 * log(d / dhat) - 3;
}

// ============================================================================

double ClampedLogSqBarrier::operator()(const double d, const double dhat) const
{
    if (d <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (d >= dhat) {
        return 0;
    }
    // b(d) = (d-d̂)²ln²(d / d̂)
    const double d_minus_dhat = (d - dhat);
    const double log_d_dhat = log(d / dhat);
    return d_minus_dhat * d_minus_dhat * log_d_dhat * log_d_dhat;
}

double
ClampedLogSqBarrier::first_derivative(const double d, const double dhat) const
{
    if (d <= 0.0 || d >= dhat) {
        return 0.0;
    }
    // b(d) = (d - d̂)²ln²(d / d̂)
    // b'(d) = 2 (d - d̂) ln²(d / d̂) + 2 (d - d̂)² ln(d / d̂) / d
    //       = 2 (d - d̂) ln(d / d̂) [ln(d / d̂) + (d - d̂) / d]
    const double d_minus_dhat = (d - dhat);
    const double log_d_dhat = log(d / dhat);
    return 2 * d_minus_dhat * log_d_dhat * (log_d_dhat + d_minus_dhat / d);
}

double
ClampedLogSqBarrier::second_derivative(const double d, const double dhat) const
{
    if (d <= 0.0 || d >= dhat) {
        return 0.0;
    }
    const double t0 = dhat - d;
    const double t1 = log(d / dhat);
    const double t2 = (t0 * t0) / (d * d);
    return 2 * ((t1 * t1) - (t1 - 1) * t2 - 4 * t1 * t0 / d);
}

// ============================================================================

double CubicBarrier::operator()(const double d, const double dhat) const
{
    if (d < dhat) {
        // b(d) = (d - d̂)³
        const double d_minus_dhat = (d - dhat);
        return -2.0 / 3.0 / dhat * d_minus_dhat * d_minus_dhat * d_minus_dhat;
    } else {
        return 0;
    }
}

double CubicBarrier::first_derivative(const double d, const double dhat) const
{
    if (d < dhat) {
        const double d_minus_dhat = (d - dhat);
        return -2 / dhat * d_minus_dhat * d_minus_dhat;
    } else {
        return 0;
    }
}

double CubicBarrier::second_derivative(const double d, const double dhat) const
{
    if (d < dhat) {
        return 4 * (1 - d / dhat);
    } else {
        return 0;
    }
}

// ============================================================================

double TwoStageBarrier::operator()(const double d, const double dhat) const
{
    if (d >= dhat) {
        return 0.0;
    } else if (d >= 0.5 * dhat) {
        return 0.5 * (dhat - d) * (dhat - d);
    } else {
        return -0.25 * dhat * dhat * (std::log(2 * d / dhat) - 0.5);
    }
}

double
TwoStageBarrier::first_derivative(const double d, const double dhat) const
{
    if (d >= dhat) {
        return 0.0;
    } else if (d >= 0.5 * dhat) {
        return d - dhat;
    } else {
        return -0.25 * dhat * dhat / d;
    }
}

double
TwoStageBarrier::second_derivative(const double d, const double dhat) const
{
    if (d >= dhat) {
        return 0.0;
    } else if (d >= 0.5 * dhat) {
        return 1.0;
    } else {
        return (0.25 * dhat * dhat) / (d * d);
    }
}

// ============================================================================

void InversePowerBarrier::h_and_derivs(
    const double d, const double dhat,
    double& h, double& dh, double& ddh)
{
    const double t = 2.0 * d / dhat;
    double B, dB, ddB;
    if (t < 1.0) {
        B   =  2.0/3.0 - t * t + 0.5 * t * t * t;
        dB  = -2.0 * t + 1.5 * t * t;
        ddB = -2.0 + 3.0 * t;
    } else if (t < 2.0) {
        const double s = 2.0 - t;
        B   =  s * s * s / 6.0;
        dB  = -s * s / 2.0;
        ddB =  s;
    } else {
        h = dh = ddh = 0.0;
        return;
    }
    // h = 2*B(t), dh/dd = 2*B'(t)*(dt/dd) = 2*B'*(2/dhat) = 4/dhat * B'
    h   = 2.0 * B;
    dh  = 4.0 / dhat * dB;
    ddh = 8.0 / (dhat * dhat) * ddB;
}

double InversePowerBarrier::operator()(const double d, const double dhat) const
{
    if (d <= 0.0)
        return std::numeric_limits<double>::infinity();
    double h, dh, ddh;
    h_and_derivs(d, dhat, h, dh, ddh);
    if (h == 0.0)
        return 0.0;
    return h / std::pow(d, m_power);
}

double
InversePowerBarrier::first_derivative(const double d, const double dhat) const
{
    if (d <= 0.0 || d >= dhat)
        return 0.0;
    double h, dh, ddh;
    h_and_derivs(d, dhat, h, dh, ddh);
    // b'(d) = (dh·d − p·h) / d^(p+1)
    return (dh * d - m_power * h) / std::pow(d, m_power + 1.0);
}

double
InversePowerBarrier::second_derivative(const double d, const double dhat) const
{
    if (d <= 0.0 || d >= dhat)
        return 0.0;
    double h, dh, ddh;
    h_and_derivs(d, dhat, h, dh, ddh);
    // b''(d) = (ddh·d² − 2p·dh·d + p(p+1)·h) / d^(p+2)
    const double d2 = d * d;
    return (ddh * d2 - 2.0 * m_power * dh * d + m_power * (m_power + 1.0) * h)
        / std::pow(d, m_power + 2.0);
}

// ============================================================================

double NearFarBarrier::near(const double d, const double dhat) const
{
    const double dhat_end = m_alpha * dhat;
    const double dhat_start = dhat_end / 2.0;
    return (*m_base_barrier)(d, dhat)
        * (1.0 - Math<double>::smooth_heaviside(d, dhat_start, dhat_end));
}

double NearFarBarrier::far(const double d, const double dhat) const
{
    const double dhat_end = m_alpha * dhat;
    const double dhat_start = dhat_end / 2.0;
    return (*m_base_barrier)(d, dhat)
        * Math<double>::smooth_heaviside(d, dhat_start, dhat_end);
}

double NearFarBarrier::first_derivative_near(const double d, const double dhat) const
{
    const double dhat_end = m_alpha * dhat;
    const double dhat_start = dhat_end / 2.0;
    const double b = (*m_base_barrier)(d, dhat);
    const double bp = m_base_barrier->first_derivative(d, dhat);
    const double w = Math<double>::smooth_heaviside(d, dhat_start, dhat_end);
    const double wp = Math<double>::smooth_heaviside_grad(d, dhat_start, dhat_end);
    return bp * (1.0 - w) - b * wp;
}

double NearFarBarrier::first_derivative_far(const double d, const double dhat) const
{
    const double dhat_end = m_alpha * dhat;
    const double dhat_start = dhat_end / 2.0;
    const double b = (*m_base_barrier)(d, dhat);
    const double bp = m_base_barrier->first_derivative(d, dhat);
    const double w = Math<double>::smooth_heaviside(d, dhat_start, dhat_end);
    const double wp = Math<double>::smooth_heaviside_grad(d, dhat_start, dhat_end);
    return bp * w + b * wp;
}

double NearFarBarrier::second_derivative_near(const double d, const double dhat) const
{
    const double dhat_end = m_alpha * dhat;
    const double dhat_start = dhat_end / 2.0;
    const double b = (*m_base_barrier)(d, dhat);
    const double bp = m_base_barrier->first_derivative(d, dhat);
    const double bpp = m_base_barrier->second_derivative(d, dhat);
    const double w = Math<double>::smooth_heaviside(d, dhat_start, dhat_end);
    const double wp = Math<double>::smooth_heaviside_grad(d, dhat_start, dhat_end);
    const double wpp = Math<double>::smooth_heaviside_hess(d, dhat_start, dhat_end);
    return bpp * (1.0 - w) - 2.0 * bp * wp - b * wpp;
}

double NearFarBarrier::second_derivative_far(const double d, const double dhat) const
{
    const double dhat_end = m_alpha * dhat;
    const double dhat_start = dhat_end / 2.0;
    const double b = (*m_base_barrier)(d, dhat);
    const double bp = m_base_barrier->first_derivative(d, dhat);
    const double bpp = m_base_barrier->second_derivative(d, dhat);
    const double w = Math<double>::smooth_heaviside(d, dhat_start, dhat_end);
    const double wp = Math<double>::smooth_heaviside_grad(d, dhat_start, dhat_end);
    const double wpp = Math<double>::smooth_heaviside_hess(d, dhat_start, dhat_end);
    return bpp * w + 2.0 * bp * wp + b * wpp;
}

} // namespace ipc
