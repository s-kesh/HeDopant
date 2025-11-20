#pragma once

#include <cmath>
#include "constants.hpp"
#include "distribution.hpp"

/*
 * LogNormal class for log-normal distribution
 */
class LogNormal: public Distribution {
public:
    LogNormal(double mean) : m_mu(std::log(mean) - 0.5*m_sigma*m_sigma) {}

    double pdf(double x) const override {
        if (x <= 0) return 0;
        double z = (std::log(x) - m_mu) / m_sigma;
        return std::exp(-0.5 * z * z) / (x * m_sigma * sqrt(2 * constants::pi));
    }

    // FWHM = 2.0 exp(μ - σ²) sinh(σ √(2ln(2)))
    inline double FWHM() const {
        return 2.0 * std::exp(m_mu - m_sigma*m_sigma) * std::sinh(m_sigma * std::sqrt(2.0*std::log(2)));
    }

    // ΔN = exp(μ + 0.5σ²)
    double mean() const override{
        return std::exp(m_mu + 0.5*m_sigma*m_sigma);
    }

    inline double sigma() const {return m_sigma;}
    inline double mu() const {return m_mu;}

private:
    // Value for σ such that (FWHM - ΔN) is minimized
    // FWHM = 2.0 exp(μ - σ²) sinh(σ √(2ln(2)))
    // ΔN = exp(μ + 0.5σ²)
    // Minimized by ploting FWHM - ΔN and looking at the minimum value
    const double m_sigma = 0.585431;

    double m_mu;
};
