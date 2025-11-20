#pragma once

#include "distribution.hpp"
#include <cstddef>
#include <cmath>

/*
 * Exponential class for exponential distribution
 */
class Exponential: public Distribution {
public:
    Exponential(std::size_t mean) : m_lambda(1.0/mean) {}
    ~Exponential() = default;

    double pdf(double x) const override {
        if (x <= 0) return 0;
        return m_lambda * std::exp(-m_lambda * x);
    }

    double mean() const override{
        return 1.0/m_lambda;
    }

private:
    double m_lambda;
};
