#pragma once

#include <vector>
#include <stdexcept>

/*
 * LinearInterpolator class for linear interpolation
 */
class LinearInterpolator {
public:
    LinearInterpolator() = default;

    LinearInterpolator(const std::vector<double>& xs,
                           const std::vector<double>& ys)
    : m_x(xs), m_y(ys)
    {
        if (xs.size() < 2 || xs.size() != ys.size())
            throw std::runtime_error("Invalid interpolation data");
        n = xs.size();
    }

    inline double operator()(double xq) const noexcept
    {
        // bounds
        if (xq <= m_x[0]) return m_y[0];
        if (xq >= m_x[n - 1]) return m_y[n - 1];

        const double* xptr = m_x.data();

        // find interval
        auto it = std::lower_bound(xptr, xptr + n, xq);
        size_t idx = (it - xptr) - 1;

        double x0 = m_x[idx];
        double x1 = m_x[idx + 1];
        double y0 = m_y[idx];
        double y1 = m_y[idx + 1];

        double t = (xq - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

private:
    std::size_t n{};
    std::vector<double> m_x;
    std::vector<double> m_y;
};


/*
 * Interpolators class for interpolation of data
 */
class Interpolators {
public:
    Interpolators(const std::vector<double>& N_list,
                  const std::vector<double>& Vcluster_list,
                  const std::vector<double>& Evap_list);

    Interpolators(const std::string& filename);

    double V_cluster(double N) const;
    double E_vap(double N) const;

private:
    double eval_with_extrapolation(const std::vector<double>& data,
                                   const LinearInterpolator& f,
                                   double x) const;

    std::vector<double> N_data;
    LinearInterpolator V_interp;
    LinearInterpolator E_interp;
};


// File reader
bool read_data_file(const std::string &filename,
                    std::vector<double> &N_list,
                    std::vector<double> &Vcluster_list,
                    std::vector<double> &Evap_list);
