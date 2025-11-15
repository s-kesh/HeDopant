#include "interpolate.hpp"
#include <stdexcept>
#include <gsl/gsl_spline.h>
#include <fstream>
#include <iostream>

Interpolators::Interpolators(const std::string& filename)
{
    std::vector<double> Vcluster_list, Evap_list;

    if (!read_data_file(filename, N_data, Vcluster_list, Evap_list)) {
       std::runtime_error("Could not file the file droplet.txt\n");
    }


    size_t n = N_data.size();
    if (n < 2 || Vcluster_list.size() != n || Evap_list.size() != n) {
        throw std::runtime_error("Invalid data vectors");
    }

    // Allocate accelerators
    V_acc.reset(gsl_interp_accel_alloc());
    E_acc.reset(gsl_interp_accel_alloc());

    // Allocate splines
    V_spline.reset(gsl_spline_alloc(gsl_interp_cspline, n));
    E_spline.reset(gsl_spline_alloc(gsl_interp_cspline, n));

    gsl_spline_init(V_spline.get(), N_data.data(), Vcluster_list.data(), n);
    gsl_spline_init(E_spline.get(), N_data.data(), Evap_list.data(), n);
}

Interpolators::Interpolators(const std::vector<double>& N_list,
                             const std::vector<double>& Vcluster_list,
                             const std::vector<double>& Evap_list)
    : N_data(N_list)
{
    size_t n = N_list.size();
    if (n < 2 || Vcluster_list.size() != n || Evap_list.size() != n) {
        throw std::runtime_error("Invalid data vectors");
    }

    // Allocate accelerators
    V_acc.reset(gsl_interp_accel_alloc());
    E_acc.reset(gsl_interp_accel_alloc());

    // Allocate splines
    V_spline.reset(gsl_spline_alloc(gsl_interp_cspline, n));
    E_spline.reset(gsl_spline_alloc(gsl_interp_cspline, n));

    gsl_spline_init(V_spline.get(), N_list.data(), Vcluster_list.data(), n);
    gsl_spline_init(E_spline.get(), N_list.data(), Evap_list.data(), n);
}

double Interpolators::V_cluster(double N) const {
    return eval_with_extrapolation(N_data, V_spline.get(), V_acc.get(), N);
}

double Interpolators::E_vap(double N) const {
    return eval_with_extrapolation(N_data, E_spline.get(), E_acc.get(), N);
}

double Interpolators::eval_with_extrapolation(const std::vector<double> &data,
                                            gsl_spline* spline,
                                            gsl_interp_accel* acc,
                                            double x)
{
    double x_min = data.front();
    double x_max = data.back();

    if (x < x_min) {
        // slope from first two points
        double y0 = gsl_spline_eval(spline, x_min, acc);
        double y1 = gsl_spline_eval(spline, x_min + 1e-6, acc);
        double slope = (y1 - y0) / 1e-6;
        return y0 + slope * (x - x_min);
    }
    if (x > x_max) {
        // slope from last two points
        double y0 = gsl_spline_eval(spline, x_max, acc);
        double y1 = gsl_spline_eval(spline, x_max - 1e-6, acc);
        double slope = (y0 - y1) / 1e-6;
        return y0 + slope * (x - x_max);
    }
    return gsl_spline_eval(spline, x, acc);
}

bool read_data_file(const std::string &filename,
                    std::vector<double> &N_list,
                    std::vector<double> &Vcluster_list,
                    std::vector<double> &Evap_list)
{
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return false;
    }

    std::string line;
    std::getline(in, line); // skip header

    double N, Vc, Ev;
    while (in >> N >> Vc >> Ev) {
        N_list.push_back(N);
        Vcluster_list.push_back(Vc);
        Evap_list.push_back(Ev);
    }

    return true;
}
