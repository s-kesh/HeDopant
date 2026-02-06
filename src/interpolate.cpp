#include "interpolate.hpp"
#include "constants.hpp"
#include <fstream>
#include <iostream>

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


Interpolators::Interpolators(const std::vector<double>& N_list,
                             const std::vector<double>& Vcluster_list,
                             const std::vector<double>& Evap_list)
    : N_data(N_list), V_interp(N_list, Vcluster_list), E_interp(N_list, Evap_list)
{
    size_t n = N_list.size();
    if (n < 2 || Vcluster_list.size() != n || Evap_list.size() != n)
        throw std::runtime_error("Invalid data vectors");
}

Interpolators::Interpolators(const std::string& filename)
{
    std::vector<double> N_list, Vcluster_list, Evap_list;

    if (!read_data_file(filename, N_list, Vcluster_list, Evap_list)) {
        throw std::runtime_error("Could not find or read the file");
    }

    // call the main constructor
    *this = Interpolators(N_list, Vcluster_list, Evap_list);
}


double Interpolators::eval_with_extrapolation(const std::vector<double> &data,
                                              const LinearInterpolator& f,
                                              double x) const
{
    double x_min = data.front();
    double x_max = data.back();

    if (x <= x_min) {
        // forward slope
        double y0 = f(x_min);
        double y1 = f(x_min + 1e-6);
        double slope = (y1 - y0) / 1e-6;
        return y0 + slope * (x - x_min);
    }

    if (x >= x_max) {
        // backward slope
        double y0 = f(x_max);
        double y1 = f(x_max - 1e-6);
        double slope = (y0 - y1) / 1e-6;
        return y0 + slope * (x - x_max);
    }

    return f(x)*constants::kB_e;
}

double Interpolators::V_cluster(double N) const {
    return eval_with_extrapolation(N_data, V_interp, N);
}

double Interpolators::E_vap(double N) const {
    return eval_with_extrapolation(N_data, E_interp, N);
}
