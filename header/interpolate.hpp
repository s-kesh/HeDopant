#pragma once

#include <string>
#include <vector>
#include <memory>
#include <gsl/gsl_spline.h>

bool read_data_file(const std::string &filename,
                    std::vector<double> &N_list,
                    std::vector<double> &Vcluster_list,
                    std::vector<double> &Evap_list);

class Interpolators {
public:
    // Constructor: build splines from vectors
    Interpolators(const std::string &filename);
    Interpolators(const std::vector<double>& N_list,
                  const std::vector<double>& Vcluster_list,
                  const std::vector<double>& Evap_list);

    // Interpolation methods (with extrapolation)
    double V_cluster(double N) const;
    double E_vap(double N) const;

private:
    std::vector<double> N_data;

    std::unique_ptr<gsl_spline, void(*)(gsl_spline*)> V_spline{nullptr, gsl_spline_free};
    std::unique_ptr<gsl_interp_accel, void(*)(gsl_interp_accel*)> V_acc{nullptr, gsl_interp_accel_free};

    std::unique_ptr<gsl_spline, void(*)(gsl_spline*)> E_spline{nullptr, gsl_spline_free};
    std::unique_ptr<gsl_interp_accel, void(*)(gsl_interp_accel*)> E_acc{nullptr, gsl_interp_accel_free};

    static double eval_with_extrapolation(const std::vector<double> &data,
                                        gsl_spline* spline,
                                        gsl_interp_accel* acc,
                                        double x);
};
