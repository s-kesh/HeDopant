#pragma once

#include "Eigen/Core"
#include "dopant.hpp"

#include <cstddef>
#include <vector>
#include <Eigen/Dense>

// enum struct DistributionType {
//     NONE,
//     LOGNORMAL,
//     EXPONENTIAL,
// };

class Droplet {
public:
    Droplet(
        const std::size_t number,
        const std::string dopant_name,
        const std::size_t max_k,
        const std::string prefix,
        const std::string datadir
    );
    ~Droplet();


    void evolove_rk(
        const std::size_t no_of_steps,
        const double initial_x,
        const double final_x,
        const double pressure,
        const bool trajectory,
        const std::string filename,
        std::vector<double>& y
    );

private:
    int m_number;
    Dopant m_dopX;
    std::string m_prefix;
    std::string m_datadir;

    std::vector<double> m_vcluster;
    std::vector<double> m_evap;

    Eigen::VectorXd m_alpha;
    Eigen::VectorXd m_diag;

    void _calculate_alpha(const double con1, const double con2, const std::size_t max_k);
    void _fun(
        const double pressure,
        Eigen::VectorXd& x
    );
};
