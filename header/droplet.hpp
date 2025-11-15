#pragma once

#include "dopant.hpp"
#include <cstddef>
#include <vector>

// enum struct DistributionType {
//     NONE,
//     LOGNORMAL,
//     EXPONENTIAL,
// };

class Droplet {
public:
    Droplet(std::size_t number, std::string dopant_name, std::size_t max_k);
    ~Droplet();

    std::vector<double> alpha;

    void evolove_rk(
        const std::size_t no_of_steps,
        const double initial_x,
        const double final_x,
        const double pressure,
        std::vector<double>& y
    );

private:
    int m_number;
    // DistributionType m_type;
    Dopant m_dopX;

    std::vector<double> m_vcluster;
    std::vector<double> m_evap;

    void _calculate_alpha(const double con1, const double con2, const std::size_t max_k);
    static void _fun( const double pressure,
        const std::vector<double>& alph,
        const std::vector<double>& x,
        std::vector<double>& y);
};
