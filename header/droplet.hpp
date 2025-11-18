#pragma once

#include "dopant.hpp"

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

// enum struct DistributionType {
//     NONE,
//     LOGNORMAL,
//     EXPONENTIAL,
// };

/*
 * Droplet class
 * It holds the properties of He droplet and evolves its over doping cell.
 * It solves for the distribution of number of dopants in the droplet I_k,
 * where k is the number of dopants in the droplet.
 * I_0 means pure droplet, I_1 means one dopant, I_2 means two dopants, etc.
 * Master equation for droplet evolution is given by:
 * dI_0/dx = p(-α_0 * I_0).
 * dI_k/dx = p(-α_k * I_k + α_{k-1} * I_{k-1}); k > 0.
 * α_k = (1/(k_B T)) * √((V_{cluster}² - V_{dopant}²)/ (V_{cluster}²)) * (4πr²N_k^(2/3)).
 */
class Droplet {
public:
    /*
     * Initialize droplet with given parameters
     * number: size of the droplet
     * dopant_name: name of the dopant
     * max_k: maximum no of dopants (just set something high)
     * prefix: prefix for output files
     * datadir: directory for output files
     */
    Droplet(
        const std::size_t number,
        const std::string dopant_name,
        const std::size_t max_k,
        const std::string prefix,
        const std::string datadir
    );
    ~Droplet();


    /*
     * Evolve droplet over doping cell using Runge-Kutta method
     * no_of_steps: number of steps to evolve, make it large to keep step_size small
     * initial_x: initial position, normally 0
     * final_x: final position, normally doping cell length
     * pressure: pressure
     * trajectory: whether to save trajectory, keep true if you want to save I_k at each RK4 step
     * filename: filename to save I_k at each RK4 step
     * I_k: distribution of dopants to be evolved over doping cell
     */
    void evolove_rk(
        const std::size_t no_of_steps,
        const double initial_x,
        const double final_x,
        const double pressure,
        const bool trajectory,
        const std::string filename,
        std::vector<double>& I_k
    );

    /*
     * No of atoms in droplet left after doping of k dopants
     */
    std::vector<std::size_t> N_k_vec;

private:
    int m_number; // number of atoms in droplet
    Dopant m_dopX; // Dopant
    std::string m_prefix; // prefix for output files
    std::string m_datadir; // directory for output files

    std::vector<double> m_vcluster; // velocity of droplet for certain droplet size
    std::vector<double> m_evap; // Binding energy of droplet for certain droplet size

    Eigen::VectorXd m_alpha;
    Eigen::VectorXd m_diag;

    void _calculate_alpha(const double con1, const double con2, const std::size_t max_k);
    void _fun(
        const double pressure,
        Eigen::VectorXd& x
    );
};
