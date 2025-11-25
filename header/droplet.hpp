#pragma once

#include "dopant.hpp"
#include <cstddef>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

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
     * mean_number: mean size of the droplet
     * distribution_type: type of distribution for droplet
     * dist_step: step size for distribution
     * dopant_name: name of the dopant
     * max_k: maximum no of dopants (just set something high)
     * prefix: prefix for output files
     * datadir: directory for output files
     */
    Droplet(
        const std::vector<std::size_t> mean_numbers,
        const std::string distribution_type,
        const std::size_t dist_step,
        const std::string dopant_name,
        const std::size_t max_k,
        const std::string prefix,
        const std::string datadir
    );
    ~Droplet();


    /*
     * Simulate droplet propagation through the doping cell using Runge-Kutta method
     * no_of_steps: number of steps to evolve, make it large to keep step_size small
     * doping_length: doping cell length
     * pressure: pressure
     * trajectory: whether to save trajectory, keep true if you want to save I_k at each RK4 step
     * input: input file name
     * I_k: distribution of dopants to be evolved over doping cell
     * N_k: distribution of remaining droplet atoms
     */
    void simulate(
        const std::vector<std::size_t> no_of_steps,
        const std::vector<double> doping_length,
        const std::vector<double> pressure,
        const bool trajectory,
        const std::string input,
        Eigen::MatrixXd& I_k,
        Eigen::MatrixXd& N_k
    );

private:
    std::size_t max_mean_number;
    std::vector<std::size_t> m_numbers; // mean numbers of atoms in droplet
    std::string m_type; // Distribution type for droplet
    std::size_t m_dist_step; // Distribution step size
    Dopant m_dopX; // Dopant
    std::string m_prefix; // prefix for output files
    std::string m_datadir; // directory for output files

    std::vector<double> m_vcluster; // velocity of droplet for certain droplet size
    std::vector<double> m_evap; // Binding energy of droplet for certain droplet size

    Eigen::VectorXi m_sizes;
    Eigen::MatrixXd m_sizes_dist;
    Eigen::MatrixXi m_N_k_vec;
    Eigen::MatrixXd m_alpha;

    void _calculate_alpha(const double con1, const double con2, const std::size_t max_k);
    void _calculate_nk_distribution(
        const Eigen::MatrixXd& Ik,
        Eigen::MatrixXd& Nk
    );
};
