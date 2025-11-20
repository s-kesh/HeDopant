#include "droplet.hpp"
#include "Eigen/Core"
#include "constants.hpp"
#include "interpolate.hpp"
#include "dopant.hpp"
#include "arrow_io.hpp"
#include "log_normal.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

void print_progress_bar(float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth*progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">>";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress*100.0) << " %\r";
    std::cout.flush();
}

Droplet::Droplet(
    const std::vector<std::size_t> numbers,
    DistributionType type,
    const std::string dopant_name,
    const std::size_t max_k,
    const std::string prefix,
    const std::string datadir
)
    : m_numbers(numbers), m_type(type), m_dopX(dopant_name, datadir)
    , m_prefix(prefix), m_datadir(datadir)
{
    double con1 = (1.0 / constants::kB / m_dopX.temprature()) * 4.0 * constants::pi * constants::he_density * constants::he_density;
    double con2 = m_dopX.e_Int() + (1.5 * constants::kB * m_dopX.temprature()) + (m_dopX.e_He_X() + m_dopX.e_X_X())*constants::e;

    max_mean_number = *std::max_element(numbers.begin(), numbers.end());

    if (m_type == DistributionType::NONE) {
        m_vcluster.resize(max_mean_number+5);
        m_evap.resize(max_mean_number+5);
        m_sizes.resize(m_numbers.size());
        m_sizes_dist.resize(1, m_numbers.size());
        for (int n = 0; n < m_sizes.size(); n++) {
            m_sizes[n] = m_numbers[n];
            m_sizes_dist(0, n) = 1.0;
        }

        // Resize alpha, diagonal
        m_N_k_vec.resize(m_numbers.size(), max_k + 1);
        m_alpha.resize(m_numbers.size(), max_k + 1);
        m_diag.resize(m_numbers.size(), max_k + 1);
    }
    else {
        m_vcluster.resize(12*max_mean_number);
        m_evap.resize(12*max_mean_number);

        m_sizes.resize(max_mean_number);
        for (int i = 0; i < m_sizes.size(); i++) {
            m_sizes[i] = 10*i;
        }

        m_sizes_dist.resize(m_sizes.size(), numbers.size());
        // create a lognormal distribution
        for (std::size_t n = 0; n < numbers.size(); n++) {
            LogNormal log_normal(numbers[n]);
            for (std::size_t i = 0; i < max_mean_number; i++) {
                m_sizes_dist(i, n) = log_normal.pdf(10*i);
            }
        }

        // Save size distributions to a txt file
        for (std::size_t n = 0; n < numbers.size(); n++) {
            std::ofstream file(std::format("{}_{}_size_distribution.txt", m_prefix, numbers[n]));
            file << "Index\tdroplet_size\tLognormal\n";
            for (std::size_t i = 0; i < max_mean_number; i++) {
                file << std::format("{}\t{}\t{}\n", i, m_sizes[i], m_sizes_dist(i, n));
            }
        }

        // Resize alpha, diagonal
        // They are in Column Major order
        // ROWS -> sizes.size()
        // COLUMNS -> max_k + 1
        m_N_k_vec.resize(m_sizes.size(), max_k + 1);
        m_alpha.resize(m_sizes.size(), max_k + 1);
        m_diag.resize(m_sizes.size(), max_k + 1);
    }

    auto interp = Interpolators(std::format("{}/droplet.txt", m_datadir));
    for (std::size_t i = 0; i < m_vcluster.size(); i++) {
        m_vcluster[i] = interp.V_cluster(i+1);
        m_evap[i] = interp.E_vap(i+1)*constants::kB;
    }

    // Save the calculated vcluster and evap values to a file
    std::ofstream file(std::format("{}_vcluster_ebe.txt", m_prefix));
    file << "droplet_size\tvelocity\tbinding_energy\n";
    for (std::size_t i = 0; i < max_mean_number+5; i++) {
        file << std::format("{}\t{}\t{}\n", i, m_vcluster[i], m_evap[i]);
    }
    file.close();

    _calculate_alpha(con1, con2, max_k);
}

Droplet::~Droplet()
{
}

void Droplet::_calculate_alpha(const double con1, const double con2, const std::size_t max_k) {
    // TO calculate alpha
    // α (k) = (1/kT) * sqrt(v_cluster^2 + vx^2 / v_cluster^2) * 4*pi*r_0^2 N_k^(2/3)

    double v_x2 = m_dopX.velocity()*m_dopX.velocity();
    double mass = m_dopX.mass()*constants::amu;

    for (int i = 0; i < m_sizes.size(); i++) {
        std::size_t number = m_sizes[i];
        if (number < 10) continue;

        double v_cluster2 = m_vcluster[number]*m_vcluster[number];
        double E_total = 0.0;
        std::size_t N_evap = 0;

        std::size_t N_k = number;
        std::size_t N_old = number;

        m_N_k_vec(i, 0) = N_k;
        m_alpha(i, 0) = con1 * std::sqrt((v_cluster2 + v_x2)/ v_cluster2) * std::pow(N_k, 2.0/3.0);

        for (std::size_t k = 1; k < max_k + 1; k++) {
            E_total = con2 + (0.5 * mass * v_cluster2);
            N_evap = (std::size_t)(E_total / m_evap[N_old]);
            if (N_old > N_evap)
                N_k = (std::size_t)(N_old - N_evap);
            else
                N_k = 0;
            m_N_k_vec(i, k) = N_k;
            v_cluster2 = m_vcluster[N_k] * m_vcluster[N_k];
            m_alpha(i, k) = con1 * std::sqrt((v_cluster2 + v_x2) / v_cluster2) * std::pow(N_k, 2.0/3.0);
            N_old = N_k;
        }
    }

    // Output in a text file
    // Header: k, N_k, alpha
    std::ofstream file(std::format("{}_evap.txt", m_prefix));
    for (std::size_t i = 0; i < (std::size_t)m_sizes.size(); i++) {
        file << "=========================\n";
        file << std::format("For Initial size: {}\n", m_sizes[i]);
        file << "k\tN_k\talpha\n";
        for (std::size_t k = 0; k < max_k + 1; k++) {
            file << std::format("{}\t{}\t{}\n", k, m_N_k_vec(i, k), m_alpha(i, k));
        }
        file << "=========================\n";
    }
    file.close();
}

void Droplet::_fun(
    const double pressure,
    Eigen::MatrixXd& x
)
{
    // shape of x: N x k
    // y[0] = -pressure * alpha[0] * x[0];
    // y[i] = diag[i-1] - diag[i]; for i > 0

    m_diag = pressure * m_alpha.array() * x.array();

    x.block(0, 1, x.rows(), x.cols() - 1).noalias() =
        m_diag.block(0, 0, m_diag.rows(), m_diag.cols() - 1) -
        m_diag.block(0, 1, m_diag.rows(), m_diag.cols() - 1);

    x.col(0).noalias() = -m_diag.col(0);
}

void Droplet::evolove_rk(
    const std::size_t no_of_steps,
    const double final_x,
    const double pressure,
    const bool trajectory,
    const std::string filename,
    Eigen::MatrixXd& y_out
) {

    // setup output file

    ArrowIO arrowIO(std::format("{}_{}.arrow", m_prefix, filename));

    const double stepsize = (final_x - 0) / no_of_steps;
    Eigen::MatrixXd y(m_sizes.size(), y_out.cols());

    // Set Initial conditions
    y.fill(0);
    y.col(0).setOnes();

    Eigen::MatrixXd temp(y.rows(), y.cols()), kfinal(y.rows(), y.cols());

    if (trajectory) arrowIO.write_step(0, y.size(), y.data());

    // Progress bar
    float progress = 0.0;
    for (std::size_t step = 1; step <= no_of_steps; step++) {

        // k1
        temp.noalias() = y;
        _fun(pressure, temp); // now temp = f(y); k1
        kfinal.noalias() = temp; // k_final = k1

        // k2
        temp.noalias() = y + 0.5*stepsize * temp; // temp = y + (h/2)k1
        _fun(pressure, temp); // now temp = f(y + (h/2)*k1); k2
        kfinal.noalias() += 2.0*temp; // k_final = k1 + 2k2

        // k3
        temp.noalias() = y + 0.5*stepsize * temp; // temp = y + (h/2)k2
        _fun(pressure, temp); // now temp = f(y + (h/2)*k2); k3
        kfinal.noalias() += 2.0*temp; // k_final = k1 + 2k2 + 2k3

        // k4
        temp.noalias() = y + stepsize * temp; // temp = y + h*k3
        _fun(pressure, temp); // now temp = f(y + h*k3); k4
        kfinal.noalias() += temp; // k_final = k1 + 2k2 + 2k3 + k4

        // Make RK4 step
        // y += (h/6)*(k1 + 2k2 + 2k3 + k4)
        y.noalias() += (stepsize/6.0) * kfinal;

        // stream new y
        if (trajectory) arrowIO.write_step(step, y.size(), y.data());

        if ((step*100) % no_of_steps ) print_progress_bar(progress);

        progress = (step*1.0 + 1)/no_of_steps;
    }
    print_progress_bar(1.0);
    std::cout << std::endl;

    // Save final y
    // Output in a text file
    // Header: k, N_k, alpha
    std::ofstream file(std::format("{}_final_y.txt", m_prefix));
    for (std::size_t i = 0; i < (std::size_t)y.rows(); i++) {
        file << "=========================\n";
        file << std::format("For Initial size: {}\n", m_sizes[i]);
        file << "k\ty\n";
        for (std::size_t k = 0; k < (std::size_t)y.cols(); k++) {
            file << std::format("{}\t{}\n", k, y(i, k));
        }
        file << "=========================\n";
    }
    file.close();

    // Now for each mean number, we have a distribution
    if (m_type == DistributionType::NONE) {
        y_out.noalias() = y;
    }
    else {
        // Now multiple y with size distribution
        // shape[k] = shape[Nxk].T shape[N]
        for (int n = 0; n < y_out.rows(); n++) {
            y_out.row(n) = y.transpose() * m_sizes_dist.col(n);
        }
    }

    arrowIO.close();
}
