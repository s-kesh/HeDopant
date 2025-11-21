#include <print>
#include <fstream>

#include "droplet.hpp"
#include "constants.hpp"
#include "interpolate.hpp"
#include "dopant.hpp"

#include "distribution.hpp"
#include "log_normal.hpp"
#include "exponential.hpp"

#include "rk4_backend.hpp"
#include "rk4_cpu.hpp"

#ifdef RK4_HAS_GPU
#include <cuda_runtime.h>
#include "rk4_gpu.hpp"
#endif

std::shared_ptr<RK4Backend> create_backend()
{
#ifdef RK4_HAS_GPU
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        std::println("GPU backend selected");
        return std::make_unique<RK4Backend_GPU>();
    }
#endif
    std::println("CPU backend selected");
    return std::make_unique<RK4Backend_CPU>();
}

std::unique_ptr<Distribution> create_distribution(
    const std::string& type, double mean
) {
    DistributionType dist_type = convert_type(type);
    switch (dist_type) {
        case DistributionType::LOGNORMAL:
            return std::make_unique<LogNormal>(mean);
        case DistributionType::EXPONENTIAL:
            return std::make_unique<Exponential>(mean);
        default:
            throw std::invalid_argument("Invalid distribution type specified: " + type);
    }
}

Droplet::Droplet(
    const std::vector<std::size_t> numbers,
    const std::string type,
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

    if (m_type == "NONE") {
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
    }
    else {
        m_vcluster.resize(12*max_mean_number);
        m_evap.resize(12*max_mean_number);

        // Creating an array from 0 to 10*max_mean_number with step of DIST_STEP
        int size = (10*max_mean_number - 0) / DIST_STEP;
        m_sizes.resize(size);
        for (int i = 0; i < m_sizes.size(); i++) {
            m_sizes[i] = 0 + DIST_STEP*i;
        }

        m_sizes_dist.resize(m_sizes.size(), numbers.size());
        for (std::size_t n = 0; n < numbers.size(); n++) {
            auto drop_dist = create_distribution(type, numbers[n]);
            for (int i = 0; i < m_sizes.size(); i++) {
                m_sizes_dist(i, n) = drop_dist->pdf(m_sizes[i]);
            }
        }

        // Save size distributions to a txt file
        for (std::size_t n = 0; n < numbers.size(); n++) {
            std::ofstream file(std::format("{}_{}_size_distribution.txt", m_prefix, numbers[n]));
            file << "Index\tdroplet_size\tLognormal\n";
            for (int i = 0; i < m_sizes.size(); i++) {
                file << std::format("{}\t{}\t{}\n", i, m_sizes[i], m_sizes_dist(i, n));
            }
        }

        // Resize alpha, diagonal
        // They are in Column Major order
        // ROWS -> sizes.size()
        // COLUMNS -> max_k + 1
        m_N_k_vec.resize(m_sizes.size(), max_k + 1);
        m_alpha.resize(m_sizes.size(), max_k + 1);
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

void Droplet::simulate(
    const std::size_t no_of_steps,
    const double final_x,
    const double pressure,
    const bool trajectory,
    Eigen::MatrixXd& y_out
) {

    const double stepsize = (final_x - 0) / no_of_steps;
    Eigen::MatrixXd y(m_sizes.size(), y_out.cols());

    // Set Initial conditions
    y.fill(0);
    y.col(0).setOnes();

    auto rk4 = create_backend();

    rk4->solve_ode(
        no_of_steps,
        stepsize,
        pressure,
        m_alpha.rows(),
        m_alpha.cols(),
        m_alpha.data(),
        trajectory,
        std::format("{}_trajectory.arrow", m_prefix),
        y.data()
    );

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
    if (m_type == "NONE") {
        y_out.noalias() = y;
    }
    else {
        // Now multiple y with size distribution
        // shape[k] = shape[Nxk].T shape[N]
        for (int n = 0; n < y_out.rows(); n++) {
            y_out.row(n) = y.transpose() * m_sizes_dist.col(n);
        }
    }
}
