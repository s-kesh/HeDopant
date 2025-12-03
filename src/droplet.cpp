#include <cstddef>
#include <print>
#include <fstream>

#include "droplet.hpp"
#include "Eigen/Core"
#include "constants.hpp"
#include "interpolate.hpp"
#include "dopant.hpp"

#include "distribution.hpp"
#include "log_normal.hpp"
#include "exponential.hpp"

#include "rk4_backend.hpp"
#include "rk4_cpu.hpp"

#ifdef RK4_BACKEND_CUDA
#include <cuda_runtime.h>
#include "rk4_cuda.hpp"
#endif

#ifdef RK4_BACKEND_OPENCL
#include "rk4_opencl.hpp"
#endif

std::shared_ptr<RK4Backend> create_backend()
{
#ifdef RK4_BACKEND_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        std::println("GPU backend selected");
        return std::make_unique<RK4Backend_CUDA>();
    }
#endif
#ifdef RK4_BACKEND_OPENCL
    std::println("OpenCL backend selected");
    return std::make_unique<RK4Backend_OpenCL>();
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
    const std::size_t dist_step,
    const std::string dopant_name,
    const std::size_t max_k,
    const std::string outdir,
    const std::string datadir
)
    : m_numbers(numbers), m_type(type), m_dist_step(dist_step),
    m_dopX(dopant_name, datadir), m_output(outdir), m_datadir(datadir)
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
        m_N_k_vec.resize(max_k+1, m_numbers.size());
        m_alpha.resize(max_k+1, m_numbers.size());
    }
    else {
        m_vcluster.resize(12*max_mean_number);
        m_evap.resize(12*max_mean_number);

        // Creating an array from 0 to 10*max_mean_number with step of DIST_STEP
        int size = 0;
        if (type == "LOGNORMAL") size = (5*max_mean_number - 0) / dist_step;
        else size = (10*max_mean_number - 0) / dist_step;

        m_sizes.resize(size);
        for (int i = 0; i < m_sizes.size(); i++) {
            m_sizes[i] = 0 + dist_step*i;
        }

        m_sizes_dist.resize(m_sizes.size(), numbers.size());
        for (std::size_t n = 0; n < numbers.size(); n++) {
            auto drop_dist = create_distribution(type, numbers[n]);
            for (int i = 0; i < m_sizes.size(); i++) {
                m_sizes_dist(i, n) = drop_dist->pdf(m_sizes[i]);
            }
        }

        // Resize alpha, diagonal
        // They are in Column Major order
        // ROWS -> sizes.size()
        // COLUMNS -> max_k + 1
        m_N_k_vec.resize(max_k + 1, m_sizes.size());
        m_alpha.resize(max_k + 1, m_sizes.size());
    }

    // Save size distributions to a txt file
    std::ofstream file(std::format("{}/dist_N.txt", m_output));
    std::print(file, "droplet_size");
    for (std::size_t n = 0; n < numbers.size(); n++) {
        std::print(file, "\t{}", numbers[n]);
    }
    std::print(file, "\n");

    for (int i = 0; i < m_sizes.size(); i++) {
        std::print(file, "{}", m_sizes[i]);
        for (std::size_t n = 0; n < numbers.size(); n++) {
            std::print(file, "\t{}", m_sizes_dist(i, n));
        }
        std::print(file, "\n");
    }
    file.close();

    auto interp = Interpolators(std::format("{}/droplet.txt", m_datadir));
    for (std::size_t i = 0; i < m_vcluster.size(); i++) {
        m_vcluster[i] = interp.V_cluster(i+1);
        m_evap[i] = interp.E_vap(i+1)*constants::kB;
    }

    // Save the calculated vcluster and evap values to a file
    std::ofstream filev(std::format("{}/vcluster_ebe.txt", m_output));
    std::println(filev, "droplet_size\tvelocity\tbinding_energy");
    for (std::size_t i = 0; i < max_mean_number+5; i++) {
        std::println(filev, "{}\t{}\t{}", i, m_vcluster[i], m_evap[i]);
    }
    filev.close();

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

        m_N_k_vec(0, i) = N_k;
        m_alpha(0, i) = con1 * std::sqrt((v_cluster2 + v_x2)/ v_cluster2) * std::pow(N_k, 2.0/3.0);

        for (std::size_t k = 1; k < max_k + 1; k++) {
            E_total = con2 + (0.5 * mass * v_cluster2);
            N_evap = (std::size_t)(E_total / m_evap[N_old]);
            if (N_old > N_evap)
                N_k = (std::size_t)(N_old - N_evap);
            else
                N_k = 0;
            m_N_k_vec(k, i) = N_k;
            v_cluster2 = m_vcluster[N_k] * m_vcluster[N_k];
            m_alpha(k, i) = con1 * std::sqrt((v_cluster2 + v_x2) / v_cluster2) * std::pow(N_k, 2.0/3.0);
            N_old = N_k;
        }
    }

    // Output in a text file
    // Header: k, N_k, alpha
    std::ofstream file(std::format("{}/evap.txt", m_output));
    for (std::size_t i = 0; i < (std::size_t)m_sizes.size(); i++) {
        std::println(file, "=========================");
        std::println(file, "For Initial size: {}", m_sizes[i]);
        std::println(file, "k\tN_k\talpha");
        for (std::size_t k = 0; k < max_k + 1; k++) {
            std::println(file, "{}\t{}\t{}", k, m_N_k_vec(k, i), m_alpha(k, i));
        }
        std::println(file, "=========================");
    }
    file.close();
}

void Droplet::simulate(
    const std::vector<std::size_t> no_of_steps,
    const std::vector<double> doping_length,
    const std::vector<double> pressure,
    const bool trajectory,
    const std::string input,
    Eigen::MatrixXd& y_out,
    Eigen::MatrixXd& N_k_out
) {

    Eigen::MatrixXd y(y_out.rows(), m_sizes.size());

    // Set Initial conditions
    y.fill(0);
    y.row(0).setOnes();

    // initialize N_k
    Eigen::MatrixXd N_k;
    if (m_type != "NONE") {
        N_k.resize(m_sizes.size(), m_sizes.size());
        N_k_out.resize(m_sizes.size(), y_out.cols());
    } else {
        N_k.resize((max_mean_number/m_dist_step)+1, m_sizes.size());
        N_k_out.resize((max_mean_number/m_dist_step)+1, y_out.cols());
    }

    N_k.fill(0);
    N_k_out.fill(0);

    // Input file exists and is not empty
    if (!input.empty()) {
        std::ifstream file(input, std::ios::binary);
        if (file.is_open()) {
            // Read the binary data to y
            file.read(reinterpret_cast<char*>(y.data()), y.size() * sizeof(double));
        }
    }

    // RK4 Solver
    auto rk4 = create_backend();

    std::size_t ll = no_of_steps.size();
    for (std::size_t i = 0; i < ll; i++) {
        const double stepsize = doping_length[i]/ no_of_steps[i];
        const double pr = pressure[i]*100;

        rk4->solve_ode(
            no_of_steps[i],
            stepsize,
            pr,
            m_alpha.rows(),
            m_alpha.cols(),
            m_alpha.data(),
            trajectory,
            std::format("{}/trajectory_{}.arrow", m_output, i),
            y.data()
        );

        _calculate_nk_distribution(y, N_k);

        // Save final y in a binary file
        std::ofstream filebin(std::format("{}/final_y_{}.bin", m_output, i), std::ios::binary);
        filebin.write(reinterpret_cast<char*>(y.data()), y.size() * sizeof(double));
        filebin.close();

        // Save final y and N_k
        // Output in a text file
        std::ofstream filetxt_k(std::format("{}/final_k_{}.txt", m_output, i));
        std::ofstream filetxt_Nk(std::format("{}/final_Nk_{}.txt", m_output, i));
        for (std::size_t i = 0; i < (std::size_t)y.cols(); i++) {
            std::println(filetxt_k,  "=========================");
            std::println(filetxt_Nk, "=========================");
            std::println(filetxt_k, "For Initial size: {}", m_sizes[i]);
            std::println(filetxt_Nk, "For Initial size: {}", m_sizes[i]);
            std::println(filetxt_k, "k\ty");
            std::println(filetxt_Nk, "N_k\ty");
            for (std::size_t k = 0; k < (std::size_t)y.rows(); k++) {
                std::println(filetxt_k, "{}\t{}", k, y(k, i));
            }
            for (std::size_t nk = 0; nk < (std::size_t)N_k.rows(); nk++) {
                std::println(filetxt_Nk, "{}\t{}", nk, N_k(nk, i));
            }
            std::println(filetxt_k,  "=========================");
            std::println(filetxt_Nk, "=========================");
        }
        filetxt_k.close();
        filetxt_Nk.close();
    }

    // Now for each mean number, we have a distribution
    if (m_type == "NONE") {
        // Distribution of dopant size
        y_out.noalias() = y;
        N_k_out.noalias() = N_k;
    }
    else {
        // Now multiple y with size distribution
        // shape[k] = shape[kxN] shape[N]
        y_out.noalias() = y * m_sizes_dist;
        N_k_out.noalias() = N_k * m_sizes_dist;
    }
}

void Droplet::_calculate_nk_distribution(
    const Eigen::MatrixXd& Ik,
    Eigen::MatrixXd& Nk
)
{
    int rows = Ik.rows();
    int cols = Ik.cols();
    for (int i = 0; i < cols; i++) {
        for (int k = 0; k < rows; k++) {
            int initial_size = m_N_k_vec(k, i);
            Nk(initial_size/m_dist_step, i) += Ik(k, i);
        }
    }
}
