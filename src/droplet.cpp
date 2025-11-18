#include "droplet.hpp"
#include "Eigen/Core"
#include "constants.hpp"
#include "interpolate.hpp"
#include "dopant.hpp"
#include "arrow_io.hpp"

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>

Droplet::Droplet(
    const std::size_t number,
    const std::string dopant_name,
    const std::size_t max_k,
    const std::string prefix,
    const std::string datadir
)
    : m_number(number), m_dopX(dopant_name, datadir)
    , m_prefix(prefix), m_datadir(datadir)
{
    double con1 = (1.0 / constants::kB / m_dopX.temprature()) * 4.0 * constants::pi * constants::he_density * constants::he_density;
    double con2 = 4.5 * constants::kB * m_dopX.temprature() + (m_dopX.e_He_X() + m_dopX.e_X_X())*constants::e;

    m_vcluster.resize(number+5);
    m_evap.resize(number+5);
    auto interp = Interpolators(std::format("{}/droplet.txt", m_datadir));
    for (std::size_t i = 0; i < number+5; i++) {
        m_vcluster[i] = interp.V_cluster(i+1);
        m_evap[i] = interp.E_vap(i+1)*constants::kB;
    }

    // Save the calculated vcluster and evap values to a file
    std::ofstream file(std::format("{}_vcluster_evap.txt", m_prefix));
    for (std::size_t i = 0; i < number+5; i++) {
        file << std::format("{}\t{}\t{}\n", i, m_vcluster[i], m_evap[i]);
    }
    file.close();

    // Resize alpha, diagonal
    m_alpha.resize(max_k + 1);
    m_diag.resize(max_k + 1);

    _calculate_alpha(con1, con2, max_k);
}

Droplet::~Droplet()
{
}

void Droplet::_calculate_alpha(const double con1, const double con2, const std::size_t max_k) {
    // TO calculate alpha
    // α (k) = (1/kT) * sqrt(v_cluster^2 + vx^2 / v_cluster^2) * 4*pi*r_0^2 N_k^(2/3)
    double v_cluster = m_vcluster[m_number];
    double v_x = m_dopX.velocity();
    double E_total = 0.0;
    std::size_t N_evap = 0;
    double mass = m_dopX.mass()*constants::amu;

    std::size_t N_k = m_number;
    std::size_t N_old = m_number;

    N_k_vec.resize(max_k + 1);
    N_k_vec[0] = N_k;

    m_alpha[0] = con1 * std::sqrt((v_cluster*v_cluster + v_x*v_x) / (v_cluster * v_cluster)) * std::pow(N_k, 2.0/3.0);
    for (std::size_t k = 1; k < max_k + 1; k++) {
        E_total = con2 + (0.5 * mass * v_cluster * v_cluster);
        N_evap = (std::size_t)(E_total / m_evap[N_old]);
        if (N_old > N_evap)
            N_k = (std::size_t)(N_old - N_evap);
        else
            N_k = 0;
        N_k_vec[k] = N_k;
        v_cluster = m_vcluster[N_old];
        m_alpha[k] = con1 * std::sqrt((v_cluster*v_cluster + v_x*v_x) / (v_cluster * v_cluster)) * std::pow(N_k, 2.0/3.0);
        N_old = N_k;
    }

    // Output in a text file
    // Header: k, N_k, alpha
    std::ofstream file(std::format("{}_evap.txt", m_prefix));
    file << "k\tN_k\talpha\n";
    for (std::size_t k = 0; k < max_k + 1; k++) {
        file << std::format("{}\t{}\t{}\n", k, N_k_vec[k], m_alpha[k]);
    }
    file.close();

}

void Droplet::_fun(
    const double pressure,
    Eigen::VectorXd& x
)
{
    // y[0] = -pressure * alpha[0] * x[0];
    // y[i] = diag[i-1] - diag[i]; for i > 0
    //
    std::size_t size = x.size();
    m_diag = pressure * m_alpha.array() * x.array();
    x[0] = -m_diag[0];
    x.segment(1, size-1).noalias() = m_diag.segment(0, size-1) - m_diag.segment(1, size-1);
}

void Droplet::evolove_rk(
    const std::size_t no_of_steps,
    const double initial_x,
    const double final_x,
    const double pressure,
    const bool trajectory,
    const std::string filename,
    std::vector<double>& y_og
) {

    // setup output file

    ArrowIO arrowIO(std::format("{}_{}.arrow", m_prefix, filename));

    const std::size_t maxx = y_og.size();
    Eigen::Map<Eigen::VectorXd> y(y_og.data(), y_og.size());

    const double stepsize = (final_x - initial_x) / no_of_steps;

    if (trajectory) arrowIO.write_step(0, y_og);

    Eigen::VectorXd temp(maxx), kfinal(maxx);

    for (std::size_t step = 1; step <= no_of_steps; step++) {
        temp.noalias() = y;
        _fun(pressure, temp); // now temp = f(y); k1
        kfinal.noalias() = temp; // k_final = k1

        temp.noalias() = y + 0.5*stepsize * temp; // temp = y + (h/2)k1
        _fun(pressure, temp); // now temp = f(y + (h/2)*k1); k2
        kfinal.noalias() += 2.0*temp; // k_final = k1 + 2k2

        temp.noalias() = y + 0.5*stepsize * temp; // temp = y + (h/2)k2
        _fun(pressure, temp); // now temp = f(y + (h/2)*k2); k3
        kfinal.noalias() += 2.0*temp; // k_final = k1 + 2k2 + 2k3

        temp.noalias() = y + stepsize * temp; // temp = y + h*k3
        _fun(pressure, temp); // now temp = f(y + h*k3); k4
        kfinal.noalias() += temp; // k_final = k1 + 2k2 + 2k3 + k4

        // Make RK4 step
        // y += (h/6)*(k1 + 2k2 + 2k3 + k4)
        y.noalias() += (stepsize/6.0) * kfinal;

        // stream new y
        if (trajectory) arrowIO.write_step(step, y_og);
    }

    arrowIO.close();
}
