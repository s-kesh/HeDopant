#include "droplet.hpp"
#include "Eigen/Core"
#include "constants.hpp"
#include "interpolate.hpp"
#include "dopant.hpp"

#include <cstddef>
#include <vector>
#include <fstream>
#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/io/api.h>


#define ARROW_THROW_NOT_OK(expr) \
    do { auto _s = (expr); if (!_s.ok()) throw std::runtime_error(_s.ToString()); } while (0)

Droplet::Droplet(
    const std::size_t number,
    const std::string dopant_name,
    const std::size_t max_k,
    const std::string output,
    const std::string datadir
)
    : m_number(number), m_dopX(dopant_name, datadir)
    , m_output(output), m_datadir(datadir)
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
    std::ofstream file(std::format("{}_vcluster_evap.txt", m_output));
    for (std::size_t i = 0; i < number+5; i++) {
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
    m_alpha.resize(max_k + 1);
    m_diag.resize(max_k + 1);
    double v_cluster = m_vcluster[m_number];
    double v_x = m_dopX.velocity();
    double E_total = 0.0;
    std::size_t N_evap = 0;
    double mass = m_dopX.mass()*constants::amu;

    std::size_t N_k = m_number;
    std::size_t N_old = m_number;

    std::vector<std::size_t> N_k_vec(max_k + 1);
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
    std::ofstream file(std::format("{}_evap.txt", m_output));
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
    const std::size_t size,
    double *y_raw
) {

    using arrow::DoubleBuilder;
    using arrow::Int64Builder;
    using arrow::ListBuilder;

    arrow::MemoryPool* pool = arrow::default_memory_pool();

    Eigen::Map<Eigen::VectorXd> y(y_raw, size);

    const std::size_t maxx = y.size();
    y.fill(0.0);
    y[0] = 1;

    auto schema = arrow::schema({
        arrow::field("step", arrow::int64()),
        arrow::field("y", arrow::list(arrow::float64()))
    });

    auto file_res = arrow::io::FileOutputStream::Open(
        std::format("{}_trajectory.arrow", m_output)
    );
    if (!file_res.ok())
        throw std::runtime_error("Cannot open trajectory.arrow");

    auto outfile = *file_res;

    auto writer_res = arrow::ipc::MakeStreamWriter(outfile, schema);
    if (!writer_res.ok())
        throw std::runtime_error("Cannot create IPC StreamWriter");

    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer = *writer_res;

    const double stepsize = (final_x - initial_x) / no_of_steps;

    auto write_step = [&](std::size_t step_index, const double *vec) {
        Int64Builder step_builder(pool);
        ListBuilder list_builder(pool,
            std::make_shared<DoubleBuilder>(pool));
        DoubleBuilder* value_builder =
            static_cast<DoubleBuilder*>(list_builder.value_builder());

        // step index
        ARROW_THROW_NOT_OK(step_builder.Append((int64_t)step_index));

        // y vector
        ARROW_THROW_NOT_OK(list_builder.Append()); // start list
        for (std::size_t i = 0; i < size; i++)
            ARROW_THROW_NOT_OK(value_builder->Append(vec[i]));

        // finalize arrays
        std::shared_ptr<arrow::Array> step_array;
        std::shared_ptr<arrow::Array> y_array;
        ARROW_THROW_NOT_OK(step_builder.Finish(&step_array));
        ARROW_THROW_NOT_OK(list_builder.Finish(&y_array));

        auto batch = arrow::RecordBatch::Make(
            schema, 1, {step_array, y_array});

        ARROW_THROW_NOT_OK(writer->WriteRecordBatch(*batch));
    };

    if (trajectory) write_step(0, y.data());

    Eigen::VectorXd temp(maxx), kfinal(maxx);

    for (std::size_t step = 1; step <= no_of_steps; step++) {
        temp = y;
        _fun(pressure, temp); // now temp = f(y)
        kfinal.noalias() = temp; // k_final = k1

        temp = y + 0.5 * stepsize * temp;
        _fun(pressure, temp); // now temp = f(y + (h/2)*k1)
        kfinal.noalias() += 2.0*temp; // k_final = k1 + 2k2

        temp = y + 0.5 * stepsize * temp;
        _fun(pressure, temp); // now temp = f(y + (h/2)*k2)
        kfinal.noalias() += 2.0*temp; // k_final = k1 + 2k2 + 2k3

        temp = y + stepsize * temp;
        _fun(pressure, temp); // now temp = f(y + h*k3)
        kfinal.noalias() += temp; // k_final = k1 + 2k2 + 2k3 + k4

        // Make RK4 step
        y.noalias() += (stepsize/6.0) * kfinal;

        // stream new y
        if (trajectory) write_step(step, y.data());
    }

    ARROW_THROW_NOT_OK(writer->Close());
    ARROW_THROW_NOT_OK(outfile->Close());
}
