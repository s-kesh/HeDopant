#include "droplet.hpp"
#include "constants.hpp"
#include "interpolate.hpp"
#include "dopant.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/io/api.h>


#define ARROW_THROW_NOT_OK(expr) \
    do { auto _s = (expr); if (!_s.ok()) throw std::runtime_error(_s.ToString()); } while (0)

Droplet::Droplet(std::size_t number, std::string dopant_name, std::size_t max_k)
    : m_number(number), m_dopX(dopant_name)
{
    double con1 = (1.0 / constants::kB / m_dopX.temprature()) * 4.0 * constants::pi * constants::he_density * constants::he_density;
    double con2 = 4.5 * constants::kB * m_dopX.temprature() + (m_dopX.e_He_X() + m_dopX.e_X_X())*constants::e;

    m_vcluster.resize(number+5);
    m_evap.resize(number+5);
    auto interp = Interpolators("./data/droplet.txt");
    for (std::size_t i = 0; i < number+5; i++) {
        m_vcluster[i] = interp.V_cluster(i+1);
        m_evap[i] = interp.E_vap(i+1)*constants::kB;
    }

    _calculate_alpha(con1, con2, max_k);
}

Droplet::~Droplet()
{
}

void Droplet::_calculate_alpha(const double con1, const double con2, const std::size_t max_k) {
    // TO calculate alpha
    // α (k) = (1/kT) * sqrt(v_cluster^2 + vx^2 / v_cluster^2) * 4*pi*r_0^2 N_k^(2/3)
    alpha.resize(max_k + 1);
    double v_cluster = m_vcluster[m_number];
    double v_x = m_dopX.velocity();
    double E_total = 0.0;
    std::size_t N_evap = 0;
    double mass = m_dopX.mass()*constants::amu;

    std::size_t N_k = m_number;
    std::size_t N_old = m_number;

    alpha[0] = con1 * std::sqrt((v_cluster*v_cluster + v_x*v_x) / (v_cluster * v_cluster)) * std::pow(N_k, 2.0/3.0);
    for (std::size_t k = 1; k < max_k + 1; k++) {
        E_total = con2 + (0.5 * mass * v_cluster * v_cluster);
        N_evap = (std::size_t)(E_total / m_evap[N_old]);
        if (N_old > N_evap)
            N_k = (std::size_t)(N_old - N_evap);
        else
            N_k = 0;
        v_cluster = m_vcluster[N_old];
        alpha[k] = con1 * std::sqrt((v_cluster*v_cluster + v_x*v_x) / (v_cluster * v_cluster)) * std::pow(N_k, 2.0/3.0);
        N_old = N_k;
    }
}

void Droplet::_fun(
    const double pressure,
    const std::vector<double>& alph,
    const std::vector<double>& x,
    std::vector<double>& y)
{
    double mul_old = pressure * alph[0] * x[0];
    double mul_new = pressure * alph[0] * x[0];
    y[0] = -mul_new;
    for (std::size_t i = 1; i < y.size(); i++) {
        mul_new = pressure * alph[i] * x[i];
        y[i] = -mul_new + mul_old;
        mul_old = mul_new;
    }
}

void Droplet::evolove_rk(
    const std::size_t no_of_steps,
    const double initial_x,
    const double final_x,
    const double pressure,
    std::vector<double>& y
) {
    using arrow::DoubleBuilder;
    using arrow::Int64Builder;
    using arrow::ListBuilder;

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    const std::size_t maxx = y.size();
    y[0] = 1;
    for (std::size_t i = 0; i < maxx - 1; i++) y[i+1] = 0.0;

    auto schema = arrow::schema({
        arrow::field("step", arrow::int64()),
        arrow::field("y", arrow::list(arrow::float64()))
    });

    auto file_res = arrow::io::FileOutputStream::Open("trajectory.arrow");
    if (!file_res.ok())
        throw std::runtime_error("Cannot open trajectory.arrow");

    auto outfile = *file_res;

    auto writer_res = arrow::ipc::MakeStreamWriter(outfile, schema);
    if (!writer_res.ok())
        throw std::runtime_error("Cannot create IPC StreamWriter");

    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer = *writer_res;

    std::vector<double> k1(maxx), k2(maxx), k3(maxx), k4(maxx);
    std::vector<double> temp(maxx);

    const double stepsize = (final_x - initial_x) / no_of_steps;

    auto write_step = [&](std::size_t step_index, const std::vector<double>& vec) {
        Int64Builder step_builder(pool);
        ListBuilder list_builder(pool,
            std::make_shared<DoubleBuilder>(pool));
        DoubleBuilder* value_builder =
            static_cast<DoubleBuilder*>(list_builder.value_builder());

        // step index
        ARROW_THROW_NOT_OK(step_builder.Append((int64_t)step_index));

        // y vector
        ARROW_THROW_NOT_OK(list_builder.Append()); // start list
        for (double v : vec)
            ARROW_THROW_NOT_OK(value_builder->Append(v));

        // finalize arrays
        std::shared_ptr<arrow::Array> step_array;
        std::shared_ptr<arrow::Array> y_array;
        ARROW_THROW_NOT_OK(step_builder.Finish(&step_array));
        ARROW_THROW_NOT_OK(list_builder.Finish(&y_array));

        auto batch = arrow::RecordBatch::Make(
            schema, 1, {step_array, y_array});

        ARROW_THROW_NOT_OK(writer->WriteRecordBatch(*batch));
    };

    write_step(0, y);

    for (std::size_t step = 1; step <= no_of_steps; step++) {

        _fun(pressure, alpha, y, k1);

        for (std::size_t i = 0; i < maxx; i++)
            temp[i] = y[i] + 0.5 * stepsize * k1[i];
        _fun(pressure, alpha, temp, k2);

        for (std::size_t i = 0; i < maxx; i++)
            temp[i] = y[i] + 0.5 * stepsize * k2[i];
        _fun(pressure, alpha, temp, k3);

        for (std::size_t i = 0; i < maxx; i++)
            temp[i] = y[i] + stepsize * k3[i];
        _fun(pressure, alpha, temp, k4);

        for (std::size_t i = 0; i < maxx; i++)
            y[i] += (stepsize/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);

        // stream new y
        write_step(step, y);
    }

    ARROW_THROW_NOT_OK(writer->Close());
    ARROW_THROW_NOT_OK(outfile->Close());
}
