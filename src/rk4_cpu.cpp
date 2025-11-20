#include "rk4_backend.hpp"

#include <iostream>
#include <Eigen/Dense>

void _fun(
    const double pressure,
    const Eigen::MatrixXd& alpha,
    Eigen::MatrixXd& diag,
    Eigen::MatrixXd& x)
{
    // shape of x: N x k
    // diag = pressure * alpha * x
    // y[0] = -diag[0]
    // y[i] = diag[i-1] - diag[i]; for i > 0

    diag = pressure * alpha.array() * x.array();

    x.block(0, 1, x.rows(), x.cols() - 1).noalias() =
        diag.block(0, 0, diag.rows(), diag.cols() - 1) -
        diag.block(0, 1, diag.rows(), diag.cols() - 1);

    x.col(0).noalias() = -diag.col(0);
}


void RK4Backend_CPU::solve_ode(
    const std::size_t number_of_steps,
    const double step_size,
    const double pressure,
    const int rows,
    const int cols,
    const double* alpha_raw,
    // const bool trajectory,
    // ArrowIO& arrow_io,
    double *y_raw
)
{
    Eigen::MatrixXd temp(rows, cols), dia(rows, cols), kfinal(rows, cols);
    Eigen::Map<Eigen::MatrixXd> y(y_raw, rows, cols);
    Eigen::Map<const Eigen::MatrixXd> alpha(alpha_raw, rows, cols);

    float progress = 0.0f;
    for (std::size_t i = 0; i < number_of_steps; ++i)
    {
        // k1
        temp.noalias() = y;
        _fun(pressure, alpha, dia, temp); // now temp = f(y); k1
        kfinal.noalias() = temp; // kfinal = k1

        // k2
        temp.noalias() = y + 0.5 * step_size * temp;
        _fun(pressure, alpha, dia, temp); // now temp = f(y + 0.5 * step_size * k1); k2
        kfinal.noalias() += 2 * temp; // kfinal = k1 + 2k2

        // k3
        temp.noalias() = y + 0.5 * step_size * temp;
        _fun(pressure, alpha, dia, temp); // now temp = f(y + 0.5 * step_size * k2); k3
        kfinal.noalias() += 2 * temp; // kfinal = k1 + 2k2 + 2k3

        // k4
        temp.noalias() = y + step_size * temp;
        _fun(pressure, alpha, dia, temp); // now temp = f(y + step_size * k3); k4
        kfinal.noalias() += temp; // kfinal = k1 + 2k2 + 2k3 + k4

        // update y
        y.noalias() += step_size * kfinal / 6.0;

        progress = static_cast<float>(i + 1) / number_of_steps;
        if (100*i % number_of_steps == 0) print_progress_bar(progress);

        // if (trajectory) {
        //     arrow_io.write_step(i, y.size(), y.data());
        // }
    }
    print_progress_bar(1.0);
    std::cout << std::endl;
}
