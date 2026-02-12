#include "rk4_cpu.hpp"

#ifdef HAS_ARROW
#include "arrow_io.hpp"
#endif


#include <Eigen/Dense>

void print_array(const double* dist) {
    for (int i = 0; i < 10; ++i) {
        std::cout << dist[i] << " ";
    }
    std::cout << std::endl;
}

void RK4Backend_CPU::_fun(
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

    x.block(1, 0, x.rows() - 1, x.cols()).noalias() =
        diag.block(0, 0, diag.rows() - 1, diag.cols()) -
        diag.block(1, 0, diag.rows() - 1, diag.cols());

    x.row(0).noalias() = -diag.row(0);
}


void RK4Backend_CPU::solve_ode(
    const std::size_t number_of_steps,
    const double step_size,
    const double pressure,
    const int rows,
    const int cols,
    const double* alpha_raw,
    const bool trajectory,
    const std::string& output_file,
    double *y_raw
)
{
    Eigen::MatrixXd temp(rows, cols), dia(rows, cols), kfinal(rows, cols);
    Eigen::Map<Eigen::MatrixXd> y(y_raw, rows, cols);
    Eigen::Map<const Eigen::MatrixXd> alpha(alpha_raw, rows, cols);

    float progress = 0.0f;
    #ifdef HAS_ARROW
    ArrowIO arrow_io(output_file);
    #endif

    if (trajectory) {
        #ifdef HAS_ARROW
        arrow_io.write_step(0, y.rows(), y.cols(), y.data());
        #endif
    }

    // print_progress_bar(progress);
    print_progress_bar(0, number_of_steps);
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

        // progress = static_cast<float>(i + 1) / number_of_steps;
        // if (100*i % number_of_steps == 0) print_progress_bar(progress);
        print_progress_bar(i + 1, number_of_steps);

        if (trajectory) {
            #ifdef HAS_ARROW
            arrow_io.write_step(i + 1, y.rows(), y.cols(), y.data());
            #endif
        }
    }
    // print_progress_bar(1.0);

    print_progress_bar(number_of_steps, number_of_steps);
    std::cout << std::endl;

    #ifdef HAS_ARROW
    arrow_io.close();
    #endif
}
