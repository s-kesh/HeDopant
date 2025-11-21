#pragma once
#include "rk4_backend.hpp"

#include <Eigen/Dense>

class RK4Backend_CPU : public RK4Backend {
public:
    void solve_ode(
        const std::size_t number_of_steps,
        const double step_size,
        const double pressure,
        const int rows,
        const int cols,
        const double *alpha,
        const bool trajectory,
        const std::string& output_file,
        double *y
    );
private:
    void _fun(
        const double pressure,
        const Eigen::MatrixXd& alpha,
        Eigen::MatrixXd& diag,
        Eigen::MatrixXd& off_diag
    );
};
