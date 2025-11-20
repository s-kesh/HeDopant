#pragma once

 // #include "arrow_io.hpp"

#include <cstddef>
#include <iostream>

static void print_progress_bar(float progress) {
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

class RK4Backend {
public:
    virtual void solve_ode(
        const std::size_t number_of_steps,
        const double step_size,
        const double pressure,
        const int rows,
        const int cols,
        const double *alpha,
        // const bool trajectory,
        // ArrowIO& arrow_io,
        double *y
    )=0;
    virtual ~RK4Backend() = default;
};

class RK4Backend_CPU : public RK4Backend {
public:
    void solve_ode(
        const std::size_t number_of_steps,
        const double step_size,
        const double pressure,
        const int rows,
        const int cols,
        const double *alpha,
        // const bool trajectory,
        // ArrowIO& arrow_io,
        double *y
    );
};

class RK4Backend_GPU : public RK4Backend {
public:
    void solve_ode(
        const std::size_t number_of_steps,
        const double step_size,
        const double pressure,
        const int rows,
        const int cols,
        const double *alpha,
        // const bool trajectory,
        // ArrowIO& arrow_io,
        double *y
    );
};
