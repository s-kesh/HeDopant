#pragma once

#include <cstddef>
#include <iostream>

class RK4Backend {
public:
    virtual void solve_ode(
        const std::size_t number_of_steps,
        const double step_size,
        const double pressure,
        const int rows,
        const int cols,
        const double *alpha,
        const bool trajectory,
        const std::string& output_file,
        double *y
    )=0;
    virtual ~RK4Backend() = default;

    static void print_progress_bar(std::size_t current, std::size_t total) {
        int barWidth = 70;
        std::cout << "[";
        float progress = (1.0 * current) / total;
        int pos = barWidth*progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">>";
            else std::cout << " ";
        }
        std::cout << "] " << current << "/" << total << "\r";
        std::cout.flush();
    }
};
