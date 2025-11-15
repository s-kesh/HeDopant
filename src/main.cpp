#include "droplet.hpp"
#include <cstddef>
#include <exception>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

static void print_help(const char* prog) {
    std::println("Usage: {} [--config FILE] [--help]", prog);
    std::println();
    std::println("Options:");
    std::println("  --config <file>   Path to configuration YAML file (default: config.yaml)");
    std::println("  --help            Show this help message");
}

int main(int argc, char* argv[]) {

    std::string filename = "config.yaml";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_help(argv[0]);
            return 0;
        }

        else if (arg == "--config") {
            if (i + 1 >= argc) {
                std::println(stderr, "Error: --config requires a file path.");
                return 1;
            }
            filename = argv[++i];
        }

        else {
            std::println(stderr, "Unknown option: {}", arg);
            std::println(stderr, "Use --help for usage information.");
            return 1;
        }
    }

    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Error: Cannot load YAML file " + filename + ": " + e.what());
    }

    std::size_t he_number;
    std::size_t max_k;
    std::size_t rk_steps;
    double L_cell;
    std::string dopant;
    double doping_pressure;

    try {
        he_number = config["number_of_atoms"].as<std::size_t>();
        max_k = config["max_dopant"].as<std::size_t>();
        L_cell = config["doping_cell"].as<double>();
        rk_steps = config["rk_steps"].as<std::size_t>();
        dopant = config["dopant"].as<std::string>();
        doping_pressure = config["dopant_pressure"].as<double>() * 100; // to Pa
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Error: Cannot read specific keys in YAML file " + filename +
            ": " + e.what());
    }

    Droplet helium(he_number, dopant, max_k);
    std::vector<double> y_final(max_k + 1);
    helium.evolove_rk(rk_steps, 0, L_cell, doping_pressure, y_final);

    for (std::size_t k = 0; k < max_k; k++) {
        std::println("{}\t{}", k, y_final[k]);
    }

    return 0;
}
