#include "droplet.hpp"
#include <cstddef>
#include <exception>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

static void print_help(const char* prog) {
    std::println("Usage: {} [OPTIONS]", prog);
    std::println();
    std::println("Options:");
    std::println("  --config <file>     Path to configuration YAML file (default: config.yaml)");
    std::println("  --help              Show this help message");
    std::println();
    std::println("Configuration Overrides (these override the YAML file):");
    std::println("  --number_of_atoms <N>  Number of helium atoms");
    std::println("  --max_dopant <K>       Max dopant cluster size to track");
    std::println("  --doping_cell <L>      Length of the doping cell (in meters)");
    std::println("  --rk_steps <S>         Number of Runge-Kutta steps");
    std::println("  --dopant <name>        Dopant species (e.g., 'krypton', 'water')");
    std::println("  --dopant_pressure <P>  Dopant pressure (in mbar, will be converted to Pa)");
    std::println("  --output <prefix>      File prefix for output files");
    std::println("  --datadir <path>       Path to data directory");
    std::println("  --trajectory <bool>  Whether to output trajectory data per step");
}

int main(int argc, char* argv[]) {

    std::string filename = "config.yaml";
    std::vector<std::string> override_args;

    // All other args are saved for Pass 2
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
            filename = argv[++i]; // Consume the filename
        }

        else {
            // Save for Pass 2
            override_args.push_back(arg);
        }
    }

    // --- Load Config File ---
    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Error: Cannot load YAML file " + filename + ": " + e.what());
    }

    // This loop modifies the 'config' node before variables are read from it.
    try {
        for (size_t i = 0; i < override_args.size(); i++) {
            std::string arg = override_args[i];

            // Helper lambda to get the value for an option
            auto get_value = [&]() -> std::string {
                if (i + 1 >= override_args.size() || override_args[i + 1].starts_with("--")) {
                    throw std::runtime_error("Error: " + arg + " requires a value.");
                }
                return override_args[++i]; // Consume and return value
            };

            if (arg == "--number_of_atoms") {
                config["number_of_atoms"] = get_value();
            } else if (arg == "--max_dopant") {
                config["max_dopant"] = get_value();
            } else if (arg == "--doping_cell") {
                config["doping_cell"] = get_value();
            } else if (arg == "--rk_steps") {
                config["rk_steps"] = get_value();
            } else if (arg == "--dopant") {
                config["dopant"] = get_value();
            } else if (arg == "--dopant_pressure") {
                config["dopant_pressure"] = get_value();
            } else if (arg == "--output") {
                config["output"] = get_value();
            } else if (arg == "--datadir") {
                config["datadir"] = get_value();
            } else if (arg == "--trajectory") {
                config["trajectory"] = get_value();
            } else {
                throw std::runtime_error("Unknown option: " + arg);
            }
        }
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        std::println(stderr, "Use --help for usage information.");
        return 1;
    }


    std::size_t he_number;
    std::size_t max_k;
    std::size_t rk_steps;
    double L_cell;
    std::string dopant;
    double doping_pressure;
    std::string output;
    std::string datadir;
    bool trajectory;

    try {
        he_number = config["number_of_atoms"].as<std::size_t>();
        max_k = config["max_dopant"].as<std::size_t>();
        L_cell = config["doping_cell"].as<double>();
        rk_steps = config["rk_steps"].as<std::size_t>();
        dopant = config["dopant"].as<std::string>();
        doping_pressure = config["dopant_pressure"].as<double>() * 100; // to Pa
        output = config["output"].as<std::string>();
        datadir = config["datadir"].as<std::string>();
        trajectory = config["trajectory"].as<bool>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Error: Cannot read specific keys in YAML file " + filename +
            ". Make sure all keys are present in the file or provided as arguments: " +
            e.what());
    }

    Droplet helium(he_number, dopant, max_k, output, datadir);
    std::vector<double> y_final(max_k + 1);
    helium.evolove_rk(rk_steps, 0, L_cell, doping_pressure, trajectory, y_final.size(), y_final.data());

    // Write output to file
    // Header k, y_final
    std::ofstream file(std::format("{}_output.txt", output));
    file << "k\ty\n";
    for (std::size_t k = 0; k < max_k; k++) {
        file << std::format("{}\t{}\n", k, y_final[k]);
    }

    return 0;
}
