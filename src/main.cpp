#include "droplet.hpp"

#include <cstddef>
#include <exception>
#include <format>
#include <numeric>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>

#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "rk4_backend.hpp"

#ifdef RK4_HAS_GPU
#include <cuda_runtime.h>
#endif

std::shared_ptr<RK4Backend> create_backend()
{
#ifdef RK4_HAS_GPU
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    std::println("GPU backend should be selected, device count: {}", count);
    if (status == cudaSuccess && count > 0) {
        std::println("GPU backend selected");
        return std::make_shared<RK4Backend_GPU>();
    }
#endif
    std::println("CPU backend selected");
    return std::make_shared<RK4Backend_CPU>();
}

/*
 * Print help message
 */
static void print_help(const char* prog) {
    std::println("Usage: {} [OPTIONS]", prog);
    std::println();
    std::println("Options:");
    std::println("  --config <file>     Path to configuration YAML file (default: config.yaml)");
    std::println("  --help              Show this help message");
    std::println();
    std::println("Configuration Overrides (these override the YAML file):");
    std::println("  --number_of_atoms <N>  Number of helium atoms");
    std::println("  --type <TYPE>          Distribution type: NONE or LOGNORMAL");
    std::println("  --max_dopant <K>       Max dopant cluster size to track");
    std::println("  --doping_cell <L>      Length of the doping cell (in meters)");
    std::println("  --rk_steps <S>         Number of Runge-Kutta steps");
    std::println("  --dopant <name>        Dopant species (e.g., 'krypton', 'water')");
    std::println("  --dopant_pressure <P>  Dopant pressure (in mbar, will be converted to Pa)");
    std::println("  --output <prefix>      File prefix for output files");
    std::println("  --datadir <path>       Path to data directory");
    std::println("  --trajectory <bool>  Whether to output trajectory data per step");
}

/*
 * Print configuration details
 */
static void print_config(const YAML::Node& config) {
    std::println("Configuration:");

    // Check if number_of_atoms is a sequence
    if (config["number_of_atoms"].IsSequence()) {
        std::println("  Number of atoms: {}", config["number_of_atoms"].as<std::vector<std::size_t>>());
    } else {
        std::println("  Number of atoms: {}", config["number_of_atoms"].as<std::size_t>());
    }
    std::println("  Type: {}", config["type"].as<std::string>());
    std::println("  Max dopant: {}", config["max_dopant"].as<int>());
    std::println("  Doping cell: {}", config["doping_cell"].as<double>());
    std::println("  Runge-Kutta steps: {}", config["rk_steps"].as<int>());
    std::println("  Dopant: {}", config["dopant"].as<std::string>());

    // Check if doping pressure is a sequence
    if (config["dopant_pressure"].IsSequence()) {
        std::println("  Dopant pressures: {}", config["dopant_pressure"].as<std::vector<double>>());
    } else {
        std::println("  Dopant pressure: {}", config["dopant_pressure"].as<double>());
    }

    std::println("  Output prefix: {}", config["output"].as<std::string>());
    std::println("  Data directory: {}", config["datadir"].as<std::string>());
    std::println("  Trajectory: {}", config["trajectory"].as<bool>());
}

/*
 * Calculate mean of a distribution
 */
template<typename T>
T calculate_mean(std::vector<T>& distribution) {
    std::size_t size = distribution.size();
    T sum_dist = std::reduce(distribution.begin(), distribution.end(), 0.0);

    // k*I_k
    std::vector<T> k_I_k(size);
    for (std::size_t i = 0; i < size; i++) {
        k_I_k[i] = i*distribution[i];
    }
    T sum_kIk = std::reduce(k_I_k.begin(), k_I_k.end(), 0.0);
    return sum_kIk/sum_dist;
}

/*
 * The main function
 *
 * This function is the entry point of the program.
 * It parses command-line arguments, loads configuration settings,
 * and performs the main computation.
 */
int main(int argc, char* argv[]) {

    // Default name for the configuration file
    std::string filename = "config.yaml";
    std::vector<std::string> override_args;

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

        // Read the other arguments then config file if need overload
        else {
            override_args.push_back(arg);
        }
    }

    // --- Load Config File ---
    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        print_help(argv[0]);
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
            } else if (arg == "--type") {
                config["type"] = get_value();
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

    // Print configuration
    print_config(config);


    // Read all the variables from config file or overriden arguments
    std::vector<std::size_t> he_numbers;
    std::string type;
    std::size_t max_k;
    std::size_t rk_steps;
    double L_cell;
    std::string dopant;
    std::vector<double> doping_pressure;
    std::string output;
    std::string datadir;
    bool trajectory;

    try {
        if (config["number_of_atoms"].IsSequence())
            he_numbers = config["number_of_atoms"].as<std::vector<std::size_t>>();
        else
            he_numbers.push_back(config["number_of_atoms"].as<std::size_t>());
        type = config["type"].as<std::string>();
        max_k = config["max_dopant"].as<std::size_t>();
        L_cell = config["doping_cell"].as<double>();
        rk_steps = config["rk_steps"].as<std::size_t>();
        dopant = config["dopant"].as<std::string>();
        // Check if doping_pressure is a Sequence
        if (config["dopant_pressure"].IsSequence())
            doping_pressure = config["dopant_pressure"].as<std::vector<double>>();
        else
            doping_pressure.push_back(config["dopant_pressure"].as<double>());
        output = config["output"].as<std::string>();
        datadir = config["datadir"].as<std::string>();
        trajectory = config["trajectory"].as<bool>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Error: Cannot read specific keys in YAML file " + filename +
            ". Make sure all keys are present in the file or provided as arguments: " +
            e.what());
    }

    // To keep state of intensities
    // one vector for each mean_size
    Eigen::MatrixXd y_matrix(he_numbers.size(), max_k+1);
    y_matrix.fill(0.0);

    // Simulation
    auto rk4_solver = create_backend();

    // Define the Droplet object
    Droplet helium(he_numbers, type,
        dopant, max_k, output, datadir);

    // if pressure is a list, then iterate over it
    for (double pressure : doping_pressure) {
        // Solve dI_k/dz = AI_k
        helium.evolove_rk(
            rk4_solver, // Solver backend
            rk_steps, // Number of steps for Runge-Kutta method
            L_cell, // Length of doping cell
            pressure*100, // Pressure in Pa
            trajectory, // Flag to save trajectory
            "trajectory", // Filename for trajectory data
            y_matrix // Initial condition for y; final condition would be saved in it
        );

        // Write output to file
        // Header k, y_final
        std::string name = std::format("{}_{}_mbar_output.txt", output, pressure);
        std::ofstream file(name);
        file << "k\t";
        for (std::size_t n = 0; n < he_numbers.size(); n++) {
            file << n << "\t";
        }
        file << "\n";
        for (std::size_t k = 0; k < max_k; k++) {
            file << k << "\t";
            for (std::size_t n = 0; n < he_numbers.size(); n++) {
                file << y_matrix(n, k) << "\t";
            }
            file << "\n";
        }
    }


    return 0;
}
