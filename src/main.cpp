#include "droplet.hpp"

#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
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

struct DistributionStats {
    double mean;
    double mode;
    double stddev;
};

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
    std::println("  --dist_step <D>        Step size for distribution");
    std::println("  --max_dopant <K>       Max dopant cluster size to track");
    std::println("  --doping_cell <L>      Length of the doping cell (in meters)");
    std::println("  --rk_steps <S>         Number of Runge-Kutta steps");
    std::println("  --dopant <name>        Dopant species (e.g., 'krypton', 'water')");
    std::println("  --dopant_pressure <P>  Dopant pressure (in mbar, will be converted to Pa)");
    std::println("  --input <path>         Path to input matrix");
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
    std::println("  Step size: {}", config["dist_step"].as<int>());
    std::println("  Max dopant: {}", config["max_dopant"].as<int>());

    // Check if doping cell is a sequence
    if (config["doping_cell"].IsSequence()) {
        std::println("  Doping cells: {}", config["doping_cell"].as<std::vector<double>>());
    } else {
        std::println("  Doping cell: {}", config["doping_cell"].as<double>());
    }

    // Check if Runge-Kutta steps is a sequence
    if (config["rk_steps"].IsSequence()) {
        std::println("  Runge-Kutta steps: {}", config["rk_steps"].as<std::vector<int>>());
    } else {
        std::println("  Runge-Kutta steps: {}", config["rk_steps"].as<int>());
    }

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

    std::println("  Input matrix: {}", config["input"].as<std::string>());
}

/*
 * Calculate distribution stats
 */
inline DistributionStats compute_distribution_stats(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& w
)
{
    DistributionStats stats{};

    const double wsum = w.sum();
    if (wsum == 0.0) {
        stats.mean   = 0.0;
        stats.mode   = 0.0;
        stats.stddev = 0.0;
        return stats;
    }

    // Mean
    stats.mean = w.dot(x) / wsum;

    // Mode
    Eigen::Index mode_idx;
    w.maxCoeff(&mode_idx);
    stats.mode = x(mode_idx);

    // Standard deviation (population)
    Eigen::VectorXd diff = x.array() - stats.mean;
    double variance = w.dot(diff.array().square().matrix()) / wsum;
    stats.stddev = std::sqrt(variance);

    return stats;
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
                return override_args[++i];
            };

            if (arg == "--number_of_atoms") {
                config["number_of_atoms"] = get_value();
            } else if (arg == "--type") {
                config["type"] = get_value();
            } else if (arg == "--dist_step") {
                config["dist_step"] = get_value();
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
            } else if (arg == "--input") {
                config["input"] = get_value();
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
    std::size_t dist_step;
    std::size_t max_k;
    std::vector<std::size_t> rk_steps;
    std::vector<double> L_cell;
    std::string dopant;
    std::vector<double> doping_pressure;
    std::string input;
    std::string output;
    std::string datadir;
    bool trajectory;

    try {
        if (config["number_of_atoms"].IsSequence())
            he_numbers = config["number_of_atoms"].as<std::vector<std::size_t>>();
        else
            he_numbers.push_back(config["number_of_atoms"].as<std::size_t>());
        type = config["type"].as<std::string>();
        dist_step = config["dist_step"].as<std::size_t>();
        max_k = config["max_dopant"].as<std::size_t>();
        if (config["doping_cell"].IsSequence())
            L_cell = config["doping_cell"].as<std::vector<double>>();
        else
            L_cell.push_back(config["doping_cell"].as<double>());
        if (config["rk_steps"].IsSequence())
            rk_steps = config["rk_steps"].as<std::vector<std::size_t>>();
        else
            rk_steps.push_back(config["rk_steps"].as<std::size_t>());
        dopant = config["dopant"].as<std::string>();

        // Check if doping_pressure is a Sequence
        if (config["dopant_pressure"].IsSequence())
            doping_pressure = config["dopant_pressure"].as<std::vector<double>>();
        else
            doping_pressure.push_back(config["dopant_pressure"].as<double>());

        input = config["input"].as<std::string>();
        output = config["output"].as<std::string>();
        datadir = config["datadir"].as<std::string>();
        trajectory = config["trajectory"].as<bool>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Error: Cannot read specific keys in YAML file " + filename +
            ". Make sure all keys are present in the file or provided as arguments: " +
            e.what());
    }

    // Create output directory
    std::filesystem::create_directory(output);

    // To keep state of intensities
    // one vector for each mean_size
    Eigen::MatrixXd I_k_matrix(max_k+1, he_numbers.size());
    std::size_t max_size = *std::max_element(he_numbers.begin(), he_numbers.end());
    Eigen::MatrixXd N_k_matrix(max_size, he_numbers.size());

    // Define the Droplet object
    Droplet helium(
        he_numbers,
        type,
        dist_step,
        dopant,
        max_k,
        output,
        datadir
    );

    // Solve dI_k/dz = AI_k
    helium.simulate(
        rk_steps, // Number of steps for Runge-Kutta method
        L_cell, // Length of doping cell
        doping_pressure, // Pressure in mbar
        trajectory, // Flag to save trajectory
        input, // Input file name
        I_k_matrix, // final condition would be saved in it
        N_k_matrix // final condition would be saved in it
    );

    // Write output to file
    // Header k, y_final
    std::string k_name = std::format("{}/dist_k.txt", output);
    std::string nk_name = std::format("{}/dist_Nk.txt", output);
    std::ofstream k_file(k_name);
    std::ofstream nk_file(nk_name);

    std::print(k_file, "k\t");
    std::print(nk_file, "droplet_size\t");
    for (std::size_t n = 0; n < he_numbers.size(); n++) {
        std::print(k_file, "{}\t", he_numbers[n]);
        std::print(nk_file, "{}\t", he_numbers[n]);
    }
    std::print(k_file, "\n");
    std::print(nk_file, "\n");

    for (std::size_t k = 0; k < max_k; k++) {
        std::print(k_file, "{}\t", k);
        for (std::size_t n = 0; n < he_numbers.size(); n++) {
            std::print(k_file, "{}\t", I_k_matrix(k, n));
        }
        std::print(k_file, "\n");
    }

    for (int nk = 0; nk < N_k_matrix.rows(); nk++) {
        std::print(nk_file, "{}\t", dist_step*nk);
        for (std::size_t n = 0; n < he_numbers.size(); n++) {
            std::print(nk_file, "{}\t", N_k_matrix(nk, n));
        }
        std::print(nk_file, "\n");
    }

    // Lets calculate the mean dopant size for distributions
    std::string result_filename = std::format("{}/result.txt", output);
    std::ofstream result(result_filename);
    std::println(result, "Type\tN0\tMean\tMode\tStdDev");
    Eigen::VectorXd dopant_sizes = Eigen::VectorXd::LinSpaced(I_k_matrix.rows(), 0, I_k_matrix.rows());
    Eigen::VectorXd droplet_sizes = Eigen::VectorXd::LinSpaced(N_k_matrix.rows(), 0, dist_step*N_k_matrix.rows());
    for (std::size_t n = 0; n < he_numbers.size(); n++) {
        // Dopant distribution
        auto dopant_stats =
            compute_distribution_stats(dopant_sizes, I_k_matrix.col(n));


        std::println(
            result,
            "Dopant\t{}\t{}\t{}\t{}",
            he_numbers[n],
            dopant_stats.mean,
            dopant_stats.mode,
            dopant_stats.stddev
        );

        // Droplet distribution
        N_k_matrix.row(0).setZero();

        auto droplet_stats =
            compute_distribution_stats(droplet_sizes, N_k_matrix.col(n));

        std::println(
            result,
            "Droplet\t{}\t{}\t{}\t{}",
            he_numbers[n],
            droplet_stats.mean,
            droplet_stats.mode,
            droplet_stats.stddev
        );
    }

    return 0;
}
