#include "dopant.hpp"
#include "constants.hpp"
#include <yaml-cpp/yaml.h>
#include <format>
#include <stdexcept>

Dopant::Dopant(const std::string name, const std::string datadir)
    : m_name(std::move(name)),
      m_temprature(0.0),
      m_mass(0.0),
      m_velocity(0.0),
      m_e_He_X(0.0),
      m_e_X_X(0.0)
{
    // File name = "<name>.yaml"
    std::string filename = std::format("{}/{}.yaml", datadir, m_name);

    // Load YAML file
    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: Cannot load YAML file " + filename +
                                 ": " + e.what());
    }

    // Required fields
    try {
        m_temprature = config["temprature"].as<double>();
        m_mass       = config["mass"].as<double>();
        m_velocity   = config["velocity"].as<double>();
        m_e_He_X     = config["E_He_X"].as<double>()*constants::e;
        m_e_X_X      = config["E_X_X"].as<double>()*constants::e;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: Missing or invalid fields in " +
                                 filename + ": " + e.what());
    }
}

Dopant::~Dopant() = default;
