#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <format>

#include "constants.hpp"

class Dopant {
public:
    Dopant(const std::string name, const std::string datadir)
        : m_name(name), m_temprature(0), m_mass(0), m_velocity(0), m_e_He_X(0), m_e_X_X(0) {
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

    ~Dopant() = default;

    // Getters
    const std::string& name() const noexcept { return m_name; }
    double temprature() const noexcept { return m_temprature; }
    double mass() const noexcept { return m_mass; }
    double velocity() const noexcept { return m_velocity; }
    double e_He_X() const noexcept { return m_e_He_X; }
    double e_X_X() const noexcept { return m_e_X_X; }

private:
    std::string m_name;
    double m_temprature;
    double m_mass;
    double m_velocity;
    double m_e_He_X;
    double m_e_X_X;
};
