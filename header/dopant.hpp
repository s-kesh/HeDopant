#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <format>

/*
 * Dopant class
 * Represents the properties of dopant in He droplet
 * Fields:
 *   name: Name of the dopant
 *   temperature: Temperature of the dopant
 *   mass: Mass of the dopant
 *   velocity: Velocity of the dopant
 *   e_Int: Internal energy
 *   e_He_X: Energy of He-dopant interaction
 *   e_X_X: Energy of dopant-dopant interaction
 */
class Dopant {
public:
    /*
    * Initialize dopant properties from YAML file
    * Input:
    *   name: Name of the dopant
    *   datadir: Directory containing the YAML file
    * It reads name.yaml file from the datadir and initializes the dopant properties
    */
    Dopant(const std::string name, const std::string datadir)
        : m_name(name), m_temperature(0), m_mass(0), m_velocity(0), m_e_He_X(0), m_e_X_X(0) {
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
            m_temperature = config["temperature"].as<double>();
            m_mass       = config["mass"].as<double>();
            m_velocity   = config["velocity"].as<double>();
            m_e_Int      = config["E_Int"].as<double>();
            m_e_He_X     = config["E_He_X"].as<double>();
            m_e_X_X      = config["E_X_X"].as<double>();
        } catch (const std::exception& e) {
            throw std::runtime_error("Error: Missing or invalid fields in " +
                                        filename + ": " + e.what());
        }
    }

    ~Dopant() = default;

    // Getters
    const std::string& name() const noexcept { return m_name; }
    double temperature() const noexcept { return m_temperature; }
    double mass() const noexcept { return m_mass; }
    double velocity() const noexcept { return m_velocity; }
    double e_Int() const noexcept { return m_e_Int;}
    double e_He_X() const noexcept { return m_e_He_X; }
    double e_X_X() const noexcept { return m_e_X_X; }

private:
    std::string m_name;
    double m_temperature;
    double m_mass;
    double m_velocity;
    double m_e_Int;
    double m_e_He_X;
    double m_e_X_X;
};
