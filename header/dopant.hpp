#pragma once

#include <string>

class Dopant {
public:
    Dopant(std::string name);
    ~Dopant();

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
