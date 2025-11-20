#pragma once

#include <string>

enum struct DistributionType {
    NONE,
    LOGNORMAL,
    EXPONENTIAL,
};

inline static DistributionType convert_type(std::string type) {
    if (type == "NONE") return DistributionType::NONE;
    if (type == "LOGNORMAL") return DistributionType::LOGNORMAL;
    if (type == "EXPONENTIAL") return DistributionType::EXPONENTIAL;
    return DistributionType::NONE; // Return NONE if type is unrecognized
}

// --- Base Class ---
class Distribution {
public:
    virtual double pdf(double x) const = 0;
    virtual double mean() const = 0;
    virtual ~Distribution() = default;
};
