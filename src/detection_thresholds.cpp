#include "detection_thresholds.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

constexpr float kUnknownClassThreshold = 0.25f;

std::string trim(const std::string& value)
{
    const std::string whitespace = " \t\r\n";
    const std::string::size_type begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return std::string();
    }

    const std::string::size_type end = value.find_last_not_of(whitespace);
    return value.substr(begin, end - begin + 1);
}

int classIdForName(const std::string& name)
{
    for (int class_id = 0; class_id < kDetectionClassCount; ++class_id) {
        if (name == detectionClassName(class_id)) {
            return class_id;
        }
    }
    return -1;
}

bool parseThreshold(const std::string& text, float* threshold)
{
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || errno == ERANGE) {
        return false;
    }

    while (*end == ' ' || *end == '\t' || *end == '\r') {
        ++end;
    }
    if (*end != '\0' || !std::isfinite(parsed)
        || parsed < 0.0f || parsed > 1.0f) {
        return false;
    }

    *threshold = parsed;
    return true;
}

}  // namespace

DetectionThresholds::DetectionThresholds()
    : values_{{0.25f, 0.50f, 0.50f, 0.50f, 0.50f}}
{
}

float DetectionThresholds::forClass(int class_id) const
{
    if (class_id < 0 || class_id >= kDetectionClassCount) {
        return kUnknownClassThreshold;
    }
    return values_[static_cast<std::size_t>(class_id)];
}

bool DetectionThresholds::setForClass(int class_id, float threshold)
{
    if (class_id < 0 || class_id >= kDetectionClassCount
        || !std::isfinite(threshold)
        || threshold < 0.0f || threshold > 1.0f) {
        return false;
    }
    values_[static_cast<std::size_t>(class_id)] = threshold;
    return true;
}

const char* detectionClassName(int class_id)
{
    static const char* names[kDetectionClassCount] = {
        "insulator", "crack", "pollution", "flashover", "broken"};
    if (class_id < 0 || class_id >= kDetectionClassCount) {
        return "unknown";
    }
    return names[class_id];
}

ThresholdConfigLoadResult loadDetectionThresholds(const std::string& path)
{
    ThresholdConfigLoadResult result;
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        result.warnings.push_back(
            "cannot open threshold configuration '" + path
            + "'; using built-in defaults");
        return result;
    }

    bool in_threshold_section = false;
    bool threshold_section_seen = false;
    std::array<bool, kDetectionClassCount> threshold_seen{{false, false, false, false, false}};
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string cleaned = trim(line);
        if (cleaned.empty() || cleaned[0] == '#' || cleaned[0] == ';') {
            continue;
        }

        if (cleaned.front() == '[' && cleaned.back() == ']') {
            in_threshold_section =
                trim(cleaned.substr(1, cleaned.size() - 2))
                == "ConfidenceThresholds";
            threshold_section_seen = threshold_section_seen || in_threshold_section;
            continue;
        }
        if (!in_threshold_section) {
            continue;
        }

        const std::string::size_type equals = cleaned.find('=');
        if (equals == std::string::npos) {
            result.warnings.push_back(
                "invalid threshold entry at line "
                + std::to_string(line_number));
            continue;
        }

        const std::string key = trim(cleaned.substr(0, equals));
        const std::string value = trim(cleaned.substr(equals + 1));
        const int class_id = classIdForName(key);
        if (class_id < 0) {
            result.warnings.push_back(
                "unknown threshold key '" + key + "' at line "
                + std::to_string(line_number));
            continue;
        }
        threshold_seen[static_cast<std::size_t>(class_id)] = true;

        float threshold = 0.0f;
        if (!parseThreshold(value, &threshold)) {
            result.warnings.push_back(
                "invalid threshold for '" + key + "' at line "
                + std::to_string(line_number) + "; using default");
            continue;
        }
        result.thresholds.setForClass(class_id, threshold);
    }

    if (!threshold_section_seen) {
        result.warnings.push_back(
            "missing [ConfidenceThresholds] section; using built-in defaults");
    }
    for (int class_id = 0; class_id < kDetectionClassCount; ++class_id) {
        if (!threshold_seen[static_cast<std::size_t>(class_id)]) {
            result.warnings.push_back(
                "missing threshold for '"
                + std::string(detectionClassName(class_id))
                + "'; using default");
        }
    }

    return result;
}

bool passesConfidenceThreshold(
    int class_id,
    float confidence,
    const DetectionThresholds& thresholds)
{
    return confidence >= thresholds.forClass(class_id);
}

std::string defaultThresholdConfigPath(const std::string& executable_path)
{
    const std::string::size_type separator =
        executable_path.find_last_of("/\\");
    if (separator == std::string::npos) {
        return "detection_thresholds.ini";
    }
    return executable_path.substr(0, separator + 1)
        + "detection_thresholds.ini";
}
