#ifndef DETECTION_THRESHOLDS_HPP
#define DETECTION_THRESHOLDS_HPP

#include <array>
#include <string>
#include <vector>

constexpr int kDetectionClassCount = 5;

struct DetectionThresholds
{
    DetectionThresholds();

    float forClass(int class_id) const;
    bool setForClass(int class_id, float threshold);

private:
    std::array<float, kDetectionClassCount> values_;
};

struct ThresholdConfigLoadResult
{
    DetectionThresholds thresholds;
    std::vector<std::string> warnings;
};

ThresholdConfigLoadResult loadDetectionThresholds(const std::string& path);
bool passesConfidenceThreshold(
    int class_id,
    float confidence,
    const DetectionThresholds& thresholds);
std::string defaultThresholdConfigPath(const std::string& executable_path);
const char* detectionClassName(int class_id);

#endif
