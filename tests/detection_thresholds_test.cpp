#include "detection_thresholds.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

namespace {

bool nearlyEqual(float left, float right)
{
    return std::fabs(left - right) < 0.0001f;
}

void writeTextFile(const std::string& path, const std::string& contents)
{
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    assert(output.is_open());
    output << contents;
    assert(output.good());
}

}  // namespace

int main()
{
    const std::string valid_path = "detection_thresholds_valid_test.ini";
    writeTextFile(
        valid_path,
        "[ConfidenceThresholds]\n"
        "insulator=0.15\n"
        "crack=0.35\n"
        "pollution=0.45\n"
        "flashover=0.55\n"
        "broken=0.65\n");

    const ThresholdConfigLoadResult valid =
        loadDetectionThresholds(valid_path);
    assert(valid.warnings.empty());
    assert(nearlyEqual(valid.thresholds.forClass(0), 0.15f));
    assert(nearlyEqual(valid.thresholds.forClass(1), 0.35f));
    assert(nearlyEqual(valid.thresholds.forClass(2), 0.45f));
    assert(nearlyEqual(valid.thresholds.forClass(3), 0.55f));
    assert(nearlyEqual(valid.thresholds.forClass(4), 0.65f));
    assert(passesConfidenceThreshold(1, 0.35f, valid.thresholds));
    assert(!passesConfidenceThreshold(1, 0.349f, valid.thresholds));
    std::remove(valid_path.c_str());

    const ThresholdConfigLoadResult missing =
        loadDetectionThresholds("detection_thresholds_missing_test.ini");
    assert(!missing.warnings.empty());
    assert(nearlyEqual(missing.thresholds.forClass(0), 0.25f));
    assert(nearlyEqual(missing.thresholds.forClass(1), 0.50f));
    assert(nearlyEqual(missing.thresholds.forClass(2), 0.50f));
    assert(nearlyEqual(missing.thresholds.forClass(3), 0.50f));
    assert(nearlyEqual(missing.thresholds.forClass(4), 0.50f));
    assert(nearlyEqual(missing.thresholds.forClass(99), 0.25f));

    const std::string invalid_path = "detection_thresholds_invalid_test.ini";
    writeTextFile(
        invalid_path,
        "[ConfidenceThresholds]\n"
        "insulator=0.20\n"
        "crack=not-a-number\n"
        "pollution=-0.1\n"
        "flashover=1.1\n"
        "broken=nan\n"
        "unknown=0.9\n");

    const ThresholdConfigLoadResult invalid =
        loadDetectionThresholds(invalid_path);
    assert(invalid.warnings.size() == 5);
    assert(nearlyEqual(invalid.thresholds.forClass(0), 0.20f));
    assert(nearlyEqual(invalid.thresholds.forClass(1), 0.50f));
    assert(nearlyEqual(invalid.thresholds.forClass(2), 0.50f));
    assert(nearlyEqual(invalid.thresholds.forClass(3), 0.50f));
    assert(nearlyEqual(invalid.thresholds.forClass(4), 0.50f));
    std::remove(invalid_path.c_str());

    const std::string partial_path = "detection_thresholds_partial_test.ini";
    writeTextFile(
        partial_path,
        "[ConfidenceThresholds]\n"
        "broken=0.72\n");

    const ThresholdConfigLoadResult partial =
        loadDetectionThresholds(partial_path);
    assert(partial.warnings.size() == 4);
    assert(nearlyEqual(partial.thresholds.forClass(1), 0.50f));
    assert(nearlyEqual(partial.thresholds.forClass(4), 0.72f));
    std::remove(partial_path.c_str());

    assert(defaultThresholdConfigPath("/opt/demo/rknn_yolov8_demo") ==
           "/opt/demo/detection_thresholds.ini");
    assert(defaultThresholdConfigPath("rknn_yolov8_demo") ==
           "detection_thresholds.ini");
}
