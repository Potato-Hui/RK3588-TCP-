#include "letterbox_geometry.hpp"

#include <cassert>
#include <cmath>

int main()
{
    const LetterboxGeometry geometry =
        makeLetterboxGeometry(1280, 720, 640, 640);
    assert(std::fabs(geometry.scale - 0.5f) < 0.0001f);
    assert(geometry.resizedWidth == 640);
    assert(geometry.resizedHeight == 360);
    assert(geometry.padLeft == 0);
    assert(geometry.padTop == 140);

    const FloatRect full = mapModelRectToSource(
        FloatRect{0.0f, 140.0f, 640.0f, 360.0f}, geometry);
    assert(std::fabs(full.x) < 0.01f);
    assert(std::fabs(full.y) < 0.01f);
    assert(std::fabs(full.width - 1280.0f) < 0.01f);
    assert(std::fabs(full.height - 720.0f) < 0.01f);

    const FloatRect detection = mapModelRectToSource(
        FloatRect{160.0f, 230.0f, 320.0f, 180.0f}, geometry);
    assert(std::fabs(detection.x - 320.0f) < 0.01f);
    assert(std::fabs(detection.y - 180.0f) < 0.01f);
    assert(std::fabs(detection.width - 640.0f) < 0.01f);
    assert(std::fabs(detection.height - 360.0f) < 0.01f);

    const FloatRect padding = mapModelRectToSource(
        FloatRect{10.0f, 10.0f, 100.0f, 50.0f}, geometry);
    assert(padding.width == 0.0f);
    assert(padding.height == 0.0f);
}
