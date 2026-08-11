#ifndef LETTERBOX_GEOMETRY_HPP
#define LETTERBOX_GEOMETRY_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

struct FloatRect { float x; float y; float width; float height; };

struct LetterboxGeometry {
    int sourceWidth;
    int sourceHeight;
    int modelWidth;
    int modelHeight;
    int resizedWidth;
    int resizedHeight;
    int padLeft;
    int padTop;
    float scale;
};

inline LetterboxGeometry makeLetterboxGeometry(
    int sourceWidth, int sourceHeight, int modelWidth, int modelHeight)
{
    if (sourceWidth <= 0 || sourceHeight <= 0 ||
        modelWidth <= 0 || modelHeight <= 0) {
        throw std::invalid_argument("invalid letterbox dimensions");
    }
    LetterboxGeometry g{};
    g.sourceWidth = sourceWidth;
    g.sourceHeight = sourceHeight;
    g.modelWidth = modelWidth;
    g.modelHeight = modelHeight;
    g.scale = std::min(static_cast<float>(modelWidth) / sourceWidth,
                       static_cast<float>(modelHeight) / sourceHeight);
    g.resizedWidth = static_cast<int>(std::round(sourceWidth * g.scale));
    g.resizedHeight = static_cast<int>(std::round(sourceHeight * g.scale));
    g.padLeft = (modelWidth - g.resizedWidth) / 2;
    g.padTop = (modelHeight - g.resizedHeight) / 2;
    return g;
}

inline FloatRect mapModelRectToSource(
    const FloatRect& box, const LetterboxGeometry& g)
{
    const float contentLeft = static_cast<float>(g.padLeft);
    const float contentTop = static_cast<float>(g.padTop);
    const float contentRight = contentLeft + g.resizedWidth;
    const float contentBottom = contentTop + g.resizedHeight;
    const float left = std::max(contentLeft, box.x);
    const float top = std::max(contentTop, box.y);
    const float right = std::min(contentRight, box.x + box.width);
    const float bottom = std::min(contentBottom, box.y + box.height);
    if (right <= left || bottom <= top) return FloatRect{0, 0, 0, 0};
    const float x = std::max(0.0f, (left - contentLeft) / g.scale);
    const float y = std::max(0.0f, (top - contentTop) / g.scale);
    const float r = std::min(static_cast<float>(g.sourceWidth),
                             (right - contentLeft) / g.scale);
    const float b = std::min(static_cast<float>(g.sourceHeight),
                             (bottom - contentTop) / g.scale);
    return FloatRect{x, y, r - x, b - y};
}

#endif
