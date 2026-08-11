#ifndef CAMERA_PIPELINE_HPP
#define CAMERA_PIPELINE_HPP
#include <stdexcept>
#include <string>

inline std::string buildCameraPipelineDescription(
    const std::string& device, int width, int height, int fps)
{
    if (device.empty() || width <= 0 || height <= 0 || fps <= 0)
        throw std::invalid_argument("invalid camera settings");
    return "v4l2src device=" + device + " ! image/jpeg,width=" +
        std::to_string(width) + ",height=" + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! jpegdec ! videoconvert ! "
        "video/x-raw,format=BGR,width=" + std::to_string(width) +
        ",height=" + std::to_string(height) + " ! "
        "appsink name=camera_sink sync=false drop=true max-buffers=1";
}
#endif
