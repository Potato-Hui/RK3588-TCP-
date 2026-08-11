#ifndef VIDEO_PUBLISHER_PIPELINE_HPP
#define VIDEO_PUBLISHER_PIPELINE_HPP
#include <stdexcept>
#include <string>

inline std::string buildTcpJpegPublisherPipeline(
    int quality, const std::string& host, int port)
{
    if (quality < 1 || quality > 100 || host.empty() || port < 1 || port > 65535)
        throw std::invalid_argument("invalid TCP-JPEG settings");
    return "appsrc name=video_source is-live=true block=false format=time "
           "do-timestamp=true ! queue max-size-buffers=1 leaky=downstream ! "
           "videoconvert ! jpegenc quality=" + std::to_string(quality) +
           " ! tcpserversink host=" + host + " port=" +
           std::to_string(port) + " sync=false";
}
#endif
