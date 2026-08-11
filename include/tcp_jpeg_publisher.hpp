#ifndef TCP_JPEG_PUBLISHER_HPP
#define TCP_JPEG_PUBLISHER_HPP
#include <gst/gst.h>
#include <opencv2/core/core.hpp>
#include <string>

class TcpJpegPublisher {
public:
    ~TcpJpegPublisher();
    bool open(int width, int height, int fps, int quality,
              const std::string& host, int port);
    bool push(const cv::Mat& frame);
    void close();
    const std::string& lastError() const { return lastError_; }
private:
    GstElement* pipeline_ = nullptr;
    GstElement* appSrc_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    std::string lastError_;
};
#endif
