#include "tcp_jpeg_publisher.hpp"
#include "video_publisher_pipeline.hpp"
#include <gst/app/gstappsrc.h>
#include <cstring>
#include <exception>

TcpJpegPublisher::~TcpJpegPublisher() { close(); }

bool TcpJpegPublisher::open(int width, int height, int fps, int quality,
                            const std::string& host, int port)
{
    close();
    try {
        const std::string description =
            buildTcpJpegPublisherPipeline(quality, host, port);
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(description.c_str(), &error);
        if (error) {
            lastError_ = error->message;
            g_error_free(error);
        } 
    } catch (const std::exception& error) {
        lastError_ = error.what();
        return false;
    }
    if (!pipeline_ || !lastError_.empty()) { close(); return false; }
    appSrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "video_source");
    //遍历 bin（也就是整条流水线）内部所有子插件，找到 name="video_source" 的那一个，返回它的指针。
    if (!appSrc_) { lastError_ = "cannot find appsrc video_source"; close(); return false; }
    GstCaps* caps = gst_caps_new_simple(
        "video/x-raw", "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, width, "height", G_TYPE_INT, height,
        "framerate", GST_TYPE_FRACTION, fps, 1, nullptr);
    gst_app_src_set_caps(GST_APP_SRC(appSrc_), caps);
    gst_caps_unref(caps);
    gst_app_src_set_stream_type(GST_APP_SRC(appSrc_), GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_max_bytes(GST_APP_SRC(appSrc_),
        static_cast<guint64>(width) * height * 3U * 2U);
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        lastError_ = "failed to start TCP-JPEG publisher";
        close();
        return false;
    }
    width_ = width; height_ = height; lastError_.clear();
    return true;
}

bool TcpJpegPublisher::push(const cv::Mat& frame)
{
    if (!appSrc_ || frame.empty() || frame.type() != CV_8UC3 ||
        frame.cols != width_ || frame.rows != height_) {
        lastError_ = "invalid TCP-JPEG frame";
        return false;
    }
    const cv::Mat image = frame.isContinuous() ? frame : frame.clone();
    const gsize size = static_cast<gsize>(image.total() * image.elemSize());
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo info{};
    if (!buffer || !gst_buffer_map(buffer, &info, GST_MAP_WRITE)) {
        if (buffer) gst_buffer_unref(buffer);
        lastError_ = "failed to allocate TCP-JPEG buffer";
        return false;
    }
    std::memcpy(info.data, image.data, size);
    gst_buffer_unmap(buffer, &info);
    const GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appSrc_), buffer);
    if (flow != GST_FLOW_OK) {
        lastError_ = std::string("TCP-JPEG push failed: ") + gst_flow_get_name(flow);
        return false;
    }
    lastError_.clear();
    return true;
}

void TcpJpegPublisher::close()
{
    if (appSrc_) gst_app_src_end_of_stream(GST_APP_SRC(appSrc_));
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (appSrc_) { gst_object_unref(appSrc_); appSrc_ = nullptr; }
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
    width_ = height_ = 0;
}
