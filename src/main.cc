#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "camera_pipeline.hpp"
#include "monitor_protocol.hpp"
#include "tcp_jpeg_publisher.hpp"

namespace {

constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
constexpr int kDefaultFps = 30;
constexpr int kDefaultThreadNum = 12;
constexpr const char* kDefaultCamera = "/dev/video41";
constexpr int kTcpPort = 5000;
constexpr const char* kRecordsRoot = "/data/records/";
volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) { gStopRequested = 1; }
void writeProtocolLine(const std::string& line)
{
    std::fprintf(stdout, "%s\n", line.c_str());
    std::fflush(stdout);
}

struct SnapshotRequest {
    std::string requestId;
    std::string outputDir;
};

bool readJsonString(const std::string& json, const char* key, std::string* value)
{
    const std::string name = std::string("\"") + key + "\"";
    const std::size_t key_pos = json.find(name);
    if (key_pos == std::string::npos) return false;
    const std::size_t colon = json.find(':', key_pos + name.size());
    const std::size_t first_quote = json.find('\"', colon + 1);
    if (colon == std::string::npos || first_quote == std::string::npos) return false;
    const std::size_t second_quote = json.find('\"', first_quote + 1);
    if (second_quote == std::string::npos) return false;
    *value = json.substr(first_quote + 1, second_quote - first_quote - 1);
    return true;
}

bool isSafeSnapshotRequest(const SnapshotRequest& request)
{
    if (request.requestId.empty()
        || request.outputDir != std::string(kRecordsRoot) + request.requestId) {
        return false;
    }
    for (const unsigned char ch : request.requestId) {
        if (!std::isalnum(ch) && ch != '-') return false;
    }
    return true;
}

bool pollSnapshotRequest(std::string* input_buffer, SnapshotRequest* request)
{
    char buffer[512];
    for (;;) {
        const ssize_t count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
        if (count <= 0) break;
        input_buffer->append(buffer, static_cast<std::size_t>(count));
    }

    const std::size_t line_end = input_buffer->find('\n');
    if (line_end == std::string::npos) return false;
    std::string line = input_buffer->substr(0, line_end);
    input_buffer->erase(0, line_end + 1);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.compare(0, 10, "@snapshot ") != 0) return false;

    return readJsonString(line.substr(10), "request_id", &request->requestId)
        && readJsonString(line.substr(10), "output_dir", &request->outputDir)
        && isSafeSnapshotRequest(*request);
}

void writeSnapshotReply(const SnapshotRequest& request,
                        const char* state,
                        const std::string& message = std::string())
{
    std::ostringstream out;
    out << "@snapshot {\"request_id\":\"" << jsonEscape(request.requestId)
        << "\",\"state\":\"" << state << "\"";
    if (std::string(state) == "ready") {
        out << ",\"record_dir\":\"" << jsonEscape(request.outputDir) << "\"";
    } else {
        out << ",\"message\":\"" << jsonEscape(message) << "\"";
    }
    out << "}";
    writeProtocolLine(out.str());
}

bool mapClass(int model_class_id, int* class_id, const char** class_name)
{
    switch (model_class_id) {
    case 0: *class_id = 0; *class_name = "insulator"; return true;
    case 1: *class_id = 1; *class_name = "crack"; return true;
    case 2: *class_id = 2; *class_name = "pollution"; return true;
    case 3: *class_id = 3; *class_name = "flashover"; return true;
    case 4: *class_id = 4; *class_name = "broken"; return true;
    default: return false;
    }
}

void removeTemporaryRecord(const std::string& directory, int mask_count)
{
    std::remove((directory + "/image.jpg").c_str());
    std::remove((directory + "/metadata.json").c_str());
    for (int index = 1; index <= mask_count; ++index) {
        char name[64];
        std::snprintf(name, sizeof(name), "/masks/instance_%02d.png", index);
        std::remove((directory + name).c_str());
    }
    ::rmdir((directory + "/masks").c_str());
    ::rmdir(directory.c_str());
}

bool saveSnapshot(const SnapshotRequest& request,
                  const InferenceResult& result,
                  std::string* error)
{
    if (!result.succeeded || result.originalFrame.empty()) {
        *error = "当前帧推理未成功，无法保存快照";
        return false;
    }

    const std::string temporary_dir = request.outputDir + ".tmp";
    if (::mkdir(temporary_dir.c_str(), 0755) != 0
        || ::mkdir((temporary_dir + "/masks").c_str(), 0755) != 0) {
        *error = "无法创建快照目录";
        removeTemporaryRecord(temporary_dir, 0);
        return false;
    }

    int saved_masks = 0;
    const std::string image_path = temporary_dir + "/image.jpg";
    if (!cv::imwrite(image_path, result.originalFrame,
                     std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 95})) {
        *error = "无法保存原始图片";
        removeTemporaryRecord(temporary_dir, saved_masks);
        return false;
    }

    std::ofstream metadata((temporary_dir + "/metadata.json").c_str());
    if (!metadata) {
        *error = "无法写入 metadata.json";
        removeTemporaryRecord(temporary_dir, saved_masks);
        return false;
    }

    metadata << "{\n  \"image_width\": " << result.originalFrame.cols
             << ",\n  \"image_height\": " << result.originalFrame.rows
             << ",\n  \"instances\": [";
    bool first = true;
    int instance_index = 0;
    for (const SegmentationInstance& instance : result.instances) {
        ++instance_index;
        int class_id = -1;
        const char* class_name = nullptr;
        if (!mapClass(instance.classId, &class_id, &class_name)) {
            *error = "实例 " + std::to_string(instance_index)
                + " 的模型类别 " + std::to_string(instance.classId)
                + " 未配置到快照接口";
            metadata.close();
            removeTemporaryRecord(temporary_dir, saved_masks);
            return false;
        }
        if (instance.mask.empty() || instance.mask.type() != CV_8UC1
            || instance.bbox.x < 0 || instance.bbox.y < 0
            || instance.bbox.width <= 0 || instance.bbox.height <= 0
            || instance.bbox.x + instance.bbox.width > result.originalFrame.cols
            || instance.bbox.y + instance.bbox.height > result.originalFrame.rows
            || instance.mask.size() != instance.bbox.size()) {
            std::ostringstream detail;
            detail << "实例 " << instance_index << " 的 bbox 或 mask 无效"
                   << " (bbox=" << instance.bbox.x << ',' << instance.bbox.y
                   << ',' << instance.bbox.width << ',' << instance.bbox.height
                   << ", mask=" << instance.mask.cols << 'x' << instance.mask.rows
                   << ", type=" << instance.mask.type() << ')';
            *error = detail.str();
            metadata.close();
            removeTemporaryRecord(temporary_dir, saved_masks);
            return false;
        }

        ++saved_masks;
        char mask_file[64];
        std::snprintf(mask_file, sizeof(mask_file), "masks/instance_%02d.png", saved_masks);
        if (!cv::imwrite(temporary_dir + "/" + mask_file, instance.mask)) {
            *error = "无法保存实例 mask";
            metadata.close();
            removeTemporaryRecord(temporary_dir, saved_masks);
            return false;
        }

        metadata << (first ? "\n" : ",\n")
                 << "    {\"instance_id\": " << saved_masks
                 << ", \"class_id\": " << class_id
                 << ", \"class_name\": \"" << class_name << "\""
                 << ", \"confidence\": " << std::fixed << std::setprecision(6)
                 << instance.confidence
                 << ", \"bbox_xyxy\": [" << instance.bbox.x << ", "
                 << instance.bbox.y << ", "
                 << instance.bbox.x + instance.bbox.width << ", "
                 << instance.bbox.y + instance.bbox.height << "]"
                 << ", \"mask_file\": \"" << mask_file << "\"}";
        first = false;
    }
    metadata << "\n  ]\n}\n";
    metadata.close();
    if (!metadata || std::rename(temporary_dir.c_str(), request.outputDir.c_str()) != 0) {
        *error = "无法发布快照记录";
        removeTemporaryRecord(temporary_dir, saved_masks);
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// GStreamer 摄像头：v4l2src -> NV12 -> videoconvert -> BGR -> appsink
// -----------------------------------------------------------------------------
class GstCamera {
public:
    GstCamera() = default;
    ~GstCamera() { close(); }

    GstCamera(const GstCamera&) = delete;
    GstCamera& operator=(const GstCamera&) = delete;

    bool open(const std::string& device, int width, int height, int fps) {
        close();

        width_ = width;
        height_ = height;
        fps_ = fps;

        const std::string pipeline_desc =
            buildCameraPipelineDescription(device, width, height, fps);

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_desc.c_str(), &error);

        if (error != nullptr) {
            std::fprintf(stderr, "GStreamer camera pipeline error: %s\n",
                         error->message);
            g_error_free(error);
        }

        if (pipeline_ == nullptr) {
            return false;
        }

        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "camera_sink");
        if (appsink_ == nullptr) {
            std::fprintf(stderr, "Cannot find appsink: camera_sink\n");
            close();
            return false;
        }

        gst_app_sink_set_emit_signals(GST_APP_SINK(appsink_), FALSE);
        gst_app_sink_set_drop(GST_APP_SINK(appsink_), TRUE);
        gst_app_sink_set_max_buffers(GST_APP_SINK(appsink_), 1);

        const GstStateChangeReturn ret =
            gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::fprintf(stderr, "Failed to start camera pipeline\n");
            close();
            return false;
        }

        return true;
    }

    bool read(cv::Mat& frame, GstClockTime timeout = GST_SECOND) {
        if (appsink_ == nullptr) {
            return false;
        }

        GstSample* sample =
            gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), timeout);

        if (sample == nullptr) {
            return false;
        }

        GstCaps* caps = gst_sample_get_caps(sample);
        GstBuffer* buffer = gst_sample_get_buffer(sample);

        if (caps == nullptr || buffer == nullptr) {
            gst_sample_unref(sample);
            return false;
        }

        GstStructure* structure = gst_caps_get_structure(caps, 0);
        int width = 0;
        int height = 0;

        if (!gst_structure_get_int(structure, "width", &width) ||
            !gst_structure_get_int(structure, "height", &height)) {
            gst_sample_unref(sample);
            return false;
        }

        GstMapInfo map_info{};
        if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return false;
        }

        const std::size_t expected =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 3U;

        bool ok = false;
        if (map_info.size >= expected) {
            cv::Mat wrapped(height, width, CV_8UC3, map_info.data);
            frame = wrapped.clone();
            ok = !frame.empty();
        } else {
            std::fprintf(stderr,
                         "Unexpected camera buffer size: got %zu, expected >= %zu\n",
                         static_cast<std::size_t>(map_info.size),
                         expected);
        }

        gst_buffer_unmap(buffer, &map_info);
        gst_sample_unref(sample);
        return ok;
    }

    void close() {
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }

        if (appsink_ != nullptr) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }

        if (pipeline_ != nullptr) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

private:
    GstElement* pipeline_ = nullptr;
    GstElement* appsink_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 0;
};

// -----------------------------------------------------------------------------
// GStreamer 显示：appsrc -> videoconvert -> autovideosink
// 不依赖 OpenCV highgui。
// -----------------------------------------------------------------------------
class GstDisplay {
public:
    GstDisplay() = default;
    ~GstDisplay() { close(); }

    GstDisplay(const GstDisplay&) = delete;
    GstDisplay& operator=(const GstDisplay&) = delete;

    bool open(int width, int height, int fps,
              const std::string& sink = "autovideosink") {
        close();

        width_ = width;
        height_ = height;
        fps_ = fps;
        frame_index_ = 0;

        const std::string pipeline_desc =
            "appsrc name=display_src is-live=true block=false format=time "
            "do-timestamp=false ! "
            "videoconvert ! " + sink + " sync=false";
                
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_desc.c_str(), &error);

        if (error != nullptr) {
            std::fprintf(stderr, "GStreamer display pipeline error: %s\n",
                         error->message);
            g_error_free(error);
        }

        if (pipeline_ == nullptr) {
            return false;
        }

        appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "display_src");
        if (appsrc_ == nullptr) {
            std::fprintf(stderr, "Cannot find appsrc: display_src\n");
            close();
            return false;
        }

        GstCaps* caps = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, width_,
            "height", G_TYPE_INT, height_,
            "framerate", GST_TYPE_FRACTION, fps_, 1,
            nullptr);

        gst_app_src_set_caps(GST_APP_SRC(appsrc_), caps);
        gst_caps_unref(caps);

        gst_app_src_set_stream_type(
            GST_APP_SRC(appsrc_), GST_APP_STREAM_TYPE_STREAM);
        gst_app_src_set_max_bytes(
            GST_APP_SRC(appsrc_),
            static_cast<guint64>(width_) *
            static_cast<guint64>(height_) * 3U * 2U);

        const GstStateChangeReturn ret =
            gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::fprintf(stderr, "Failed to start display pipeline\n");
            close();
            return false;
        }

        return true;
    }

    bool push(const cv::Mat& frame) {
        if (appsrc_ == nullptr || frame.empty()) {
            return false;
        }

        cv::Mat bgr;
        if (frame.type() != CV_8UC3) {
            std::fprintf(stderr, "Display frame must be CV_8UC3\n");
            return false;
        }

        if (frame.cols != width_ || frame.rows != height_) {
            cv::resize(frame, bgr, cv::Size(width_, height_));
        } else if (!frame.isContinuous()) {
            bgr = frame.clone();
        } else {
            bgr = frame;
        }

        const gsize size =
            static_cast<gsize>(bgr.total() * bgr.elemSize());

        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        if (buffer == nullptr) {
            return false;
        }

        GstMapInfo map_info{};
        if (!gst_buffer_map(buffer, &map_info, GST_MAP_WRITE)) {
            gst_buffer_unref(buffer);
            return false;
        }

        std::memcpy(map_info.data, bgr.data, size);
        gst_buffer_unmap(buffer, &map_info);

        const GstClockTime duration =
            gst_util_uint64_scale_int(1, GST_SECOND, fps_);

        GST_BUFFER_PTS(buffer) = frame_index_ * duration;
        GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(buffer) = duration;
        ++frame_index_;

        const GstFlowReturn flow =
            gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);

        return flow == GST_FLOW_OK;
    }

    void close() {
        if (appsrc_ != nullptr) {
            gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        }

        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }

        if (appsrc_ != nullptr) {
            gst_object_unref(appsrc_);
            appsrc_ = nullptr;
        }

        if (pipeline_ != nullptr) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

private:
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 0;
    guint64 frame_index_ = 0;
};

void drawHud(cv::Mat& frame,
             int thread_num,
             double fps_ema,
             int frame_number) {
    char hud[96];
    std::snprintf(hud, sizeof(hud),
                  "Threads: %d  FPS: %.1f  Frame: %d",
                  thread_num, fps_ema, frame_number);

    cv::putText(frame, hud, cv::Point(12, 34),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 0), 5);

    cv::putText(frame, hud, cv::Point(12, 34),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "Usage: %s <model.rknn> [camera_device] [thread_num]\n"
            "Example:\n"
            "  %s model/RK3588/best_yolov8_fp16.rknn /dev/video41 12\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_name = argv[1];
    const std::string camera_device =
        (argc >= 3) ? argv[2] : kDefaultCamera;

    int thread_num =
        (argc >= 4) ? std::atoi(argv[3]) : kDefaultThreadNum;

    if (thread_num <= 0) {
        thread_num = kDefaultThreadNum;
    }

    gst_init(&argc, &argv);
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    const int stdin_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (stdin_flags < 0
        || fcntl(STDIN_FILENO, F_SETFL, stdin_flags | O_NONBLOCK) != 0) {
        writeProtocolLine(makeErrorStatus(
            "stdin_setup_failed", "Failed to configure snapshot input"));
        std::fprintf(stderr, "Failed to configure non-blocking stdin\n");
        return EXIT_FAILURE;
    }

    rknnPool<rkYolov5s, cv::Mat, InferenceResult> test_pool(
        const_cast<char*>(model_name), thread_num);

    if (test_pool.init() != 0) {
        writeProtocolLine(makeErrorStatus("model_init_failed", "rknnPool init failed"));
        std::fprintf(stderr, "rknnPool init failed\n");
        return EXIT_FAILURE;
    }

    GstCamera camera;
    if (!camera.open(camera_device,
                     kDefaultWidth,
                     kDefaultHeight,
                     kDefaultFps)) {
        writeProtocolLine(makeErrorStatus(
            "camera_open_failed", "Failed to open camera"));
        std::fprintf(stderr,
                     "Failed to open camera: %s\n",
                     camera_device.c_str());
        return EXIT_FAILURE;
    }

    TcpJpegPublisher publisher;
    if (!publisher.open(kDefaultWidth, kDefaultHeight, kDefaultFps,
                        80, "127.0.0.1", kTcpPort)) {
        writeProtocolLine(makeErrorStatus("publisher_open_failed", publisher.lastError()));
        return EXIT_FAILURE;
    }

    timeval time_value{};
    gettimeofday(&time_value, nullptr);

    const auto start_time_ms =
        time_value.tv_sec * 1000LL +
        time_value.tv_usec / 1000LL;

    auto period_start_ms = start_time_ms;
    auto last_output_time = std::chrono::steady_clock::now();

    int submitted_frames = 0;
    int output_frames = 0;
    int period_output_frames = 0;
    double fps_ema = 0.0;
    std::string snapshot_input;
    std::deque<SnapshotRequest> pending_snapshots;

    std::printf(
        "Model: %s\nCamera: %s\nThreads: %d\nTCP-JPEG: 127.0.0.1:%d\n",
        model_name,
        camera_device.c_str(),
        thread_num,
        kTcpPort);
    std::fflush(stdout);
    writeProtocolLine(makeReadyStatus(kDefaultWidth, kDefaultHeight,
                                      kDefaultFps, kTcpPort));

    while (!gStopRequested) {
        SnapshotRequest request;
        while (pollSnapshotRequest(&snapshot_input, &request)) {
            if (pending_snapshots.empty()) {
                pending_snapshots.push_back(request);
            } else {
                writeSnapshotReply(request, "error", "上一张快照仍在生成");
            }
        }

        cv::Mat camera_frame;
        if (!camera.read(camera_frame)) {
            std::fprintf(stderr, "Camera frame read failed\n");
            continue;
        }

        if (test_pool.put(camera_frame) != 0) {
            std::fprintf(stderr, "rknnPool put failed\n");
            break;
        }

        ++submitted_frames;

        // 线程池预热完成后，每送入一帧取出一帧
        if (submitted_frames >= thread_num) {
            InferenceResult inference_result;
            if (test_pool.get(inference_result) != 0) {
                std::fprintf(stderr, "rknnPool get failed\n");
                break;
            }

            if (!pending_snapshots.empty()) {
                std::string error;
                const SnapshotRequest snapshot = pending_snapshots.front();
                pending_snapshots.pop_front();
                if (saveSnapshot(snapshot, inference_result, &error)) {
                    writeSnapshotReply(snapshot, "ready");
                } else {
                    writeSnapshotReply(snapshot, "error", error);
                }
            }

            cv::Mat result_frame = inference_result.annotatedFrame;

            ++output_frames;

            const auto now = std::chrono::steady_clock::now();
            
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    now - last_output_time).count();
            last_output_time = now;

            const double instant_fps =
                elapsed_ms > 0.0 ? 1000.0 / elapsed_ms : 0.0;

            fps_ema =
                (fps_ema <= 0.0)
                    ? instant_fps
                    : fps_ema * 0.9 + instant_fps * 0.1;

            drawHud(result_frame,
                    thread_num,
                    fps_ema,
                    output_frames);

            if (!publisher.push(result_frame)) {
                writeProtocolLine(makeErrorStatus("publisher_push_failed", publisher.lastError()));
                break;
            }

            std::printf(
                "[Frame] output=%d submitted=%d interval=%.3f ms fps=%.2f\n",
                output_frames,
                submitted_frames,
                elapsed_ms,
                fps_ema);

            gettimeofday(&time_value, nullptr);
            const auto now_ms =
                time_value.tv_sec * 1000LL +
                time_value.tv_usec / 1000LL;
            if (now_ms - period_start_ms >= 1000) {
                const double period_fps =
                    static_cast<double>(output_frames - period_output_frames) * 1000.0 /
                    static_cast<double>(now_ms - period_start_ms);
                std::printf(
                    "Pipeline FPS: %.3f\n",
                    period_fps);
                writeProtocolLine(makeMetricsStatus(
                    static_cast<double>(submitted_frames) * 1000.0 /
                        static_cast<double>(now_ms - start_time_ms),
                    period_fps,
                    elapsed_ms));

                period_start_ms = now_ms;
                period_output_frames = output_frames;
            }
        }
    }

    // 排空线程池中尚未取出的结果
    while (output_frames < submitted_frames) {
        InferenceResult inference_result;
        if (test_pool.get(inference_result) != 0) {
            break;
        }

        if (!pending_snapshots.empty()) {
            std::string error;
            const SnapshotRequest snapshot = pending_snapshots.front();
            pending_snapshots.pop_front();
            if (saveSnapshot(snapshot, inference_result, &error)) {
                writeSnapshotReply(snapshot, "ready");
            } else {
                writeSnapshotReply(snapshot, "error", error);
            }
        }

        cv::Mat result_frame = inference_result.annotatedFrame;

        ++output_frames;
        drawHud(result_frame,
                thread_num,
                fps_ema,
                output_frames);

        publisher.push(result_frame);
    }

    gettimeofday(&time_value, nullptr);
    const auto end_time_ms =
        time_value.tv_sec * 1000LL +
        time_value.tv_usec / 1000LL;

    const double average_fps =
        (end_time_ms > start_time_ms)
            ? static_cast<double>(output_frames) * 1000.0 /
                  static_cast<double>(end_time_ms - start_time_ms)
            : 0.0;

    std::printf(
        "Finished. Submitted=%d, Output=%d, Average FPS=%.3f\n",
        submitted_frames,
        output_frames,
        average_fps);

    publisher.close();
    camera.close();
    return EXIT_SUCCESS;
}

