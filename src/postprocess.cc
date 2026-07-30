#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "postprocess.h"
#include "rknn_api.h"
// ============================================================
// 动态标签列表
// ============================================================

static std::vector<std::string> g_labels;
static bool g_labels_loaded = false;

static bool load_labels(const char *label_path)
{
    g_labels.clear();

    std::ifstream file(label_path);
    if (!file.is_open())
    {
        fprintf(stderr,
                "Warning: cannot open labels file: %s\n",
                label_path);
        return false;
    }

    std::string line;

    while (std::getline(file, line))
    {
        // 删除 Windows 文本文件末尾可能存在的 \r
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (!line.empty())
        {
            g_labels.push_back(line);
        }
    }

    file.close();

    printf("Loaded %zu labels from %s\n",
           g_labels.size(),
           label_path);

    return !g_labels.empty();
}

static const char *class_name(int class_id)
{
    if (class_id >= 0 &&
        class_id < static_cast<int>(g_labels.size()))
    {
        return g_labels[class_id].c_str();
    }

    return "unknown";
}

// ============================================================
// 基础工具
// ============================================================

static inline float clamp_float(float value,
                                float min_value,
                                float max_value)
{
    if (value < min_value)
    {
        return min_value;
    }

    if (value > max_value)
    {
        return max_value;
    }

    return value;
}

static inline float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

static inline float dequantize_int8(int8_t value,
                                    int32_t zero_point,
                                    float scale)
{
    return (static_cast<float>(value) -
            static_cast<float>(zero_point)) *
           scale;
}

static inline float dequantize_uint8(uint8_t value,
                                     int32_t zero_point,
                                     float scale)
{
    return (static_cast<float>(value) -
            static_cast<float>(zero_point)) *
           scale;
}

/**
 * 根据 RKNN 输出类型读取一个元素。
 *
 * 支持：
 *   RKNN_TENSOR_INT8
 *   RKNN_TENSOR_UINT8
 *   RKNN_TENSOR_FLOAT32
 */
static bool read_tensor_value(const void *buffer,
                              const rknn_tensor_attr *attr,
                              size_t index,
                              float &value)
{
    if (buffer == nullptr || attr == nullptr)
    {
        return false;
    }

    switch (attr->type)
    {
        case RKNN_TENSOR_INT8:
        {
            const int8_t *data =
                static_cast<const int8_t *>(buffer);

            value = dequantize_int8(
                data[index],
                attr->zp,
                attr->scale);

            return true;
        }

        case RKNN_TENSOR_UINT8:
        {
            const uint8_t *data =
                static_cast<const uint8_t *>(buffer);

            value = dequantize_uint8(
                data[index],
                attr->zp,
                attr->scale);

            return true;
        }

        case RKNN_TENSOR_FLOAT16:
        {
            const uint16_t *data =
                static_cast<const uint16_t *>(buffer);

            uint16_t h = data[index];

            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp  = (h & 0x7C00) >> 10;
            uint32_t mant = h & 0x03FF;

            uint32_t f;

            if (exp == 0)
            {
                if (mant == 0)
                {
                    f = sign;
                }
                else
                {
                    exp = 127 - 15 + 1;

                    while ((mant & 0x0400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }

                    mant &= 0x03FF;

                    f = sign |
                        (exp << 23) |
                        (mant << 13);
                }
            }
            else if (exp == 31)
            {
                f = sign |
                    0x7F800000 |
                    (mant << 13);
            }
            else
            {
                f = sign |
                    ((exp + 127 - 15) << 23) |
                    (mant << 13);
            }


            memcpy(&value, &f, sizeof(float));

            return true;
        }
        case RKNN_TENSOR_FLOAT32:
        {
            const float *data =
                static_cast<const float *>(buffer);

            value = data[index];
            return true;
        }

        default:
        {
            fprintf(stderr,
                    "Unsupported RKNN output type: %d\n",
                    attr->type);
            return false;
        }
    }
}

// ============================================================
// 检测框和 NMS
// ============================================================

struct DetectionCandidate
{
    float x1;
    float y1;
    float x2;
    float y2;

    float score;
    int class_id;
};

static float calculate_iou(const DetectionCandidate &a,
                           const DetectionCandidate &b)
{
    const float inter_left =
        std::max(a.x1, b.x1);

    const float inter_top =
        std::max(a.y1, b.y1);

    const float inter_right =
        std::min(a.x2, b.x2);

    const float inter_bottom =
        std::min(a.y2, b.y2);

    const float inter_width =
        std::max(0.0f, inter_right - inter_left);

    const float inter_height =
        std::max(0.0f, inter_bottom - inter_top);

    const float inter_area =
        inter_width * inter_height;

    const float area_a =
        std::max(0.0f, a.x2 - a.x1) *
        std::max(0.0f, a.y2 - a.y1);

    const float area_b =
        std::max(0.0f, b.x2 - b.x1) *
        std::max(0.0f, b.y2 - b.y1);

    const float union_area =
        area_a + area_b - inter_area;

    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return inter_area / union_area;
}

static void apply_nms(std::vector<DetectionCandidate> &detections,
                      float nms_threshold)
{
    std::sort(
        detections.begin(),
        detections.end(),
        [](const DetectionCandidate &a,
           const DetectionCandidate &b)
        {
            return a.score > b.score;
        });

    std::vector<DetectionCandidate> selected;
    selected.reserve(detections.size());

    for (const DetectionCandidate &candidate : detections)
    {
        bool should_remove = false;

        for (const DetectionCandidate &kept : selected)
        {
            // 不同类别之间不互相抑制
            if (candidate.class_id != kept.class_id)
            {
                continue;
            }

            const float iou =
                calculate_iou(candidate, kept);

            if (iou > nms_threshold)
            {
                should_remove = true;
                break;
            }
        }

        if (!should_remove)
        {
            selected.push_back(candidate);

            if (selected.size() >= OBJ_NUMB_MAX_SIZE)
            {
                break;
            }
        }
    }

    detections.swap(selected);
}

// ============================================================
// 输出形状分析
// ============================================================

struct YoloOutputShape
{
    bool channel_first;

    int channel_count;
    int candidate_count;
    int class_count;
};

/**
 * 支持：
 *
 *   [1, C, N]
 *   [1, N, C]
 *
 * 其中：
 *
 *   C = 4 + 类别数
 */
static bool parse_output_shape(const rknn_tensor_attr *attr,
                               YoloOutputShape &shape)
{
    if (attr == nullptr)
    {
        return false;
    }

    std::vector<int> dims;

    // 删除 batch=1 之类的单维度
    for (uint32_t i = 0; i < attr->n_dims; ++i)
    {
        const int dim =
            static_cast<int>(attr->dims[i]);

        if (dim > 1)
        {
            dims.push_back(dim);
        }
    }

    if (dims.size() != 2)
    {
        fprintf(stderr,
                "Unsupported output dimensions: [");

        for (uint32_t i = 0;
             i < attr->n_dims;
             ++i)
        {
            fprintf(stderr,
                    "%u%s",
                    attr->dims[i],
                    i + 1 == attr->n_dims
                        ? ""
                        : ", ");
        }

        fprintf(stderr, "]\n");
        return false;
    }

    const int dim0 = dims[0];
    const int dim1 = dims[1];

    /*
     * YOLOv8 检测输出中：
     *
     * 通道数量通常比较小，例如：
     *   5、8、14、84
     *
     * 候选框数量通常比较大，例如：
     *   8400
     */
    if (dim0 < dim1)
    {
        shape.channel_first = true;
        shape.channel_count = dim0;
        shape.candidate_count = dim1;
    }
    else
    {
        shape.channel_first = false;
        shape.channel_count = dim1;
        shape.candidate_count = dim0;
    }

    shape.class_count =
        shape.channel_count - 4;

    if (shape.class_count <= 0)
    {
        fprintf(stderr,
                "Invalid YOLOv8 channel count: %d\n",
                shape.channel_count);
        return false;
    }

    printf("YOLOv8 output layout: %s\n",
           shape.channel_first
               ? "[1, channels, candidates]"
               : "[1, candidates, channels]");

    printf("YOLOv8 output: channels=%d, classes=%d, candidates=%d\n",
           shape.channel_count,
           shape.class_count,
           shape.candidate_count);

    return true;
}

// ============================================================
// YOLOv8 单输出解析
// ============================================================

static int decode_yolov8_output(
    const void *output_buffer,
    const rknn_tensor_attr *output_attr,
    int model_width,
    int model_height,
    float confidence_threshold,
    std::vector<DetectionCandidate> &detections)
{
    if (output_buffer == nullptr ||
        output_attr == nullptr)
    {
        fprintf(stderr,
                "decode_yolov8_output received null pointer\n");
        return -1;
    }

    YoloOutputShape shape{};

    if (!parse_output_shape(output_attr, shape))
    {
        return -1;
    }

    /*
     * 标签数量和模型类别数不一致时给出警告。
     * 但不让程序崩溃。
     */
    if (!g_labels.empty() &&
        static_cast<int>(g_labels.size()) != shape.class_count)
    {
        fprintf(stderr,
                "Warning: model has %d classes, "
                "but labels file contains %zu labels\n",
                shape.class_count,
                g_labels.size());
    }

    auto get_value =
        [&](int candidate_index,
            int channel_index,
            float &value) -> bool
    {
        size_t tensor_index = 0;

        if (shape.channel_first)
        {
            /*
             * [C, N]
             *
             * index = channel * N + candidate
             */
            tensor_index =
                static_cast<size_t>(channel_index) *
                    static_cast<size_t>(shape.candidate_count) +
                static_cast<size_t>(candidate_index);
        }
        else
        {
            /*
             * [N, C]
             *
             * index = candidate * C + channel
             */
            tensor_index =
                static_cast<size_t>(candidate_index) *
                    static_cast<size_t>(shape.channel_count) +
                static_cast<size_t>(channel_index);
        }

        return read_tensor_value(
            output_buffer,
            output_attr,
            tensor_index,
            value);
    };

    detections.clear();
    detections.reserve(shape.candidate_count);

    for (int candidate_index = 0;
         candidate_index < shape.candidate_count;
         ++candidate_index)
    {
        float center_x = 0.0f;
        float center_y = 0.0f;
        float box_width = 0.0f;
        float box_height = 0.0f;

        if (!get_value(candidate_index, 0, center_x) ||
            !get_value(candidate_index, 1, center_y) ||
            !get_value(candidate_index, 2, box_width) ||
            !get_value(candidate_index, 3, box_height))
        {
            fprintf(stderr,
                    "Failed to read bbox values\n");
            return -1;
        }

        float best_score = -1.0f;
        int best_class_id = -1;

        for (int class_index = 0;
             class_index < shape.class_count;
             ++class_index)
        {
            float class_score = 0.0f;

            if (!get_value(
                    candidate_index,
                    4 + class_index,
                    class_score))
            {
                fprintf(stderr,
                        "Failed to read class score\n");
                return -1;
            }

            /*
             * Ultralytics 导出的最终输出通常已经经过 sigmoid。
             *
             * 如果分数明显不在 0～1 范围，则把它当作 logits，
             * 再执行 sigmoid。
             */
            if (class_score < 0.0f ||
                class_score > 1.0f)
            {
                class_score = sigmoid(class_score);
            }

            if (class_score > best_score)
            {
                best_score = class_score;
                best_class_id = class_index;
            }
        }

        if (best_class_id < 0 ||
            best_score < confidence_threshold)
        {
            continue;
        }

        /*
         * YOLOv8 Detect 最终输出通常是：
         *
         * center_x, center_y, width, height
         */
        float x1 =
            center_x - box_width * 0.5f;

        float y1 =
            center_y - box_height * 0.5f;

        float x2 =
            center_x + box_width * 0.5f;

        float y2 =
            center_y + box_height * 0.5f;

        x1 = clamp_float(
            x1,
            0.0f,
            static_cast<float>(model_width));

        y1 = clamp_float(
            y1,
            0.0f,
            static_cast<float>(model_height));

        x2 = clamp_float(
            x2,
            0.0f,
            static_cast<float>(model_width));

        y2 = clamp_float(
            y2,
            0.0f,
            static_cast<float>(model_height));

        if (x2 <= x1 || y2 <= y1)
        {
            continue;
        }

        DetectionCandidate detection{};

        detection.x1 = x1;
        detection.y1 = y1;
        detection.x2 = x2;
        detection.y2 = y2;

        detection.score = best_score;
        detection.class_id = best_class_id;

        detections.push_back(detection);
    }

    return static_cast<int>(detections.size());
}

// ============================================================
// 对外后处理接口
// ============================================================

int post_process(
    void *output_buffer,
    const rknn_tensor_attr *output_attr,
    int model_in_h,
    int model_in_w,
    float conf_threshold,
    float nms_threshold,
    BOX_RECT pads,
    float scale_w,
    float scale_h,
    detect_result_group_t *group)
{
    if (group == nullptr)
    {
        fprintf(stderr,
                "post_process: group is null\n");
        return -1;
    }

    memset(group,
           0,
           sizeof(detect_result_group_t));

    if (!g_labels_loaded)
    {
        load_labels(LABEL_NALE_TXT_PATH);
        g_labels_loaded = true;
    }

    if (output_buffer == nullptr ||
        output_attr == nullptr)
    {
        fprintf(stderr,
                "post_process: output is null\n");
        return -1;
    }

    if (scale_w <= 0.0f ||
        scale_h <= 0.0f)
    {
        fprintf(stderr,
                "Invalid letterbox scale: "
                "scale_w=%f, scale_h=%f\n",
                scale_w,
                scale_h);
        return -1;
    }

    std::vector<DetectionCandidate> detections;

    const int decode_result =
        decode_yolov8_output(
            output_buffer,
            output_attr,
            model_in_w,
            model_in_h,
            conf_threshold,
            detections);

    if (decode_result < 0)
    {
        group->count = 0;
        return -1;
    }

    if (detections.empty())
    {
        group->count = 0;
        return 0;
    }

    apply_nms(
        detections,
        nms_threshold);

    int result_count = 0;

    for (const DetectionCandidate &detection : detections)
    {
        if (result_count >= OBJ_NUMB_MAX_SIZE)
        {
            break;
        }

        /*
         * 从模型输入坐标恢复到原图坐标：
         *
         * 1. 去掉 letterbox padding
         * 2. 除以 resize scale
         */
        float original_x1 =
            (detection.x1 -
             static_cast<float>(pads.left)) /
            scale_w;

        float original_y1 =
            (detection.y1 -
             static_cast<float>(pads.top)) /
            scale_h;

        float original_x2 =
            (detection.x2 -
             static_cast<float>(pads.left)) /
            scale_w;

        float original_y2 =
            (detection.y2 -
             static_cast<float>(pads.top)) /
            scale_h;

        original_x1 =
            std::max(0.0f, original_x1);

        original_y1 =
            std::max(0.0f, original_y1);

        original_x2 =
            std::max(0.0f, original_x2);

        original_y2 =
            std::max(0.0f, original_y2);

        detect_result_t &result =
            group->results[result_count];

        result.box.left =
            static_cast<int>(original_x1);

        result.box.top =
            static_cast<int>(original_y1);

        result.box.right =
            static_cast<int>(original_x2);

        result.box.bottom =
            static_cast<int>(original_y2);

        result.prop =
            detection.score;

        result.cls_id =
            detection.class_id;

        snprintf(
            result.name,
            OBJ_NAME_MAX_SIZE,
            "%s",
            class_name(detection.class_id));

        result_count++;
    }

    group->count = result_count;

    return 0;
}

// vector<string> 会自动释放，不再需要手工 free
void deinitPostProcess()
{
    g_labels.clear();
    g_labels_loaded = false;
}