#include <stdio.h>
#include <mutex>
#include "rknn_api.h"

#include "postprocess.h"
#include "preprocess.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "coreNum.hpp"
#include "rkYolov5s.hpp"
#include "letterbox_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    // printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
    //        "type=%s, qnt_type=%s, "
    //        "zp=%d, scale=%f\n",
    //        attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
    //        attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
    //        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

rkYolov5s::rkYolov5s(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
}

int rkYolov5s::init(rknn_context *ctx_in, bool share_weight)
{
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    // 模型参数复用/Model parameter reuse
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入参数/Set the input parameters
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 设置输出参数/Set the output parameters
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}

rknn_context *rkYolov5s::get_pctx()
{
    return &ctx;
}
struct yolov8_seg_candidate_t
{
    cv::Rect2f model_box;
    cv::Rect original_box;
    float confidence;
    int class_id;
    std::vector<float> mask_coefficients;
};

static float calculate_iou(
    const cv::Rect2f &box_a,
    const cv::Rect2f &box_b)
{
    const float left =
        std::max(box_a.x, box_b.x);

    const float top =
        std::max(box_a.y, box_b.y);

    const float right =
        std::min(
            box_a.x + box_a.width,
            box_b.x + box_b.width);

    const float bottom =
        std::min(
            box_a.y + box_a.height,
            box_b.y + box_b.height);

    const float intersection_width =
        std::max(0.0f, right - left);

    const float intersection_height =
        std::max(0.0f, bottom - top);

    const float intersection_area =
        intersection_width * intersection_height;

    const float union_area =
        box_a.area() +
        box_b.area() -
        intersection_area;

    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return intersection_area / union_area;
}

static std::vector<int> perform_class_nms(
    const std::vector<yolov8_seg_candidate_t> &candidates,
    float nms_threshold)
{
    std::vector<int> order(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i)
    {
        order[i] = static_cast<int>(i);
    }

    std::sort(
        order.begin(),
        order.end(),
        [&candidates](int left, int right)
        {
            return candidates[left].confidence >
                   candidates[right].confidence;
        });

    std::vector<int> keep_indices;
    std::vector<unsigned char> removed(
        candidates.size(),
        0);

    for (size_t i = 0; i < order.size(); ++i)
    {
        const int current_index = order[i];

        if (removed[current_index])
        {
            continue;
        }

        keep_indices.push_back(current_index);

        for (size_t j = i + 1;
             j < order.size();
             ++j)
        {
            const int compare_index = order[j];

            if (removed[compare_index])
            {
                continue;
            }

            if (candidates[current_index].class_id !=
                candidates[compare_index].class_id)
            {
                continue;
            }

            const float iou = calculate_iou(
                candidates[current_index].model_box,
                candidates[compare_index].model_box);

            if (iou > nms_threshold)
            {
                removed[compare_index] = 1;
            }
        }
    }

    return keep_indices;
}

static cv::Scalar get_seg_color(int class_id)
{
    static const cv::Scalar colors[] =
    {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 255, 0)
    };

    const int color_count =
        static_cast<int>(
            sizeof(colors) / sizeof(colors[0]));

    int color_index = class_id % color_count;

    if (color_index < 0)
    {
        color_index = 0;
    }

    return colors[color_index];
}

static std::string get_defect_name(int class_id)
{
    // 类别顺序必须和训练数据集中的类别顺序完全一致
    static const std::vector<std::string> class_names =
    {
        "insulator",
        "crack",
        "pollution",
        "flashover",
        "broken"
    };

    if (class_id >= 0 &&
        class_id < static_cast<int>(class_names.size()))
    {
        return class_names[class_id];
    }

    return "class_" + std::to_string(class_id);
}
InferenceResult rkYolov5s::infer(cv::Mat &orig_img)
{
    InferenceResult result;
    std::lock_guard<std::mutex> lock(mtx);

    if (orig_img.empty())
    {
        fprintf(stderr, "infer: input image is empty\n");
        return result;
    }

    result.originalFrame = orig_img.clone();
    result.annotatedFrame = orig_img.clone();
    cv::Mat &annotated_img = result.annotatedFrame;

    // OpenCV 摄像头图像为 BGR，模型输入为 RGB
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    img_width = img.cols;
    img_height = img.rows;

    const LetterboxGeometry geometry =
        makeLetterboxGeometry(img.cols, img.rows, width, height);

    cv::Mat resized_img(
        geometry.resizedHeight,
        geometry.resizedWidth,
        CV_8UC3);
    cv::Mat model_input(height, width, CV_8UC3,
                        cv::Scalar(114, 114, 114));

    if (img_width != width || img_height != height)
    {
        rga_buffer_t src;
        rga_buffer_t dst;

        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));

        ret = resize_rga(
            src,
            dst,
            img,
            resized_img,
            resized_img.size());

        if (ret != 0)
        {
            fprintf(stderr,
                    "resize_rga failed, ret=%d\n",
                    ret);

            return result;
        }

        if (resized_img.empty() ||
            resized_img.data == nullptr)
        {
            fprintf(stderr,
                    "resize_rga returned an empty image\n");

            return result;
        }

        resized_img.copyTo(model_input(cv::Rect(
            geometry.padLeft,
            geometry.padTop,
            geometry.resizedWidth,
            geometry.resizedHeight)));
        inputs[0].buf = model_input.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    // 设置 RKNN 输入
    ret = rknn_inputs_set(
        ctx,
        io_num.n_input,
        inputs);

    if (ret != RKNN_SUCC)
    {
        fprintf(stderr,
                "rknn_inputs_set failed, ret=%d\n",
                ret);

        return result;
    }

    // YOLOv8-Seg需要两个输出：检测结果和原型掩码
    if (io_num.n_output != 2)
    {
        fprintf(stderr,
                "YOLOv8-Seg expects 2 outputs, but model has %u outputs\n",
                io_num.n_output);

        return result;
    }

    if (output_attrs[0].n_dims != 3 ||
        output_attrs[1].n_dims != 4)
    {
        fprintf(stderr,
                "Unsupported YOLOv8-Seg output dimensions: "
                "output0 n_dims=%u, output1 n_dims=%u\n",
                output_attrs[0].n_dims,
                output_attrs[1].n_dims);

        return result;
    }

    std::vector<rknn_output> outputs(io_num.n_output);

    memset(
        outputs.data(),
        0,
        outputs.size() * sizeof(rknn_output));

    for (uint32_t i = 0;
         i < io_num.n_output;
         ++i)
    {
        outputs[i].index = i;

        // RKNN Runtime将输出统一转换为float
        outputs[i].want_float = 1;
    }

    // 执行推理
    ret = rknn_run(ctx, nullptr);

    if (ret != RKNN_SUCC)
    {
        fprintf(stderr,
                "rknn_run failed, ret=%d\n",
                ret);

        return result;
    }

    // 获取输出
    ret = rknn_outputs_get(
        ctx,
        io_num.n_output,
        outputs.data(),
        nullptr);

    if (ret != RKNN_SUCC)
    {
        fprintf(stderr,
                "rknn_outputs_get failed, ret=%d\n",
                ret);

        return result;
    }

    const float *detection_output =
        static_cast<const float *>(outputs[0].buf);

    const float *prototype_output =
        static_cast<const float *>(outputs[1].buf);

    if (detection_output == nullptr ||
        prototype_output == nullptr)
    {
        fprintf(stderr,
                "YOLOv8-Seg output buffer is null\n");

        rknn_outputs_release(
            ctx,
            io_num.n_output,
            outputs.data());

        return result;
    }

    const int output_channels =
        static_cast<int>(output_attrs[0].dims[1]);

    const int candidate_count =
        static_cast<int>(output_attrs[0].dims[2]);

    const int mask_channels =
        static_cast<int>(output_attrs[1].dims[1]);

    const int mask_height =
        static_cast<int>(output_attrs[1].dims[2]);

    const int mask_width =
        static_cast<int>(output_attrs[1].dims[3]);

    // output0通道组成：4个框坐标 + 类别置信度 + mask系数
    const int class_count =
        output_channels - 4 - mask_channels;

    if (class_count <= 0 ||
        candidate_count <= 0 ||
        mask_channels <= 0 ||
        mask_width <= 0 ||
        mask_height <= 0)
    {
        fprintf(stderr,
                "Invalid YOLOv8-Seg shape: "
                "channels=%d candidates=%d classes=%d "
                "mask=[%d,%d,%d]\n",
                output_channels,
                candidate_count,
                class_count,
                mask_channels,
                mask_height,
                mask_width);

        rknn_outputs_release(
            ctx,
            io_num.n_output,
            outputs.data());

        return result;
    }

    const float mask_threshold = 0.5f;
    const int maximum_detections = 50;

    std::vector<yolov8_seg_candidate_t> candidates;
    candidates.reserve(100);

    for (int candidate_index = 0;
         candidate_index < candidate_count;
         ++candidate_index)
    {
        int best_class_id = -1;
        float best_confidence = 0.0f;

        for (int class_index = 0;
             class_index < class_count;
             ++class_index)
        {
            const int channel = 4 + class_index;

            const float confidence =
                detection_output[
                    channel * candidate_count +
                    candidate_index];

            if (confidence > best_confidence)
            {
                best_confidence = confidence;
                best_class_id = class_index;
            }
        }

        if (best_class_id < 0 ||
            best_confidence < box_conf_threshold)
        {
            continue;
        }

        const float center_x =
            detection_output[
                0 * candidate_count +
                candidate_index];

        const float center_y =
            detection_output[
                1 * candidate_count +
                candidate_index];

        const float box_width =
            detection_output[
                2 * candidate_count +
                candidate_index];

        const float box_height =
            detection_output[
                3 * candidate_count +
                candidate_index];

        float model_left =
            center_x - box_width * 0.5f;

        float model_top =
            center_y - box_height * 0.5f;

        float model_right =
            center_x + box_width * 0.5f;

        float model_bottom =
            center_y + box_height * 0.5f;

        model_left =
            std::max(
                0.0f,
                std::min(
                    model_left,
                    static_cast<float>(width - 1)));

        model_top =
            std::max(
                0.0f,
                std::min(
                    model_top,
                    static_cast<float>(height - 1)));

        model_right =
            std::max(
                0.0f,
                std::min(
                    model_right,
                    static_cast<float>(width)));

        model_bottom =
            std::max(
                0.0f,
                std::min(
                    model_bottom,
                    static_cast<float>(height)));

        if (model_right <= model_left ||
            model_bottom <= model_top)
        {
            continue;
        }

        const FloatRect mapped = mapModelRectToSource(
            FloatRect{model_left, model_top,
                      model_right - model_left,
                      model_bottom - model_top},
            geometry);
        if (mapped.width <= 0.0f || mapped.height <= 0.0f)
        {
            continue;
        }

        const float content_left = static_cast<float>(geometry.padLeft);
        const float content_top = static_cast<float>(geometry.padTop);
        const float content_right = content_left + geometry.resizedWidth;
        const float content_bottom = content_top + geometry.resizedHeight;
        model_left = std::max(model_left, content_left);
        model_top = std::max(model_top, content_top);
        model_right = std::min(model_right, content_right);
        model_bottom = std::min(model_bottom, content_bottom);

        int original_left = static_cast<int>(std::floor(mapped.x));
        int original_top = static_cast<int>(std::floor(mapped.y));
        int original_right = static_cast<int>(std::ceil(mapped.x + mapped.width));
        int original_bottom = static_cast<int>(std::ceil(mapped.y + mapped.height));

        original_left =
            std::max(
                0,
                std::min(
                    original_left,
                    orig_img.cols - 1));

        original_top =
            std::max(
                0,
                std::min(
                    original_top,
                    orig_img.rows - 1));

        original_right =
            std::max(
                0,
                std::min(
                    original_right,
                    orig_img.cols));

        original_bottom =
            std::max(
                0,
                std::min(
                    original_bottom,
                    orig_img.rows));

        if (original_right <= original_left ||
            original_bottom <= original_top)
        {
            continue;
        }

        yolov8_seg_candidate_t candidate;

        candidate.model_box = cv::Rect2f(
            model_left,
            model_top,
            model_right - model_left,
            model_bottom - model_top);

        candidate.original_box = cv::Rect(
            original_left,
            original_top,
            original_right - original_left,
            original_bottom - original_top);

        candidate.confidence = best_confidence;
        candidate.class_id = best_class_id;

        candidate.mask_coefficients.resize(
            mask_channels);

        for (int mask_channel = 0;
             mask_channel < mask_channels;
             ++mask_channel)
        {
            const int output_channel =
                4 +
                class_count +
                mask_channel;

            candidate.mask_coefficients[mask_channel] =
                detection_output[
                    output_channel * candidate_count +
                    candidate_index];
        }

        candidates.push_back(candidate);
    }

    const std::vector<int> keep_indices =
        perform_class_nms(
            candidates,
            nms_threshold);

    int draw_count = 0;

    for (size_t keep_position = 0;
         keep_position < keep_indices.size();
         ++keep_position)
    {
        if (draw_count >= maximum_detections)
        {
            break;
        }

        const yolov8_seg_candidate_t &candidate =
            candidates[keep_indices[keep_position]];

        cv::Mat mask_logits(
            mask_height,
            mask_width,
            CV_32FC1,
            cv::Scalar(0));

        for (int mask_y = 0;
             mask_y < mask_height;
             ++mask_y)
        {
            float *mask_row =
                mask_logits.ptr<float>(mask_y);

            for (int mask_x = 0;
                 mask_x < mask_width;
                 ++mask_x)
            {
                const int pixel_index =
                    mask_y * mask_width +
                    mask_x;

                float logit = 0.0f;

                for (int mask_channel = 0;
                     mask_channel < mask_channels;
                     ++mask_channel)
                {
                    const int prototype_index =
                        mask_channel *
                            mask_height *
                            mask_width +
                        pixel_index;

                    logit +=
                        candidate.mask_coefficients[
                            mask_channel] *
                        prototype_output[
                            prototype_index];
                }

                mask_row[mask_x] =
                    1.0f /
                    (1.0f + std::exp(-logit));
            }
        }

        int prototype_left =
            static_cast<int>(
                candidate.model_box.x *
                static_cast<float>(mask_width) /
                static_cast<float>(width));

        int prototype_top =
            static_cast<int>(
                candidate.model_box.y *
                static_cast<float>(mask_height) /
                static_cast<float>(height));

        int prototype_right =
            static_cast<int>(
                (candidate.model_box.x +
                 candidate.model_box.width) *
                static_cast<float>(mask_width) /
                static_cast<float>(width));

        int prototype_bottom =
            static_cast<int>(
                (candidate.model_box.y +
                 candidate.model_box.height) *
                static_cast<float>(mask_height) /
                static_cast<float>(height));

        prototype_left =
            std::max(
                0,
                std::min(
                    prototype_left,
                    mask_width - 1));

        prototype_top =
            std::max(
                0,
                std::min(
                    prototype_top,
                    mask_height - 1));

        prototype_right =
            std::max(
                prototype_left + 1,
                std::min(
                    prototype_right,
                    mask_width));

        prototype_bottom =
            std::max(
                prototype_top + 1,
                std::min(
                    prototype_bottom,
                    mask_height));

        const cv::Rect prototype_roi(
            prototype_left,
            prototype_top,
            prototype_right - prototype_left,
            prototype_bottom - prototype_top);

        cv::Mat cropped_mask =
            mask_logits(prototype_roi);

        cv::Mat resized_mask;

        cv::resize(
            cropped_mask,
            resized_mask,
            candidate.original_box.size(),
            0.0,
            0.0,
            cv::INTER_LINEAR);

        cv::Mat binary_mask;

        cv::threshold(
            resized_mask,
            binary_mask,
            mask_threshold,
            255.0,
            cv::THRESH_BINARY);

        binary_mask.convertTo(
            binary_mask,
            CV_8UC1);

        const int defect_pixels =
            cv::countNonZero(binary_mask);

        const int box_pixels =
            candidate.original_box.width *
            candidate.original_box.height;

        const float defect_ratio =
            box_pixels > 0
                ? static_cast<float>(defect_pixels) /
                      static_cast<float>(box_pixels)
                : 0.0f;

        const cv::Scalar color =
            get_seg_color(candidate.class_id);

        SegmentationInstance instance;
        instance.classId = candidate.class_id;
        instance.confidence = candidate.confidence;
        instance.bbox = candidate.original_box;
        instance.mask = binary_mask.clone();
        result.instances.push_back(instance);

        cv::Mat image_roi =
            annotated_img(candidate.original_box);

        cv::Mat color_layer(
            image_roi.size(),
            image_roi.type(),
            color);

        cv::Mat blended_roi;

        cv::addWeighted(
            image_roi,
            0.55,
            color_layer,
            0.45,
            0.0,
            blended_roi);

        blended_roi.copyTo(
            image_roi,
            binary_mask);

        cv::rectangle(
            annotated_img,
            candidate.original_box,
            color,
            2);

        char text[256];

        snprintf(
            text,
            sizeof(text),
            "%s %.1f%% area %.1f%%",
            get_defect_name(candidate.class_id).c_str(),
            candidate.confidence * 100.0f,
            defect_ratio * 100.0f);

        int text_y =
            candidate.original_box.y > 18
                ? candidate.original_box.y - 5
                : candidate.original_box.y + 18;

        cv::putText(
            annotated_img,
            text,
            cv::Point(
                candidate.original_box.x,
                text_y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2);

        ++draw_count;
    }

    // 输出数据使用完后及时释放
    ret = rknn_outputs_release(
        ctx,
        io_num.n_output,
        outputs.data());

    if (ret != RKNN_SUCC)
    {
        fprintf(stderr,
                "rknn_outputs_release failed, ret=%d\n",
                ret);

        return result;
    }

    result.succeeded = true;
    return result;
}
rkYolov5s::~rkYolov5s()
{
    deinitPostProcess();

    ret = rknn_destroy(ctx);

    if (model_data)
        free(model_data);

    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
}

