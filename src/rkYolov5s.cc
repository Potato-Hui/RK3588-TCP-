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

cv::Mat rkYolov5s::infer(cv::Mat &orig_img)
{
    std::lock_guard<std::mutex> lock(mtx);

    if (orig_img.empty())
    {
        fprintf(stderr, "infer: input image is empty\n");
        return orig_img;
    }

    // OpenCV 摄像头图像为 BGR，模型输入为 RGB
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    img_width = img.cols;
    img_height = img.rows;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));

    const cv::Size target_size(width, height);

    cv::Mat resized_img(
        target_size.height,
        target_size.width,
        CV_8UC3);

    // 当前 RGA resize 是直接拉伸到模型输入尺寸，不使用 letterbox
    float scale_w =
        static_cast<float>(target_size.width) /
        static_cast<float>(img.cols);

    float scale_h =
        static_cast<float>(target_size.height) /
        static_cast<float>(img.rows);

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
            target_size);

        if (ret != 0)
        {
            fprintf(stderr,
                    "resize_rga failed, ret=%d\n",
                    ret);

            return orig_img;
        }

        if (resized_img.empty() ||
            resized_img.data == nullptr)
        {
            fprintf(stderr,
                    "resize_rga returned an empty image\n");

            return orig_img;
        }

        inputs[0].buf = resized_img.data;
    }
    else
    {
        scale_w = 1.0f;
        scale_h = 1.0f;
        inputs[0].buf = img.data;
    }

    // 设置 RKNN 输入
    ret = rknn_inputs_set(
        ctx,
        io_num.n_input,
        inputs);

    if (ret < 0)
    {
        fprintf(stderr,
                "rknn_inputs_set failed, ret=%d\n",
                ret);

        return orig_img;
    }

    // 当前通用后处理只支持 YOLOv8 Detect 单输出模型
    if (io_num.n_output != 1)
    {
        fprintf(stderr,
                "Current postprocess expects 1 output, "
                "but model has %u outputs\n",
                io_num.n_output);

        return orig_img;
    }

    std::vector<rknn_output> outputs(io_num.n_output);

    memset(
        outputs.data(),
        0,
        sizeof(rknn_output) * outputs.size());

    for (uint32_t i = 0;
         i < io_num.n_output;
         ++i)
    {
        outputs[i].index = i;

        // 返回模型原始输出类型，由 postprocess.cc 根据 attr->type 解析
        outputs[i].want_float = 0;
    }

    // 执行推理
    ret = rknn_run(ctx, nullptr);

    if (ret < 0)
    {
        fprintf(stderr,
                "rknn_run failed, ret=%d\n",
                ret);

        return orig_img;
    }

    // 获取输出
    ret = rknn_outputs_get(
        ctx,
        io_num.n_output,
        outputs.data(),
        nullptr);

    if (ret < 0)
    {
        fprintf(stderr,
                "rknn_outputs_get failed, ret=%d\n",
                ret);

        return orig_img;
    }

    detect_result_group_t detect_result_group;
    memset(
        &detect_result_group,
        0,
        sizeof(detect_result_group));

    // 新版 YOLOv8 单输出后处理
    const int post_ret = post_process(
        outputs[0].buf,
        &output_attrs[0],
        height,
        width,
        box_conf_threshold,
        nms_threshold,
        pads,
        scale_w,
        scale_h,
        &detect_result_group);

    if (post_ret != 0)
    {
        fprintf(stderr,
                "post_process failed, ret=%d\n",
                post_ret);
    }

    // 输出数据使用完后及时释放
    ret = rknn_outputs_release(
        ctx,
        io_num.n_output,
        outputs.data());

    if (ret < 0)
    {
        fprintf(stderr,
                "rknn_outputs_release failed, ret=%d\n",
                ret);
    }

    // 绘制检测结果
    char text[256];

    for (int i = 0;
         i < detect_result_group.count;
         ++i)
    {
        detect_result_t *det_result =
            &detect_result_group.results[i];

        snprintf(
            text,
            sizeof(text),
            "%s %.1f%%",
            det_result->name,
            det_result->prop * 100.0f);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        // 防止坐标超过原图范围
        x1 = std::max(0, std::min(x1, orig_img.cols - 1));
        y1 = std::max(0, std::min(y1, orig_img.rows - 1));
        x2 = std::max(0, std::min(x2, orig_img.cols - 1));
        y2 = std::max(0, std::min(y2, orig_img.rows - 1));

        if (x2 <= x1 || y2 <= y1)
        {
            continue;
        }

        cv::rectangle(
            orig_img,
            cv::Point(x1, y1),
            cv::Point(x2, y2),
            cv::Scalar(255, 0, 0),
            3);

        int text_y = y1 > 18
                         ? y1 - 5
                         : y1 + 18;

        cv::putText(
            orig_img,
            text,
            cv::Point(x1, text_y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 255, 255),
            1);
    }

    return orig_img;
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
