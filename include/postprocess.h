#ifndef _RKNN_YOLOV8_POSTPROCESS_H_
#define _RKNN_YOLOV8_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#define OBJ_NAME_MAX_SIZE     16
#define OBJ_NUMB_MAX_SIZE     128
#define OBJ_CLASS_NUM         1          // ← 你的类别数（本例 ball/human/rim = 3）
#define NMS_THRESH            0.45f
#define BOX_THRESH            0.25f
#define DFL_LEN               16         // 64 / 4，YOLOv8 固定
#define LABEL_NALE_TXT_PATH   "./model/labels_list.txt"  // ← 你的标签文件

typedef struct _BOX_RECT {
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t {
    char     name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float    prop;
    int      cls_id;
} detect_result_t;

typedef struct _detect_result_group_t {
    int             id;
    int             count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

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
    detect_result_group_t *group);

void deinitPostProcess();

#endif // _RKNN_YOLOV8_POSTPROCESS_H_
