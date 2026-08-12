# RK3588 单模型绝缘子缺陷检测程序

## 项目用途

本项目是在 RK3588 Linux 板卡上运行的 C++ 单模型实时检测程序。程序从
USB 摄像头采集画面，使用 RKNN Runtime 调用 YOLOv8-Seg RKNN 模型完成
缺陷检测和分割，然后把下列信息直接绘制到视频帧中：

- 缺陷类别；
- 检测框；
- 置信度；
- 分割掩码与缺陷面积比例；
- 推理线程数、FPS 和帧编号。

标注后的最终画面不在 RKNN 程序中创建本地窗口，而是通过本机
TCP-JPEG 发送给 Qt 程序 `InsulatorMonitorSingle` 显示。

```text
/dev/video41（MJPG 1280×720@30 FPS）
    ↓ GStreamer 解码为 BGR
1280×720 原始画面
    ↓ 等比例缩放并填充
640×640 YOLOv8-Seg 模型输入
    ↓ RKNN 多线程推理
检测框、类别、置信度和掩码映射回 1280×720
    ↓ JPEG 编码
127.0.0.1:5000
    ↓
InsulatorMonitorSingle
```

## 运行环境

- RK3588 或 RK3588S Linux 板卡；
- RKNN Runtime；
- RGA；
- OpenCV 3.4.5；
- GStreamer 1.0；
- 支持 MJPG 1280×720@30 FPS 的 USB 摄像头；
- 默认摄像头设备：`/dev/video41`。

检查必要的 GStreamer 插件：

```bash
for e in appsrc appsink queue videoconvert jpegenc jpegparse jpegdec \
         tcpserversink tcpclientsrc; do
    gst-inspect-1.0 "$e" >/dev/null 2>&1 \
        && echo "$e: YES" \
        || echo "$e: MISSING"
done
```

检查摄像头是否支持目标格式：

```bash
v4l2-ctl -d /dev/video41 --list-formats-ext
```

## 编译

工程默认使用 ATK RK3588 交叉工具链：

```text
/opt/atk-dlrk3588-toolchain
```

在 Linux 编译电脑中执行：

```bash
cd "rknn-cpp-Multithreading"
chmod +x build-linux_RK3588.sh
./build-linux_RK3588.sh
```

成功后主要文件位于：

```text
install/rknn_yolov5_demo_Linux/
├── rknn_yolov8_demo
├── lib/
└── model/
```

将整个 `install/rknn_yolov5_demo_Linux` 目录复制到 RK3588，避免遗漏
RKNN、RGA 动态库或模型文件。

## 命令行参数

```text
rknn_yolov8_demo <model.rknn> [camera_device] [thread_num]
```

参数说明：

| 参数 | 必需 | 说明 |
|---|---:|---|
| `model.rknn` | 是 | YOLOv8-Seg RKNN 模型路径 |
| `camera_device` | 否 | 摄像头设备，默认 `/dev/video41` |
| `thread_num` | 否 | RKNN 推理实例数量，当前代码默认 12 |

更换模型时，新模型必须保持与当前模型相同的 YOLOv8-Seg 输入、输出张量
结构。仅文件路径不同的兼容模型不需要修改后处理代码。

## 首次运行：必须先使用单线程

不要在未经验证时直接使用默认的 12 线程。每个线程都会创建 RKNN 模型
对象和推理上下文，线程数过高会显著增加内存、NPU负载和瞬时功耗。在供电、
散热或内存不足时，可能造成进程被杀死、内核异常，甚至整块板卡掉电。

首次运行请使用 `1` 个线程：

```bash
cd /root/project2

./rknn_yolov8_demo \
    model/RK3588/fixed_single.rknn \
    /dev/video41 \
    1
```

确认单线程稳定后，再依次测试 `2`、`3` 个线程。不要直接从 1 增加到 12。

## 不使用 Qt 时测试视频流

先在终端一启动 RKNN 程序：

```bash
./rknn_yolov8_demo \
    model/RK3588/fixed_single.rknn \
    /dev/video41 \
    1
```

程序完成初始化后会输出：

```text
@status {"state":"ready","width":1280,"height":720,"fps":30,"port":5000}
```

再在终端二连接视频流：

```bash
gst-launch-1.0 \
    tcpclientsrc host=127.0.0.1 port=5000 ! \
    jpegparse ! jpegdec ! videoconvert ! \
    autovideosink sync=false
```

如果可以看到检测画面，说明摄像头、RKNN 推理、JPEG 编码和 TCP 发送均已
工作，可以继续测试 Qt。

## 与 Qt 程序配合使用

Qt 单模型程序读取与其可执行文件放在同一目录的 `single_model.ini`：

```ini
[Inference]
rknn_dir=/root/project2
program=rknn_yolov8_demo
model=model/RK3588/fixed_single.rknn
```

推荐的板卡目录结构：

```text
/root/project2/
├── InsulatorMonitorSingle
├── single_model.ini
├── rknn_yolov8_demo
├── lib/
└── model/
    └── RK3588/
        └── fixed_single.rknn
```

添加执行权限并启动 Qt：

```bash
cd /root/project2
chmod +x InsulatorMonitorSingle rknn_yolov8_demo
./InsulatorMonitorSingle
```

点击“开始检测”后，Qt 使用 `QProcess` 启动 RKNN 程序。RKNN 完成模型、
摄像头和 TCP 初始化后发送 `@status ready`，Qt 随后连接
`127.0.0.1:5000` 并显示画面。

> **重要：** 当前 Qt 启动配置仍会向 RKNN 程序传入 `12` 个线程。板卡曾经
> 出现掉电时，不要继续使用该 Qt 二进制启动检测。应先把 Qt 工程
> `inferencelaunchspec.cpp` 中单模型参数的 `"12"` 改为 `"1"`，重新编译并
> 部署；或者先只使用上面的命令行单线程方式验证。

## Qt/RKNN 通讯协议

RKNN 程序通过标准输出发送一行一条的 JSON 消息：

启动成功：

```text
@status {"state":"ready","width":1280,"height":720,"fps":30,"port":5000}
```

运行指标：

```text
@metrics {"capture_fps":8.34,"pipeline_fps":0.69,"latency_ms":1431.55}
```

运行错误：

```text
@error {"code":"camera_open_failed","message":"Failed to open camera"}
```

普通诊断信息写入标准错误。Qt 根据这些消息更新启动状态、FPS、延迟和错误
提示。

## 常见问题

### 1. 出现多次 `Loading model...`

每出现一次通常代表正在创建一个模型对象或 RKNN 上下文。例如出现 12 次，
说明启动参数中的线程数为 12。先改用单线程排查。

### 2. 板卡直接关机或重启（问题已查明）

整板掉电更像供电压降、功耗峰值、温度保护、内核崩溃或硬件看门狗，而不是
普通应用退出。重启后检查：

```bash
journalctl -b -1 -k | tail -n 200
dmesg -T | grep -Ei "oom|killed|thermal|voltage|watchdog|panic|rknn|rga"
ls -l /sys/fs/pstore
```

在确认原因前不要再次运行 12 线程，也不要执行 CPU/NPU 定频脚本。

问题：使用的RKNN模型太大，模型运行负载太高，多线程会复制多份运行上下文、特征图和输出缓冲区，导致的问题有
- NPU 内存和带宽压力过大；
- 多个 RKNN context 同时占用内部资源；
- 分割输出占用大量内存；
- NPU 任务超过 6 秒超时；
- 最终触发 RKNPU 驱动异常。

经过测试，裁剪后的yolov8s_seg_ReLU_pruned10.rknn(18.4 MB)模型运行半小时后死机，在进行计算时CPU负载在90%以上。

### 3. Qt 提示找不到程序或模型

确认 `single_model.ini` 中的 `rknn_dir` 是绝对路径，并检查：

```bash
test -x /root/project2/rknn_yolov8_demo && echo "program: OK"
test -r /root/project2/model/RK3588/fixed_single.rknn && echo "model: OK"
```

### 4. TCP 端口被占用

```bash
ss -ltnp | grep ':5000'
```

同一时刻只能运行一套检测程序。双模型和单模型程序不能同时占用摄像头与
TCP 5000 端口。

### 5. 画面延迟逐渐增加

发送端和 Qt 接收端都应只保留最新帧。确认使用的是当前版本，并检查管线中
存在：

```text
queue max-size-buffers=1 leaky=downstream
appsink max-buffers=1 drop=true sync=false
```

## 项目目录

```text
include/                 RKNN、RGA、线程池和管线接口
src/main.cc              摄像头、线程池、协议与主循环
src/rkYolov5s.cc         YOLOv8-Seg 预处理、推理和后处理
src/tcp_jpeg_publisher.cpp  TCP-JPEG 发送器
tests/                   不依赖板卡硬件的逻辑测试
model/RK3588/            RKNN 模型
build-linux_RK3588.sh    RK3588 交叉编译脚本
```

## 鸣谢

- [Rockchip rknpu2](https://github.com/rockchip-linux/rknpu2)
- [rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)
- [dpool](https://github.com/senlinzhan/dpool)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
