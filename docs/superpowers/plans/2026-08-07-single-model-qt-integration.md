# Single-Model Qt Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a second Qt executable, `InsulatorMonitorSingle`, that reads `single_model.ini`, starts the single-model RKNN process, and displays its annotated 1280x720 TCP-JPEG stream while preserving the existing dual-model executable.

**Architecture:** Keep one Qt source tree and compile two entry points over shared UI, receiver, and controller code. Refactor the single-model RKNN program into a 1280x720 MJPG camera pipeline, a 640x640 letterbox inference path, and a loopback-only TCP-JPEG publisher using the same JSON-line process protocol as the dual-model program.

**Tech Stack:** C++14, Qt 5.15 Widgets/Core/Gui/Test, QProcess, QSettings, GStreamer 1.22 appsrc/appsink, OpenCV 3.4.5, RGA, RKNN Runtime, CMake/CTest.

## Global Constraints

- Target platform is RK3588 Linux with Qt 5 and C++14.
- Camera is `/dev/video41`, MJPG 1280x720 at 30 FPS.
- Model input remains 640x640 and all replacement models retain the current YOLOv8-Seg tensor layout.
- The single-model configuration file is named `single_model.ini` and is read on every Start request.
- TCP binds only to `127.0.0.1:5000`.
- RKNN draws masks, boxes, class names, confidence, and FPS; Qt displays the finished frame.
- Producer and consumer keep only the newest frame.
- `InsulatorMonitor` remains the fixed dual-model binary; `InsulatorMonitorSingle` is the new single-model binary.
- Single- and dual-model detection are not supported simultaneously.
- Preserve all unrelated existing changes. `rknn-cpp-Multithreading` is already dirty and `InsulatorMonitor` is not a Git repository; inspect every staged diff and never stage unrelated files.

---

## File Map

### Single-model RKNN repository: `D:/codex/rknn-cpp-Multithreading`

- Create `include/letterbox_geometry.hpp`: pure letterbox dimensions and coordinate mapping.
- Create `include/camera_pipeline.hpp`: pure 1280x720 MJPG GStreamer camera string builder.
- Create `include/video_publisher_pipeline.hpp`: pure loopback TCP-JPEG string builder.
- Create `include/monitor_protocol.hpp`: ready, metrics, and error JSON-line formatting.
- Create `include/tcp_jpeg_publisher.hpp`: TCP publisher public interface.
- Create `src/tcp_jpeg_publisher.cpp`: appsrc ownership, caps, buffer copy, and shutdown.
- Modify `src/rkYolov5s.cc`: replace direct stretch with letterbox and reverse-map boxes/masks.
- Modify `src/main.cc`: remove local display, add signals, TCP publisher, protocol, and metrics.
- Modify `CMakeLists.txt`: compile publisher and register host-pure tests.
- Modify `README.md`: document command line, stream, model replacement, and verification.
- Create tests under `tests/` for geometry, pipeline strings, and protocol.

### Qt repository: `D:/codex/InsulatorMonitor`

- Create `inferenceprofile.h`: `DualModel` and `SingleModel` profile enum.
- Create `inferencelaunchspec.h/.cpp`: load and validate dual or single launch settings.
- Create `inferencelifecycle.h`: pure controller-state transition helper.
- Create `monitorapplication.h/.cpp`: shared QApplication setup and signal wiring.
- Create `main_single.cpp`: `InsulatorMonitorSingle` entry point.
- Modify `main.cpp`: delegate to the shared application runner with `DualModel`.
- Modify `inferencecontroller.h/.cpp`: consume a profile/spec and lifecycle helper.
- Create `single_model.ini`: deployable configuration template.
- Modify `CMakeLists.txt`: build two binaries and Qt tests.
- Create `InsulatorMonitorSingle.pro`: optional qmake build for the single target.
- Modify `README.md`: document both binaries and INI deployment.
- Create tests for launch config and lifecycle behavior.

---

### Task 1: Letterbox Geometry Contract

**Files:**
- Create: `include/letterbox_geometry.hpp`
- Create: `tests/letterbox_geometry_test.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `LetterboxGeometry makeLetterboxGeometry(int sourceWidth, int sourceHeight, int modelWidth, int modelHeight)`.
- Produces: `FloatRect mapModelRectToSource(const FloatRect&, const LetterboxGeometry&)`.
- Consumed by: `rkYolov5s::infer(cv::Mat&)` in Task 2.

- [ ] **Step 1: Write the failing geometry test**

```cpp
#include "letterbox_geometry.hpp"
#include <cassert>
#include <cmath>

int main()
{
    const LetterboxGeometry g =
        makeLetterboxGeometry(1280, 720, 640, 640);
    assert(std::fabs(g.scale - 0.5f) < 0.0001f);
    assert(g.resizedWidth == 640);
    assert(g.resizedHeight == 360);
    assert(g.padLeft == 0);
    assert(g.padTop == 140);

    const FloatRect source = mapModelRectToSource(
        FloatRect{0.0f, 140.0f, 640.0f, 360.0f}, g);
    assert(std::fabs(source.x) < 0.01f);
    assert(std::fabs(source.y) < 0.01f);
    assert(std::fabs(source.width - 1280.0f) < 0.01f);
    assert(std::fabs(source.height - 720.0f) < 0.01f);

    const FloatRect clipped = mapModelRectToSource(
        FloatRect{-20.0f, 100.0f, 700.0f, 500.0f}, g);
    assert(clipped.x >= 0.0f && clipped.y >= 0.0f);
    assert(clipped.x + clipped.width <= 1280.0f);
    assert(clipped.y + clipped.height <= 720.0f);
}
```

- [ ] **Step 2: Compile to verify the test fails**

Run:

```bash
g++ -std=c++14 -Wall -Wextra -Werror -Iinclude \
  tests/letterbox_geometry_test.cpp -o build-host-tests/letterbox_geometry_test
```

Expected: compilation fails because `letterbox_geometry.hpp` does not exist.

- [ ] **Step 3: Implement the pure geometry helper**

```cpp
struct FloatRect { float x; float y; float width; float height; };

struct LetterboxGeometry {
    int sourceWidth;
    int sourceHeight;
    int modelWidth;
    int modelHeight;
    int resizedWidth;
    int resizedHeight;
    int padLeft;
    int padTop;
    float scale;
};
```

Calculate `scale = min(modelWidth/sourceWidth, modelHeight/sourceHeight)`, round
the resized dimensions once, split any odd padding deterministically, intersect
model rectangles with the unpadded content rectangle, then subtract padding,
divide by `scale`, and clamp to source bounds.

- [ ] **Step 4: Recompile and run the test**

Expected: exit code 0 with `-Werror`.

- [ ] **Step 5: Register the test in CMake and commit only new files if safe**

```cmake
add_executable(letterbox_geometry_test tests/letterbox_geometry_test.cpp)
target_include_directories(letterbox_geometry_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_test(NAME letterbox_geometry_test COMMAND letterbox_geometry_test)
```

Before committing, run `git diff --cached --name-only`. Do not stage the already
dirty `CMakeLists.txt` unless its pre-existing changes have been reviewed and
are intentionally included.

---

### Task 2: Apply Letterbox in YOLOv8-Seg Inference

**Files:**
- Modify: `src/rkYolov5s.cc`
- Test: `tests/letterbox_geometry_test.cpp`

**Interfaces:**
- Consumes: `makeLetterboxGeometry()` and `mapModelRectToSource()` from Task 1.
- Preserves: `cv::Mat rkYolov5s::infer(cv::Mat &orig_img)`.
- Produces: annotations aligned to the original 1280x720 BGR frame.

- [ ] **Step 1: Extend the geometry test for a real detection rectangle**

Add a model-space box `(160, 230, 320, 180)` and assert it maps to source-space
`(320, 180, 640, 360)`. Add a box wholly inside top padding and assert its
mapped width and height are zero.

- [ ] **Step 2: Run the test and verify the padding-only case fails**

Expected: FAIL until the helper explicitly returns an empty rectangle when the
box does not intersect the 640x360 content area.

- [ ] **Step 3: Replace direct stretch preprocessing**

In `rkYolov5s::infer`:

```cpp
const LetterboxGeometry geometry = makeLetterboxGeometry(
    orig_img.cols, orig_img.rows, width, height);

cv::Mat resizedRgb(geometry.resizedHeight,
                   geometry.resizedWidth,
                   CV_8UC3);
resize_rga(src, dst, rgb, resizedRgb, resizedRgb.size());

cv::Mat modelInput(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
resizedRgb.copyTo(modelInput(cv::Rect(
    geometry.padLeft,
    geometry.padTop,
    geometry.resizedWidth,
    geometry.resizedHeight)));
inputs[0].buf = modelInput.data;
```

Keep `modelInput` alive through `rknn_inputs_set()` and `rknn_run()`.

- [ ] **Step 4: Reverse-map boxes and masks**

Intersect every candidate model box with the unpadded content rectangle before
storing it. Use `mapModelRectToSource()` for the source box. Use that same
clipped model box for the prototype-mask crop, then resize the crop to the
mapped source rectangle. Continue drawing on `orig_img`, not on `modelInput`.

- [ ] **Step 5: Run host tests and inspect the target diff**

Run `letterbox_geometry_test`; then confirm there is no remaining comment or
calculation describing independent `scale_w`/`scale_h` stretching in
`rkYolov5s::infer`.

---

### Task 3: TCP-JPEG Pipeline and Process Protocol

**Files:**
- Create: `include/camera_pipeline.hpp`
- Create: `include/video_publisher_pipeline.hpp`
- Create: `include/monitor_protocol.hpp`
- Create: `tests/camera_pipeline_test.cpp`
- Create: `tests/video_publisher_pipeline_test.cpp`
- Create: `tests/monitor_protocol_test.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `buildCameraPipelineDescription(device, width, height, fps)`.
- Produces: `buildTcpJpegPublisherPipeline(quality, host, port)`.
- Produces: `makeReadyStatus`, `makeMetricsStatus`, and `makeErrorStatus`.
- Consumed by: `GstCamera`, `TcpJpegPublisher`, and `main()` in Task 4.

- [ ] **Step 1: Write failing exact-string tests**

Assert the camera description contains all of:

```text
v4l2src device=/dev/video41
image/jpeg,width=1280,height=720,framerate=30/1
queue max-size-buffers=1 leaky=downstream
video/x-raw,format=BGR,width=1280,height=720
appsink name=camera_sink sync=false drop=true max-buffers=1
```

Assert the publisher contains `appsrc name=video_source is-live=true
block=false`, a one-buffer downstream-leaky queue, `jpegenc quality=80`, and
`tcpserversink host=127.0.0.1 port=5000 sync=false`.

Assert protocol output equals:

```text
@status {"state":"ready","width":1280,"height":720,"fps":30,"port":5000}
@metrics {"capture_fps":30.00,"pipeline_fps":12.50,"latency_ms":84.20}
```

Also assert quotes, backslashes, newlines, and control bytes are JSON-escaped in
`makeErrorStatus()`.

- [ ] **Step 2: Compile all three tests and verify missing-header failures**

Use `g++ -std=c++14 -Wall -Wextra -Werror -Iinclude` for each test.

- [ ] **Step 3: Implement minimal header-only builders**

Reject empty devices/hosts, non-positive dimensions/FPS, JPEG quality outside
1..100, and ports outside 1..65535 with `std::invalid_argument`. Format metrics
with fixed two-decimal precision and flush them in the caller.

- [ ] **Step 4: Recompile and run all tests**

Expected: three exit codes of 0 and no compiler warnings.

- [ ] **Step 5: Register tests in CTest**

Each test must include only pure C++ headers so it can run on the host without
RKNN, OpenCV, or GStreamer development libraries.

---

### Task 4: TCP Publisher and Single-Model Runtime

**Files:**
- Create: `include/tcp_jpeg_publisher.hpp`
- Create: `src/tcp_jpeg_publisher.cpp`
- Modify: `src/main.cc`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `bool TcpJpegPublisher::open(int width, int height, int fps, int quality, const std::string& host, int port)`.
- Produces: `bool TcpJpegPublisher::push(const cv::Mat& frame)`, `void close()`, and `const std::string& lastError() const`.
- Command line becomes `rknn_yolov8_demo <model.rknn> [camera_device] [thread_num]`.

- [ ] **Step 1: Add publisher interface and compile the pipeline tests first**

```cpp
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
```

- [ ] **Step 2: Implement bounded appsrc publication**

Set BGR caps to 1280x720 at 30/1, `GST_APP_STREAM_TYPE_STREAM`, and at most two
raw frames of appsrc bytes. Copy a continuous `CV_8UC3` frame into a fresh
`GstBuffer`; reject wrong type/size; treat non-`GST_FLOW_OK` as an error. Make
`close()` idempotent and release appsrc before the pipeline reference.

- [ ] **Step 3: Replace `GstDisplay` with `TcpJpegPublisher`**

Use constants:

```cpp
constexpr int kFrameWidth = 1280;
constexpr int kFrameHeight = 720;
constexpr int kCameraFps = 30;
constexpr const char* kDefaultCamera = "/dev/video41";
constexpr int kThreadCount = 12;
constexpr int kJpegQuality = 80;
constexpr const char* kTcpHost = "127.0.0.1";
constexpr int kTcpPort = 5000;
```

Remove the display-sink argument and all `autovideosink`/`waylandsink` code.
Open the publisher before emitting ready. Any model, camera, or publisher
startup failure must emit `@error` and exit nonzero.

- [ ] **Step 4: Add termination and metrics**

Install SIGINT/SIGTERM handlers that only set a `volatile sig_atomic_t` flag.
Track one capture timestamp per submitted future in FIFO order; when its result
is retrieved, calculate capture-to-output latency. Every second emit and flush:

```cpp
writeProtocolLine(makeMetricsStatus(captureFps, pipelineFps, latencyMs));
```

Draw the current FPS into the 1280x720 result before publishing it. On stop,
stop camera submission, retrieve outstanding futures, close publisher and
camera, then exit.

- [ ] **Step 5: Update the target and build on Linux**

Add `src/tcp_jpeg_publisher.cpp` to `rknn_yolov5_demo`. Build with
`./build-linux_RK3588.sh`. Expected installed executable:

```text
install/rknn_yolov5_demo_Linux/rknn_yolov8_demo
```

Do not claim the RKNN build passed until the Linux command exits 0.

---

### Task 5: Qt Launch Specs and `single_model.ini`

**Files:**
- Create: `D:/codex/InsulatorMonitor/inferenceprofile.h`
- Create: `D:/codex/InsulatorMonitor/inferencelaunchspec.h`
- Create: `D:/codex/InsulatorMonitor/inferencelaunchspec.cpp`
- Create: `D:/codex/InsulatorMonitor/single_model.ini`
- Create: `D:/codex/InsulatorMonitor/tests/inferencelaunchspec_test.cpp`
- Modify: `D:/codex/InsulatorMonitor/CMakeLists.txt`

**Interfaces:**
- Produces: `enum class InferenceProfile { DualModel, SingleModel };`.
- Produces: `LaunchSpecResult buildInferenceLaunchSpec(InferenceProfile profile, const QString& applicationDirectory)`.
- Produces: `InferenceLaunchSpec { QString workingDirectory; QString program; QStringList arguments; }`.
- Consumed by: `InferenceController::startDetection()` in Task 7.

- [ ] **Step 1: Write failing QtTest cases**

Use `QTemporaryDir` to create an executable test file, readable model, and INI.
Test:

- relative program/model resolve against `rknn_dir`;
- absolute program/model remain absolute;
- missing INI, key, executable permission, or model returns a Chinese error;
- reloading after changing `model=` returns the new absolute path;
- dual profile keeps `INSULATOR_RKNN_DIR`, two current models,
  `/dev/video41`, and `mjpg`.

- [ ] **Step 2: Configure the Qt test and verify RED**

```cmake
find_package(Qt5 5.15.8 REQUIRED COMPONENTS Core Gui Widgets Test)
add_executable(inferencelaunchspec_test
    tests/inferencelaunchspec_test.cpp inferencelaunchspec.cpp)
target_link_libraries(inferencelaunchspec_test PRIVATE Qt5::Core Qt5::Test)
add_test(NAME inferencelaunchspec_test COMMAND inferencelaunchspec_test)
```

Expected: compile fails because the types/functions do not exist.

- [ ] **Step 3: Implement single-profile loading**

Read `<applicationDirectory>/single_model.ini` with `QSettings::IniFormat` on
every call. The installed template is:

```ini
[Inference]
rknn_dir=/root/rknn-single
program=rknn_yolov8_demo
model=model/RK3588/best_yolov8_fp16.rknn
```

Resolve relative values against an absolute normalized `rknn_dir`. Return
arguments in this exact order: absolute model, `/dev/video41`, `12`.

- [ ] **Step 4: Implement the dual profile without behavior changes**

Use `INSULATOR_RKNN_DIR`, falling back to `applicationDirectory`; use
`insulator_pipeline`, `mixdet_fp16.rknn`, `v8n-seg.rknn`, `/dev/video41`, and
`mjpg` exactly as the current controller does.

- [ ] **Step 5: Run the Qt test**

Expected: every case passes. Because the Qt directory is not a Git repository,
do not initialize one or create commits without explicit user permission.

---

### Task 6: Pure Qt Lifecycle Tests

**Files:**
- Create: `D:/codex/InsulatorMonitor/inferencelifecycle.h`
- Create: `D:/codex/InsulatorMonitor/tests/inferencelifecycle_test.cpp`
- Modify: `D:/codex/InsulatorMonitor/CMakeLists.txt`

**Interfaces:**
- Produces: `InferenceLifecycle::State` with Idle, Starting, Running, Stopping, Error.
- Produces: `requestStart`, `markFirstFrame`, `requestStop`, `markFailure`, `markProcessExited`, and `takeQueuedRestart`.
- Consumed by: `InferenceController` in Task 7.

- [ ] **Step 1: Write the failing transition test**

Cover these exact sequences:

```text
Idle -> requestStart -> Starting -> markFirstFrame -> Running
Running -> requestStop -> Stopping -> markProcessExited -> Idle
Starting -> markFailure -> Error
Error + process alive -> requestStart -> queued restart
Error -> markProcessExited -> takeQueuedRestart(true once, then false)
Starting -> second requestStart rejected
```

- [ ] **Step 2: Compile and verify missing-header failure**

The test is header-only C++14 and does not need Qt or GStreamer.

- [ ] **Step 3: Implement the minimal deterministic state machine**

State transitions must not create processes or timers. `requestStart` returns
an action enum (`StartNow`, `QueueAfterCleanup`, `Ignore`) so the controller can
perform side effects while the state machine remains unit-testable.

- [ ] **Step 4: Run the test with `-Werror` and register it in CTest**

Expected: exit code 0.

---

### Task 7: Two Qt Executables over Shared Application Code

**Files:**
- Create: `D:/codex/InsulatorMonitor/monitorapplication.h`
- Create: `D:/codex/InsulatorMonitor/monitorapplication.cpp`
- Create: `D:/codex/InsulatorMonitor/main_single.cpp`
- Modify: `D:/codex/InsulatorMonitor/main.cpp`
- Modify: `D:/codex/InsulatorMonitor/inferencecontroller.h`
- Modify: `D:/codex/InsulatorMonitor/inferencecontroller.cpp`
- Modify: `D:/codex/InsulatorMonitor/CMakeLists.txt`
- Create: `D:/codex/InsulatorMonitor/InsulatorMonitorSingle.pro`

**Interfaces:**
- Produces: `int runMonitorApplication(int argc, char* argv[], InferenceProfile profile)`.
- Changes: `InferenceController(InferenceProfile profile, QObject* parent = nullptr)`.
- Consumes: launch spec and lifecycle interfaces from Tasks 5 and 6.

- [ ] **Step 1: Move common startup wiring without changing behavior**

Move QApplication creation, GStreamer initialization, stylesheet loading,
MainWindow/controller construction, all signal connections, fullscreen display,
and `app.exec()` into `runMonitorApplication`. Keep the current frame, state,
log, metrics, start, and stop connections byte-for-byte equivalent.

- [ ] **Step 2: Make both entry points minimal**

```cpp
// main.cpp
return runMonitorApplication(argc, argv, InferenceProfile::DualModel);

// main_single.cpp
return runMonitorApplication(argc, argv, InferenceProfile::SingleModel);
```

- [ ] **Step 3: Refactor controller launch logic**

Store the profile. On every accepted Start request, call
`buildInferenceLaunchSpec(profile, applicationDirPath())`. If it fails, enter
Error with its exact message. Otherwise set QProcess working directory, program,
and arguments from the returned spec. Preserve the 30-second startup timer,
five-second stop timer, three-second frame watchdog, ready-before-TCP rule,
queued restart, and existing UI messages.

- [ ] **Step 4: Build two CMake targets**

```cmake
set(MONITOR_COMMON_SOURCES
    monitorapplication.cpp monitorapplication.h
    mainwindow.cpp mainwindow.h mainwindow.ui
    inferencecontroller.cpp inferencecontroller.h
    inferencelaunchspec.cpp inferencelaunchspec.h inferenceprofile.h
    inferencelifecycle.h
    gstvideoreceiver.cpp gstvideoreceiver.h
    resources.qrc)

add_executable(InsulatorMonitor main.cpp ${MONITOR_COMMON_SOURCES})
add_executable(InsulatorMonitorSingle main_single.cpp ${MONITOR_COMMON_SOURCES})
```

Apply the same Qt and GStreamer libraries and the same `bin` runtime directory
to both targets. Copy `single_model.ini` to `${CMAKE_BINARY_DIR}/bin` and install
it beside `InsulatorMonitorSingle`.

- [ ] **Step 5: Add qmake support**

Create `InsulatorMonitorSingle.pro` with `TARGET = InsulatorMonitorSingle`,
`main_single.cpp`, and the same common sources, headers, forms, resources, and
GStreamer pkg-config modules as the CMake single target.

- [ ] **Step 6: Run host tests and Linux Qt build**

Expected build outputs:

```text
build-rk3588/bin/InsulatorMonitor
build-rk3588/bin/InsulatorMonitorSingle
build-rk3588/bin/single_model.ini
```

Do not claim the complete Qt build passed until the Linux CMake build exits 0.

---

### Task 8: Documentation and RK3588 End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `D:/codex/InsulatorMonitor/README.md`

**Interfaces:**
- Documents the delivered command lines, paths, protocol, and recovery steps.

- [ ] **Step 1: Update the single-model README**

Document the new command:

```bash
./rknn_yolov8_demo \
  model/RK3588/best_yolov8_fp16.rknn \
  /dev/video41 \
  12
```

Document 1280x720 MJPG capture, 640x640 letterbox, loopback TCP port 5000,
required plugins, and the `@status/@metrics/@error` lines.

- [ ] **Step 2: Update the Qt README**

Document both binaries, `single_model.ini`, relative path rules, configuration
reload on every Start, the prohibition on simultaneous detection, and exact
deployment layout.

- [ ] **Step 3: Run all host-pure tests fresh**

Compile from current sources with `-std=c++14 -Wall -Wextra -Werror`, then run:

```text
letterbox_geometry_test
camera_pipeline_test
video_publisher_pipeline_test
monitor_protocol_test
inferencelifecycle_test
latest_value_slot_test
inferencelaunchspec_test
```

Report the exact pass count and any skipped platform tests.

- [ ] **Step 4: Build both projects in Linux**

Run `./build-linux_RK3588.sh` in the single-model repository. Configure and
build the Qt project with the RK3588 toolchain and Qt5 CMake directory. Record
both exit codes and output binary paths.

- [ ] **Step 5: Verify camera and plugins on RK3588**

```bash
v4l2-ctl -d /dev/video41 --set-fmt-video=width=1280,height=720,pixelformat=MJPG
v4l2-ctl -d /dev/video41 --set-parm=30

for e in appsrc appsink queue videoconvert jpegenc jpegparse jpegdec \
         tcpserversink tcpclientsrc; do
  gst-inspect-1.0 "$e" >/dev/null 2>&1 || echo "$e: MISSING"
done
```

- [ ] **Step 6: Verify manual TCP stream**

Start `rknn_yolov8_demo` and confirm one ready line. In another terminal run:

```bash
gst-launch-1.0 tcpclientsrc host=127.0.0.1 port=5000 \
  ! jpegparse ! jpegdec ! videoconvert ! autovideosink sync=false
```

Confirm 1280x720 video, aligned boxes/masks, visible class/confidence/FPS, and
no steadily increasing delay.

- [ ] **Step 7: Verify `InsulatorMonitorSingle`**

Place `single_model.ini` beside the binary, click Start, observe Starting then
Running after the first frame, observe FPS/latency updates, click Stop, and
confirm the child exits. Change only the `model=` line to a second compatible
model and confirm the next Start uses it.

- [ ] **Step 8: Verify failure recovery**

Test missing INI, missing model, camera already open, and port 5000 already in
use. Each must produce a visible error and allow a later manual restart after
the resource or configuration is corrected.

- [ ] **Step 9: Final diff and completion check**

Run `git diff --check` in the RKNN repository, inspect every modified target
file, and list the untracked/non-Git Qt files explicitly. Do not merge, push,
delete, or overwrite unrelated dirty work.
