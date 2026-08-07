# Single-Model RKNN and Qt Integration Design

## Goal

Add a second Qt executable, `InsulatorMonitorSingle`, that starts the
single-model RKNN application in `rknn-cpp-Multithreading` and displays its
annotated video through the same local TCP-JPEG and process-status protocol as
the existing dual-model monitor.

The existing `InsulatorMonitor` executable remains the fixed dual-model
monitor. Both executables are built from one Qt source tree so the UI, theme,
video receiver, and latency fixes remain shared.

## Confirmed Constraints

- Platform: RK3588 Linux, Qt 5 and C++14.
- Camera: `/dev/video41`, MJPG, 1280x720 at 30 FPS.
- Model input: 640x640.
- Future model replacement keeps the same YOLOv8-Seg input/output structure;
  only the RKNN file changes.
- The single-model Qt program reads its model and executable paths from an
  external `single_model.ini` file.
- The RKNN process draws masks, boxes, class names, confidence, and FPS into
  the final frame. Qt displays that final frame.
- TCP binds only to `127.0.0.1:5000`.
- The single- and dual-model programs are not expected to run detection at the
  same time because they share one camera and one TCP port.

## Qt Executables and Shared Components

The Qt CMake project produces two binaries:

```text
InsulatorMonitor        fixed dual-model launcher
InsulatorMonitorSingle  configurable single-model launcher
```

They share these components:

- `MainWindow` and the existing `.ui` file;
- the white theme and resources;
- `GstVideoReceiver` and its latest-frame-only delivery slot;
- start, stop, timeout, watchdog, process-log, and UI-state behavior;
- parsing of `@status`, `@metrics`, and `@error` JSON-line messages.

The launch configuration is kept separate from the common process lifecycle.
The dual-model entry point supplies its current fixed executable, two models,
camera, and format. The single-model entry point reloads `single_model.ini`
whenever the user clicks Start and supplies the resulting executable, one
model, `/dev/video41`, and the fixed worker count.

This avoids copying the complete Qt project and permits future UI and
GStreamer fixes to affect both binaries.

## `single_model.ini`

The file is installed beside `InsulatorMonitorSingle` and has this format:

```ini
[Inference]
rknn_dir=/root/rknn-single
program=rknn_yolov8_demo
model=model/RK3588/best_yolov8_fp16.rknn
```

Relative `program` and `model` values resolve against `rknn_dir`. Absolute
values remain absolute. The file is read again for every Start request, so a
user can stop detection, edit the model path, and start again without
recompiling or restarting Qt.

Before starting the process, Qt validates that:

- the INI file exists and all three values are non-empty;
- `rknn_dir` exists;
- the RKNN program exists and is executable;
- the model exists and is readable.

Validation errors are shown in the existing run-status area and do not create
a child process.

## Single-Model Video Pipeline

The single-model camera pipeline becomes:

```text
v4l2src device=/dev/video41
  ! image/jpeg,width=1280,height=720,framerate=30/1
  ! queue max-size-buffers=1 leaky=downstream
  ! jpegdec
  ! videoconvert
  ! video/x-raw,format=BGR,width=1280,height=720
  ! appsink drop=true max-buffers=1 sync=false
```

Each 1280x720 BGR source frame is letterboxed for the 640x640 model:

```text
source 1280x720
  -> uniform scale 0.5
  -> resized content 640x360
  -> top padding 140, bottom padding 140
  -> model input 640x640
```

Post-processing removes the padding and divides coordinates by the uniform
scale before clamping them to the 1280x720 source image. Segmentation masks are
cropped out of the padded model area, resized to the mapped source rectangle,
and blended into the original 1280x720 frame. This preserves the camera aspect
ratio instead of stretching a 16:9 image into a square.

The annotated frame is published with:

```text
appsrc is-live=true block=false format=time do-timestamp=true
  ! queue max-size-buffers=1 leaky=downstream
  ! videoconvert
  ! jpegenc quality=80
  ! tcpserversink host=127.0.0.1 port=5000 sync=false
```

Both producer and receiver retain only the newest frame. This prevents old
frames from accumulating when JPEG encoding, decoding, or Qt painting is
temporarily slower than inference.

## Process Protocol and Data Flow

The single-model executable no longer creates `autovideosink` or
`waylandsink`. Its command line is:

```text
rknn_yolov8_demo <model.rknn> [camera_device] [thread_num]
```

The fixed Qt invocation uses `/dev/video41` and the existing default worker
count of 12.

Startup flow:

1. The user clicks Start in `InsulatorMonitorSingle`.
2. Qt reloads and validates `single_model.ini`.
3. Qt starts the RKNN program with `QProcess` and a working directory of
   `rknn_dir`.
4. RKNN initializes GStreamer, the model pool, camera, and TCP publisher.
5. RKNN writes and flushes a ready line only after all startup stages succeed:

   ```text
   @status {"state":"ready","width":1280,"height":720,"fps":30,"port":5000}
   ```

6. Qt starts its TCP client pipeline after receiving the ready line.
7. The first decoded frame moves the UI from Starting to Running.
8. RKNN emits `@metrics` once per second; Qt displays pipeline FPS and latency.

Initialization and runtime failures use:

```text
@error {"code":"stable_code","message":"human readable message"}
```

All protocol messages are one complete UTF-8 JSON line on standard output and
are flushed immediately. Human diagnostic logs may continue on standard error.

## Stop and Error Handling

- Duplicate Start requests never create duplicate child processes.
- A startup timeout stops the receiver and child process and leaves a visible
  error message.
- A three-second running-frame watchdog reports video interruption and stops
  the child process.
- Stop first requests normal termination. The RKNN program handles SIGTERM,
  stops camera submission, drains or cancels work, closes GStreamer, and exits.
- If it does not exit within five seconds, Qt kills the process.
- If the user requests restart while error cleanup is still running, the
  controller queues one restart after the old process exits.
- Camera-busy and TCP-port-busy errors are reported by the RKNN process and
  displayed by Qt.
- The UI returns to a state in which the user can correct the INI file or
  external resource conflict and click Start again.

## Testing

Host-side tests cover logic that does not require RK3588 hardware:

- INI presence, required keys, and relative/absolute path resolution;
- 1280x720 to 640x640 letterbox dimensions and padding;
- forward and reverse coordinate mapping with boundary clamping;
- TCP-JPEG sender pipeline construction;
- ready, metrics, and error protocol formatting;
- latest-value frame replacement;
- launch-state transitions for start, stop, timeout, error, and queued restart.

RK3588 integration checks cover:

- compilation against Qt 5, GStreamer, OpenCV, RGA, and RKNN runtime;
- `/dev/video41` negotiation at MJPG 1280x720 30 FPS;
- boxes and masks aligning with the source image after letterbox mapping;
- Qt starting and stopping the child process;
- live video, FPS, and latency display;
- bounded latency under a temporarily slow Qt paint path;
- clear errors for missing INI/model, camera conflict, and occupied port 5000.

## Out of Scope

- Running the single- and dual-model detectors simultaneously.
- Supporting a future model with a different tensor or post-processing layout.
- Changing battery, storage, snapshot, or database features.
- Exposing camera, port, resolution, JPEG quality, or worker count in the UI.
