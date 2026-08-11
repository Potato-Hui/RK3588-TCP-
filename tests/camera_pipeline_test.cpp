#include "camera_pipeline.hpp"
#include <cassert>
#include <string>

int main()
{
    const std::string pipeline =
        buildCameraPipelineDescription("/dev/video41", 1280, 720, 30);
    assert(pipeline.find("image/jpeg,width=1280,height=720,framerate=30/1") != std::string::npos);
    assert(pipeline.find("queue max-size-buffers=1 leaky=downstream") != std::string::npos);
    assert(pipeline.find("video/x-raw,format=BGR,width=1280,height=720") != std::string::npos);
    assert(pipeline.find("appsink name=camera_sink sync=false drop=true max-buffers=1") != std::string::npos);
}
