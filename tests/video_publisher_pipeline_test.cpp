#include "video_publisher_pipeline.hpp"
#include <cassert>
#include <string>

int main()
{
    const std::string pipeline =
        buildTcpJpegPublisherPipeline(80, "127.0.0.1", 5000);
    assert(pipeline.find("appsrc name=video_source is-live=true block=false") != std::string::npos);
    assert(pipeline.find("queue max-size-buffers=1 leaky=downstream") != std::string::npos);
    assert(pipeline.find("jpegenc quality=80") != std::string::npos);
    assert(pipeline.find("tcpserversink host=127.0.0.1 port=5000 sync=false") != std::string::npos);
}
