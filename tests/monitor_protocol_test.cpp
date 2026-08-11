#include "monitor_protocol.hpp"
#include <cassert>

int main()
{
    assert(makeReadyStatus(1280, 720, 30, 5000) ==
           "@status {\"state\":\"ready\",\"width\":1280,\"height\":720,\"fps\":30,\"port\":5000}");
    assert(makeMetricsStatus(30.0, 12.5, 84.2) ==
           "@metrics {\"capture_fps\":30.00,\"pipeline_fps\":12.50,\"latency_ms\":84.20}");
    assert(makeErrorStatus("bad", "quote \" slash \\ line\n") ==
           "@error {\"code\":\"bad\",\"message\":\"quote \\\" slash \\\\ line\\n\"}");
}
