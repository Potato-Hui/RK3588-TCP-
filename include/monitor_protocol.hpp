#ifndef MONITOR_PROTOCOL_HPP
#define MONITOR_PROTOCOL_HPP
#include <iomanip>
#include <sstream>
#include <string>

inline std::string jsonEscape(const std::string& input)
{
    std::ostringstream out;
    for (unsigned char ch : input) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) out << "\\u" << std::hex << std::setw(4)
                               << std::setfill('0') << static_cast<int>(ch)
                               << std::dec << std::setfill(' ');
            else out << static_cast<char>(ch);
        }
    }
    return out.str();
}

inline std::string makeReadyStatus(int w, int h, int fps, int port)
{
    std::ostringstream out;
    out << "@status {\"state\":\"ready\",\"width\":" << w
        << ",\"height\":" << h << ",\"fps\":" << fps
        << ",\"port\":" << port << "}";
    return out.str();
}

inline std::string makeMetricsStatus(double capture, double pipeline, double latency)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << "@metrics {\"capture_fps\":" << capture
        << ",\"pipeline_fps\":" << pipeline
        << ",\"latency_ms\":" << latency << "}";
    return out.str();
}

inline std::string makeErrorStatus(const std::string& code, const std::string& message)
{
    return "@error {\"code\":\"" + jsonEscape(code) +
           "\",\"message\":\"" + jsonEscape(message) + "\"}";
}
#endif
