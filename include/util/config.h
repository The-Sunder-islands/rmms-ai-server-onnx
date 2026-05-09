#pragma once
#include <string>
#include <cstdint>

namespace rmms {

constexpr const char* RMMS_ONNX_VERSION   = "0.1.0";
constexpr const char* RMMS_PROTOCOL_VERSION = "1.0.0-alpha";

struct Config {
    std::string host{"0.0.0.0"};
    uint16_t port{8421};
    int workers{4};
    int max_concurrent_tasks{4};
    int max_queue_size{20};
    size_t max_upload_bytes{500 * 1024 * 1024};
    int task_ttl_seconds{3600};
    std::string upload_dir;
    std::string output_dir;
    std::string model_dir;
    bool mdns_enabled{true};
    std::string mdns_name{"RMMS AI Server (ONNX/C++)"};
    bool auth_enabled{false};
    std::string api_key;

    static Config load(int argc, char* argv[]);
};

} // namespace rmms
