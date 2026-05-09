#include <drogon/drogon.h>
#include <spdlog/spdlog.h>
#include "util/config.h"
#include "device/registry.h"

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("RMMS AI Server ONNX v{} (protocol {})", RMMS_ONNX_VERSION, RMMS_PROTOCOL_VERSION);

    auto cfg = rmms::Config::load(argc, argv);

    rmms::DeviceRegistry registry;
    registry.discover();

    auto& app = drogon::app();
    app.addListener(cfg.host, cfg.port);
    app.setThreadNum(cfg.workers);

    spdlog::info("Device backends: {}", registry.summary());
    spdlog::info("Listening on {}:{}", cfg.host, cfg.port);

    app.run();
    return 0;
}
