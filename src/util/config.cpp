#include "util/config.h"
#include <cstdlib>
#include <string>

namespace rmms {

Config Config::load(int /*argc*/, char* /*argv*/[]) {
    Config cfg;

    // Environment variables (AI_SERVER_ prefix, same as Python version)
    if (auto* v = std::getenv("AI_SERVER_HOST"))  cfg.host = v;
    if (auto* v = std::getenv("AI_SERVER_PORT"))  cfg.port = static_cast<uint16_t>(std::stoi(v));
    if (auto* v = std::getenv("AI_SERVER_API_KEY")) { cfg.auth_enabled = true; cfg.api_key = v; }
    if (auto* v = std::getenv("AI_SERVER_MAX_CONCURRENT_TASKS")) cfg.max_concurrent_tasks = std::stoi(v);
    if (auto* v = std::getenv("AI_SERVER_MAX_UPLOAD_MB")) cfg.max_upload_bytes = std::stoull(v) * 1024 * 1024;
    if (auto* v = std::getenv("AI_SERVER_UPLOAD_DIR"))     cfg.upload_dir = v;
    if (auto* v = std::getenv("AI_SERVER_OUTPUT_DIR"))     cfg.output_dir = v;
    if (auto* v = std::getenv("AI_SERVER_MODEL_CACHE_DIR")) cfg.model_dir = v;

    return cfg;
}

} // namespace rmms
