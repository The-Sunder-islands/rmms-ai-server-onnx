#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <unordered_map>
#include <variant>
#include <glaze/glaze.hpp>

namespace rmms::protocol {

// --- Enums ---
enum class CapabilityStatus { implemented, not_implemented };
enum class TaskStatus { queued, processing, done, partial_error, error, cancelled };
enum class FinalStatus { done, partial_error, error, cancelled };
enum class ParamType { int_t, float_t, string_t, bool_t, enum_t, multi_enum };

// --- Param Schema ---
struct ParamDef {
    std::string key;
    ParamType type;
    std::string label;
    std::string description;
    std::string default_value;
    bool required{true};
    std::optional<float> min_val;
    std::optional<float> max_val;
    std::optional<float> step;
    std::optional<int> decimals;
    std::string group;
    std::vector<std::string> choices;
    std::vector<std::pair<std::string, std::string>> options;
};

// --- Capability ---
struct Capability {
    std::string id;
    std::string label;
    std::string description;
    CapabilityStatus status{CapabilityStatus::implemented};
    std::vector<ParamDef> param_defs;
    std::vector<std::string> models;
    std::optional<std::string> default_model;
};

// --- Pipeline ---
struct StepInput {
    int from_step;
    std::optional<std::string> stem;
};

struct PipelineStep {
    std::string type;
    std::optional<std::string> model;
    std::unordered_map<std::string, std::string> params;
    std::unordered_map<std::string, std::string> model_params;
    std::optional<StepInput> input;
};

// --- Response models ---
struct HealthResponse {
    std::string status{"ok"};
    std::string version;
    double uptime_seconds{0.0};
    std::string model_loaded;
    int active_tasks{0};
    int queued_tasks{0};
};

struct CapabilitiesResponse {
    std::string protocol_version;
    std::string server_version;
    std::vector<Capability> capabilities;
    std::vector<DeviceInfo> devices;
    struct { int max_concurrent_tasks; int max_queue_size; } scheduler;
    std::vector<std::string> output_formats;
    std::vector<std::string> output_packages;
    size_t max_upload_bytes{0};
};

struct TaskSubmitResponse {
    std::string task_id;
    std::string status;
    std::string message;
    std::vector<PipelineStep> pipeline;
    bool cached{false};
    std::string created_at;
};

// --- Error ---
struct ErrorBody {
    std::string code;
    std::string message;
    std::unordered_map<std::string, std::string> details;
};

// --- Track (SSE partial_result) ---
struct Track {
    std::string track_type{"audio"};
    std::string stem;
    std::string label;
    std::string url;
    std::string format;
    std::optional<int> sample_rate;
    std::optional<double> duration;
    std::optional<int64_t> size_bytes;
};

// --- Import DeviceInfo (circular dependency break via forward decl) ---
using DeviceInfo = std::unordered_map<std::string, std::string>; // placeholder

} // namespace rmms::protocol
