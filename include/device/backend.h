#pragma once
#include <string>
#include <variant>
#include <vector>
#include <memory>
#include <optional>
#include <cstdint>

namespace rmms {

// --- Forward declarations ---

struct DeviceUnit {
    int device_index{0};
    std::string name;
    std::optional<int64_t> memory_total_mb;
    std::optional<int64_t> memory_used_mb;
};

struct DeviceInfo {
    std::string device_type;
    bool available{false};
    int count{0};
    std::optional<std::string> install_hint;
    std::vector<DeviceUnit> units;
};

struct SectionConfig {
    float threshold_s{300.0f};
    float section_duration_s{300.0f};
    int max_sections{0};
};

// --- Backend concept ---
// Each backend implements: get_info() -> DeviceInfo, get_section_config() -> SectionConfig
// Registered via dynamic library: extern "C" { void* create_backend(); }

// --- Backend variant ---
struct CPUBackend {
    static constexpr const char* type = "cpu";
    DeviceInfo get_info() const;
    SectionConfig get_section_config() const;
};

struct CUDABackend {
    static constexpr const char* type = "cuda";
    DeviceInfo get_info() const;
    SectionConfig get_section_config() const;
};

struct DMLBackend {
    static constexpr const char* type = "dml";
    DeviceInfo get_info() const;
    SectionConfig get_section_config() const;
};

struct OpenCLBackend {
    static constexpr const char* type = "opencl";
    DeviceInfo get_info() const;
    SectionConfig get_section_config() const;
};

using DeviceBackend = std::variant<CPUBackend, CUDABackend, DMLBackend, OpenCLBackend>;

// --- Utils ---
struct DeviceInfo resolve_device_info(const DeviceBackend& b);
SectionConfig resolve_section_config(const DeviceBackend& b);
const char* ep_name(const DeviceBackend& b);
bool is_available(const DeviceBackend& b);

} // namespace rmms
