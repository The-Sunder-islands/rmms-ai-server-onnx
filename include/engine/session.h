#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <memory>
#include <span>
#include <onnxruntime/onnxruntime_c_api.h>

namespace rmms::engine {

class ORTSession {
public:
    ORTSession(const std::string& model_path,
               const char* ep_name = nullptr,  // e.g. "CUDA", "DML", "OpenCL"
               int intra_threads = 0);          // 0 = auto

    ~ORTSession();

    ORTSession(const ORTSession&) = delete;
    ORTSession& operator=(const ORTSession&) = delete;
    ORTSession(ORTSession&&) noexcept;
    ORTSession& operator=(ORTSession&&) noexcept;

    // Run inference: input_shape = {batch, channels, samples}
    std::vector<std::vector<float>> run(std::span<const float> audio,
                                         std::span<const int64_t> input_shape);

    const OrtApi* api() const { return api_; }
    OrtSession* session() const { return session_; }

private:
    const OrtApi* api_{nullptr};
    OrtEnv* env_{nullptr};
    OrtSessionOptions* opts_{nullptr};
    OrtSession* session_{nullptr};
    OrtMemoryInfo* mem_info_{nullptr};

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    int num_outputs_{0};
};

// Factory: pick best available EP
std::unique_ptr<ORTSession> create_best_session(const std::string& model_path,
                                                  bool prefer_cuda = true,
                                                  bool prefer_dml = false);

} // namespace rmms::engine
