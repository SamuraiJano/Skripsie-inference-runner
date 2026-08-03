#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "runner/Types.hpp"

namespace runner {

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual InferenceResult infer(const std::vector<unsigned char>& encoded_image, float confidence, float iou) = 0;
    virtual nlohmann::json metadata() const { return nlohmann::json::object(); }
};

struct EngineRuntimeInfo {
    bool ncnn_available = false;
    bool vulkan_compiled = false;
    int gpu_count = 0;
    int default_gpu_index = -1;
    std::string default_gpu_name;
};

std::unique_ptr<InferenceEngine> create_engine(
    const ModelConfig& config,
    const std::filesystem::path& param_path,
    const std::filesystem::path& bin_path
);

std::unique_ptr<InferenceEngine> create_ncnn_engine(
    const ModelConfig& config,
    const std::filesystem::path& param_path,
    const std::filesystem::path& bin_path
);

std::unique_ptr<InferenceEngine> create_python_engine(
    const ModelConfig& config,
    const std::filesystem::path& artifact_path
);

bool ncnn_available();
EngineRuntimeInfo engine_runtime_info();
std::string engine_name();
nlohmann::json python_backend_health();
nlohmann::json model_format_capabilities();

}  // namespace runner
