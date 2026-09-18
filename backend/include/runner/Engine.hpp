#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "runner/Types.hpp"

namespace runner {

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual InferenceResult infer(
        const std::vector<unsigned char>& encoded_image,
        float confidence,
        float iou,
        const GroundingRequest& grounding = {}
    ) = 0;
    virtual nlohmann::json metadata() const { return nlohmann::json::object(); }
};

struct EngineRuntimeInfo {
    bool ncnn_available = false;
    bool tflite_compiled = false;
    bool tflite_available = false;
    bool tensorrt_compiled = false;
    bool tensorrt_available = false;
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

std::unique_ptr<InferenceEngine> create_tflite_engine(
    const ModelConfig& config,
    const std::filesystem::path& model_path
);

std::unique_ptr<InferenceEngine> create_tensorrt_engine(
    const ModelConfig& config,
    const std::filesystem::path& model_path
);

bool ncnn_available();
bool tflite_available();
bool tensorrt_available();
EngineRuntimeInfo engine_runtime_info();
std::string engine_name();
nlohmann::json python_backend_health();
nlohmann::json python_surface_mapping(const nlohmann::json& request);
nlohmann::json model_format_capabilities();
nlohmann::json discover_models();

}  // namespace runner
