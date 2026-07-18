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
};

std::unique_ptr<InferenceEngine> create_engine(
    const ModelConfig& config,
    const std::filesystem::path& param_path,
    const std::filesystem::path& bin_path
);

bool ncnn_available();
std::string engine_name();

}  // namespace runner

