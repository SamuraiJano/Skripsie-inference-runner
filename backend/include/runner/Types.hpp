#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace runner {

namespace fs = std::filesystem;
using json = nlohmann::json;

enum class ModelTask {
    Classification,
    YoloDetect,
    Raw,
};

struct ModelConfig {
    std::string id;
    std::string name;
    ModelTask task = ModelTask::YoloDetect;
    int input_width = 640;
    int input_height = 640;
    std::string input_blob = "in0";
    std::vector<std::string> output_blobs = {"out0"};
    std::array<float, 3> mean = {0.0F, 0.0F, 0.0F};
    std::array<float, 3> norm = {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F};
    std::vector<std::string> labels;
    int threads = 4;
    int top_k = 5;
    int max_detections = 300;
    bool apply_softmax = false;
    std::string created_at;

    static ModelConfig from_json(const json& value);
    json to_json() const;
    json to_public_json(bool active, std::uintmax_t size_bytes) const;
};

struct Detection {
    int class_id = 0;
    std::string label;
    float confidence = 0.0F;
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

struct ClassificationScore {
    int class_id = 0;
    std::string label;
    float score = 0.0F;
};

struct RawTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> values;
    std::size_t total_values = 0;
};

struct Timing {
    double decode_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

struct InferenceResult {
    std::string model_id;
    ModelTask task = ModelTask::Raw;
    int source_width = 0;
    int source_height = 0;
    std::vector<Detection> detections;
    std::vector<ClassificationScore> classifications;
    std::vector<RawTensor> tensors;
    Timing timing;

    json to_json() const;
};

std::string task_name(ModelTask task);
std::string utc_now_iso8601();
std::string make_model_id(const std::string& name);
void validate_model_config(const ModelConfig& config);

}  // namespace runner

