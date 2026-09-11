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
    ObjectDetection,
    YoloDetect,
    YoloSegment,
    VisionGrounding,
    Raw,
};

enum class ModelPrecision { FP32, FP16, BF16, FP8, INT8, INT4, Unknown };

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
    int threads = 8;
    bool use_vulkan_compute = true;
    int gpu_device = -1;
    int top_k = 5;
    int max_detections = 300;
    bool apply_softmax = false;
    ModelPrecision precision = ModelPrecision::FP32;
    std::string framework;
    std::string format = "ncnn";
    std::string artifact_file;
    std::string backend = "ncnn";
    std::string model_family;
    json detector_manifest = json::object();
    std::string accelerator = "Vulkan";
    bool trusted_checkpoint = false;
    std::string checkpoint_sha256;
    std::string dataset_hash;
    std::string split_hash;
    std::string class_schema_hash;
    std::string exporter_name;
    std::string exporter_version;
    std::string runtime_name;
    std::string runtime_version;
    std::string calibration_id;
    bool artifact_verified = false;
    bool artifact_quantized = false;
    bool use_int8_inference = false;
    bool use_int8_storage = false;
    bool use_fp16_storage = false;
    bool use_fp16_arithmetic = false;
    float mask_threshold = 0.5F;
    int schema_version = 2;
    bool legacy_unverified = false;
    std::string created_at;

    static ModelConfig from_json(const json& value, bool allow_legacy = false);
    json to_json() const;
    json to_public_json(bool active, std::uintmax_t size_bytes) const;
};

struct MaskRle {
    int width = 0;
    int height = 0;
    std::vector<std::uint32_t> counts;
};

struct Detection {
    int class_id = 0;
    std::string label;
    float confidence = 0.0F;
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    int track_id = -1;
    MaskRle mask;
};

struct ClassificationScore {
    int class_id = 0;
    std::string label;
    float score = 0.0F;
};

struct GroundingPoint {
    int class_id = 0;
    std::string label;
    float x = 0.0F;
    float y = 0.0F;
};

struct GroundingRequest {
    std::string prompt;
    std::string mode = "detect";
    std::string generation_mode = "hybrid";
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
    double detection_nms_ms = 0.0;
    double mask_reconstruction_ms = 0.0;
    double tracking_ms = 0.0;
    double queue_wait_ms = 0.0;
    double grading_ms = 0.0;
    double backend_total_ms = 0.0;
};

struct InferenceResult {
    std::string model_id;
    std::string model_family;
    std::string backend;
    ModelTask task = ModelTask::Raw;
    int source_width = 0;
    int source_height = 0;
    std::vector<Detection> detections;
    std::vector<GroundingPoint> points;
    std::vector<ClassificationScore> classifications;
    std::vector<RawTensor> tensors;
    std::string answer;
    Timing timing;

    static InferenceResult from_json(const json& value);
    json to_json() const;
};

std::string task_name(ModelTask task);
std::string precision_name(ModelPrecision precision);
std::string utc_now_iso8601();
std::string make_model_id(const std::string& name);
void validate_model_config(const ModelConfig& config);

}  // namespace runner
