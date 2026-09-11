#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace runner {

struct DetectorInputSpec {
    std::string name;
    int width = 0;
    int height = 0;
    std::string layout;
    std::string colour_order;
    std::string data_type;
    std::array<float, 3> mean{};
    std::array<float, 3> scale{};
    std::string resize_mode;
    float letterbox_value = 114.0F;
};

struct DetectorOutputSpec {
    std::string name;
    std::string role;
    std::vector<int> shape;
    std::string data_type;
    float quantization_scale = 0.0F;
    int quantization_zero_point = 0;
};

struct DetectorManifest {
    int schema_version = 4;
    std::string id;
    std::string name;
    std::string task;
    std::string model_family;
    std::string backend;
    std::string model_file;
    DetectorInputSpec input;
    std::vector<DetectorOutputSpec> outputs;
    std::string output_schema;
    std::string tensor_orientation;
    std::string box_format;
    std::string coordinates;
    std::string nms;
    float nms_iou = 0.5F;
    bool has_objectness = false;
    float score_floor = 0.10F;
    std::vector<std::string> labels;
    int threads = 4;

    static DetectorManifest from_json(const nlohmann::json& value);
    nlohmann::json to_json() const;
};

void validate_detector_model_extension(const DetectorManifest& manifest, const std::filesystem::path& path);

}  // namespace runner
