#include "runner/Types.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace runner {
namespace {

ModelTask parse_task(const std::string& value) {
    if (value == "classification") return ModelTask::Classification;
    if (value == "yolo-detect") return ModelTask::YoloDetect;
    if (value == "raw") return ModelTask::Raw;
    throw std::invalid_argument("task must be classification, yolo-detect, or raw");
}

std::array<float, 3> parse_triplet(const json& value, const std::array<float, 3>& fallback) {
    if (!value.is_array()) return fallback;
    if (value.size() != 3) throw std::invalid_argument("mean and norm must contain three values");
    return {value.at(0).get<float>(), value.at(1).get<float>(), value.at(2).get<float>()};
}

}  // namespace

std::string task_name(ModelTask task) {
    switch (task) {
        case ModelTask::Classification: return "classification";
        case ModelTask::YoloDetect: return "yolo-detect";
        case ModelTask::Raw: return "raw";
    }
    return "raw";
}

ModelConfig ModelConfig::from_json(const json& value) {
    ModelConfig config;
    config.id = value.value("id", "");
    config.name = value.value("name", "");
    config.task = parse_task(value.value("task", "yolo-detect"));
    config.created_at = value.value("createdAt", "");

    if (value.contains("input")) {
        const auto& input = value.at("input");
        config.input_width = input.value("width", 640);
        config.input_height = input.value("height", 640);
        config.input_blob = input.value("blob", "in0");
        config.mean = parse_triplet(input.value("mean", json::array()), config.mean);
        config.norm = parse_triplet(input.value("norm", json::array()), config.norm);
    }

    if (value.contains("outputs")) config.output_blobs = value.at("outputs").get<std::vector<std::string>>();
    if (value.contains("labels")) config.labels = value.at("labels").get<std::vector<std::string>>();
    if (value.contains("runtime")) config.threads = value.at("runtime").value("threads", 4);
    if (value.contains("classification")) {
        const auto& classification = value.at("classification");
        config.apply_softmax = classification.value("applySoftmax", false);
        config.top_k = classification.value("topK", 5);
    }
    config.max_detections = value.value("maxDetections", 300);
    validate_model_config(config);
    return config;
}

json ModelConfig::to_json() const {
    return {
        {"id", id},
        {"name", name},
        {"task", task_name(task)},
        {"input", {
            {"width", input_width}, {"height", input_height}, {"blob", input_blob},
            {"mean", mean}, {"norm", norm}
        }},
        {"outputs", output_blobs},
        {"labels", labels},
        {"runtime", {{"threads", threads}}},
        {"classification", {{"applySoftmax", apply_softmax}, {"topK", top_k}}},
        {"maxDetections", max_detections},
        {"createdAt", created_at},
    };
}

json ModelConfig::to_public_json(bool active, std::uintmax_t size_bytes) const {
    return {
        {"id", id}, {"name", name}, {"task", task_name(task)},
        {"inputWidth", input_width}, {"inputHeight", input_height},
        {"inputBlob", input_blob}, {"outputBlobs", output_blobs},
        {"threads", threads}, {"labels", labels}, {"active", active},
        {"sizeBytes", size_bytes}, {"createdAt", created_at},
    };
}

json InferenceResult::to_json() const {
    json detection_values = json::array();
    for (const auto& detection : detections) {
        detection_values.push_back({
            {"classId", detection.class_id}, {"label", detection.label}, {"confidence", detection.confidence},
            {"x", detection.x}, {"y", detection.y}, {"width", detection.width}, {"height", detection.height},
        });
    }

    json classification_values = json::array();
    for (const auto& classification : classifications) {
        classification_values.push_back({
            {"classId", classification.class_id}, {"label", classification.label}, {"score", classification.score},
        });
    }

    json tensor_values = json::array();
    for (const auto& tensor : tensors) {
        tensor_values.push_back({
            {"name", tensor.name}, {"shape", tensor.shape}, {"values", tensor.values}, {"totalValues", tensor.total_values},
        });
    }

    return {
        {"modelId", model_id}, {"task", task_name(task)},
        {"sourceWidth", source_width}, {"sourceHeight", source_height},
        {"detections", std::move(detection_values)},
        {"classifications", std::move(classification_values)},
        {"tensors", std::move(tensor_values)},
        {"timing", {
            {"decodeMs", timing.decode_ms}, {"preprocessMs", timing.preprocess_ms},
            {"inferenceMs", timing.inference_ms}, {"postprocessMs", timing.postprocess_ms},
            {"totalMs", timing.total_ms},
        }},
    };
}

std::string utc_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string make_model_id(const std::string& name) {
    std::string slug;
    slug.reserve(name.size());
    bool last_dash = false;
    for (unsigned char ch : name) {
        if (std::isalnum(ch)) {
            slug.push_back(static_cast<char>(std::tolower(ch)));
            last_dash = false;
        } else if (!last_dash && !slug.empty()) {
            slug.push_back('-');
            last_dash = true;
        }
    }
    while (!slug.empty() && slug.back() == '-') slug.pop_back();
    if (slug.empty()) slug = "model";
    if (slug.size() > 40) slug.resize(40);
    const auto ticks = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return slug + "-" + std::to_string(ticks);
}

void validate_model_config(const ModelConfig& config) {
    if (config.name.empty() || config.name.size() > 100) throw std::invalid_argument("model name must be between 1 and 100 characters");
    if (config.input_width < 32 || config.input_width > 4096 || config.input_height < 32 || config.input_height > 4096) {
        throw std::invalid_argument("input dimensions must be between 32 and 4096");
    }
    if (config.input_blob.empty() || config.input_blob.size() > 100) throw std::invalid_argument("input blob is required");
    if (config.output_blobs.empty() || config.output_blobs.size() > 16) throw std::invalid_argument("at least one output blob is required");
    if (config.threads < 1 || config.threads > 4) throw std::invalid_argument("threads must be between 1 and 4 for the Pi 4 profile");
    if (config.top_k < 1 || config.top_k > 100) throw std::invalid_argument("topK must be between 1 and 100");
    if (config.max_detections < 1 || config.max_detections > 1000) throw std::invalid_argument("maxDetections must be between 1 and 1000");
    if (config.task == ModelTask::YoloDetect && config.labels.empty()) throw std::invalid_argument("YOLO detection requires class labels");
    if (config.labels.size() > 10000) throw std::invalid_argument("too many class labels");
}

}  // namespace runner
