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
    if (value == "yolo-segment") return ModelTask::YoloSegment;
    if (value == "raw") return ModelTask::Raw;
    throw std::invalid_argument("task must be classification, yolo-detect, yolo-segment, or raw");
}

ModelPrecision parse_precision(const std::string& value) {
    if (value == "fp32") return ModelPrecision::FP32;
    if (value == "fp16") return ModelPrecision::FP16;
    if (value == "int8") return ModelPrecision::INT8;
    throw std::invalid_argument("precision must be fp32, fp16, or int8");
}

bool is_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isxdigit(ch) != 0;
    });
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
        case ModelTask::YoloSegment: return "yolo-segment";
        case ModelTask::Raw: return "raw";
    }
    return "raw";
}

std::string precision_name(ModelPrecision precision) {
    switch (precision) {
        case ModelPrecision::FP32: return "fp32";
        case ModelPrecision::FP16: return "fp16";
        case ModelPrecision::INT8: return "int8";
    }
    return "fp32";
}

ModelConfig ModelConfig::from_json(const json& value, bool allow_legacy) {
    ModelConfig config;
    config.schema_version = value.value("schemaVersion", 1);
    if (config.schema_version < 2) {
        if (!allow_legacy) {
            throw std::invalid_argument("legacy manifest requires migration to schemaVersion 2 with precision and provenance");
        }
        config.legacy_unverified = true;
    }
    config.id = value.value("id", "");
    config.name = value.value("name", "");
    config.task = parse_task(value.value("task", "yolo-detect"));
    config.created_at = value.value("createdAt", "");
    config.precision = parse_precision(value.value("precision", "fp32"));
    config.framework = value.value("framework", config.legacy_unverified ? "ncnn" : "");
    if (value.contains("artifact")) {
        const auto& artifact = value.at("artifact");
        config.format = artifact.value("format", "ncnn");
        config.artifact_file = artifact.value("file", "");
        config.backend = artifact.value("backend", config.format == "ncnn" ? "ncnn" : "ultralytics");
        config.accelerator = artifact.value("accelerator", config.format == "ncnn" ? "Vulkan" : "automatic");
        config.trusted_checkpoint = artifact.value("trustedCheckpoint", false);
    }
    config.checkpoint_sha256 = value.value("checkpointSha256", "");
    config.dataset_hash = value.value("datasetHash", "");
    config.split_hash = value.value("splitHash", "");
    config.class_schema_hash = value.value("classSchemaHash", "");
    config.calibration_id = value.contains("calibrationId") && !value.at("calibrationId").is_null()
        ? value.at("calibrationId").get<std::string>()
        : "";
    if (value.contains("exporter")) {
        config.exporter_name = value.at("exporter").value("name", "");
        config.exporter_version = value.at("exporter").value("version", "");
    }
    if (value.contains("artifactVerification")) {
        const auto& verification = value.at("artifactVerification");
        config.artifact_verified = verification.value("verified", false);
        config.artifact_quantized = verification.value("quantized", config.precision == ModelPrecision::INT8 && config.artifact_verified);
    }

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
    if (value.contains("runtime")) {
        const auto& runtime = value.at("runtime");
        config.threads = runtime.value("threads", 8);
        config.runtime_name = runtime.value("name", config.legacy_unverified ? "ncnn" : "");
        config.runtime_version = runtime.value("version", "");
        config.use_vulkan_compute = runtime.value("useVulkanCompute", true);
        config.gpu_device = runtime.value("gpuDevice", -1);
        config.use_int8_inference = runtime.value("useInt8Inference", false);
        config.use_int8_storage = runtime.value("useInt8Storage", false);
        config.use_fp16_storage = runtime.value("useFp16Storage", false);
        config.use_fp16_arithmetic = runtime.value("useFp16Arithmetic", false);
    }
    if (value.contains("classification")) {
        const auto& classification = value.at("classification");
        config.apply_softmax = classification.value("applySoftmax", false);
        config.top_k = classification.value("topK", 5);
    }
    config.max_detections = value.value("maxDetections", 300);
    config.mask_threshold = value.value("maskThreshold", 0.5F);
    validate_model_config(config);
    return config;
}

json ModelConfig::to_json() const {
    return {
        {"id", id},
        {"name", name},
        {"task", task_name(task)},
        {"schemaVersion", schema_version},
        {"precision", precision_name(precision)},
        {"framework", framework},
        {"artifact", {
            {"format", format}, {"file", artifact_file}, {"backend", backend},
            {"accelerator", accelerator}, {"trustedCheckpoint", trusted_checkpoint}
        }},
        {"checkpointSha256", checkpoint_sha256},
        {"datasetHash", dataset_hash},
        {"splitHash", split_hash},
        {"classSchemaHash", class_schema_hash},
        {"exporter", {{"name", exporter_name}, {"version", exporter_version}}},
        {"calibrationId", calibration_id.empty() ? json(nullptr) : json(calibration_id)},
        {"input", {
            {"width", input_width}, {"height", input_height}, {"blob", input_blob},
            {"mean", mean}, {"norm", norm}
        }},
        {"outputs", output_blobs},
        {"labels", labels},
        {"runtime", {
            {"name", runtime_name}, {"version", runtime_version}, {"threads", threads},
            {"useVulkanCompute", use_vulkan_compute}, {"gpuDevice", gpu_device},
            {"useInt8Inference", use_int8_inference}, {"useInt8Storage", use_int8_storage},
            {"useFp16Storage", use_fp16_storage}, {"useFp16Arithmetic", use_fp16_arithmetic}
        }},
        {"artifactVerification", {{"verified", artifact_verified}, {"quantized", artifact_quantized}}},
        {"classification", {{"applySoftmax", apply_softmax}, {"topK", top_k}}},
        {"maxDetections", max_detections},
        {"maskThreshold", mask_threshold},
        {"createdAt", created_at},
    };
}

json ModelConfig::to_public_json(bool active, std::uintmax_t size_bytes) const {
    return {
        {"id", id}, {"name", name}, {"task", task_name(task)},
        {"precision", precision_name(precision)}, {"framework", framework},
        {"format", format}, {"artifactFile", artifact_file}, {"backend", backend},
        {"accelerator", accelerator},
        {"checkpointSha256", checkpoint_sha256}, {"datasetHash", dataset_hash},
        {"splitHash", split_hash}, {"classSchemaHash", class_schema_hash},
        {"exporter", {{"name", exporter_name}, {"version", exporter_version}}},
        {"runtime", {
            {"name", runtime_name}, {"version", runtime_version},
            {"useVulkanCompute", use_vulkan_compute}, {"gpuDevice", gpu_device}
        }},
        {"calibrationId", calibration_id.empty() ? json(nullptr) : json(calibration_id)},
        {"legacyUnverified", legacy_unverified},
        {"inputWidth", input_width}, {"inputHeight", input_height},
        {"inputBlob", input_blob}, {"outputBlobs", output_blobs},
        {"threads", threads}, {"labels", labels}, {"active", active},
        {"sizeBytes", size_bytes}, {"createdAt", created_at},
    };
}

json InferenceResult::to_json() const {
    json detection_values = json::array();
    for (const auto& detection : detections) {
        json detection_value = {
            {"classId", detection.class_id}, {"label", detection.label}, {"confidence", detection.confidence},
            {"x", detection.x}, {"y", detection.y}, {"width", detection.width}, {"height", detection.height},
            {"trackId", detection.track_id < 0 ? json(nullptr) : json(detection.track_id)},
        };
        if (!detection.mask.counts.empty()) {
            detection_value["mask"] = {
                {"encoding", "rle-row-major"}, {"width", detection.mask.width},
                {"height", detection.mask.height}, {"counts", detection.mask.counts},
            };
        } else {
            detection_value["mask"] = nullptr;
        }
        detection_values.push_back(std::move(detection_value));
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
            {"inferenceMs", timing.inference_ms},
            {"detectionNmsMs", timing.detection_nms_ms}, {"maskReconstructionMs", timing.mask_reconstruction_ms},
            {"trackingMs", timing.tracking_ms}, {"queueWaitMs", timing.queue_wait_ms},
            {"gradingMs", timing.grading_ms}, {"backendTotalMs", timing.backend_total_ms},
        }},
    };
}

InferenceResult InferenceResult::from_json(const json& value) {
    InferenceResult result;
    result.model_id = value.value("modelId", "");
    result.task = parse_task(value.value("task", "raw"));
    result.source_width = value.value("sourceWidth", 0);
    result.source_height = value.value("sourceHeight", 0);
    for (const auto& item : value.value("detections", json::array())) {
        Detection detection;
        detection.class_id = item.value("classId", 0);
        detection.label = item.value("label", "");
        detection.confidence = item.value("confidence", 0.0F);
        detection.x = item.value("x", 0.0F);
        detection.y = item.value("y", 0.0F);
        detection.width = item.value("width", 0.0F);
        detection.height = item.value("height", 0.0F);
        if (item.contains("trackId") && !item.at("trackId").is_null()) detection.track_id = item.at("trackId").get<int>();
        if (item.contains("mask") && !item.at("mask").is_null()) {
            const auto& mask = item.at("mask");
            detection.mask.width = mask.value("width", 0);
            detection.mask.height = mask.value("height", 0);
            detection.mask.counts = mask.value("counts", std::vector<std::uint32_t>{});
        }
        result.detections.push_back(std::move(detection));
    }
    for (const auto& item : value.value("classifications", json::array())) {
        result.classifications.push_back({
            item.value("classId", 0), item.value("label", ""), item.value("score", 0.0F),
        });
    }
    for (const auto& item : value.value("tensors", json::array())) {
        RawTensor tensor;
        tensor.name = item.value("name", "");
        tensor.shape = item.value("shape", std::vector<int>{});
        tensor.values = item.value("values", std::vector<float>{});
        tensor.total_values = item.value("totalValues", tensor.values.size());
        result.tensors.push_back(std::move(tensor));
    }
    if (value.contains("timing")) {
        const auto& timing = value.at("timing");
        result.timing.decode_ms = timing.value("decodeMs", 0.0);
        result.timing.preprocess_ms = timing.value("preprocessMs", 0.0);
        result.timing.inference_ms = timing.value("inferenceMs", 0.0);
        result.timing.detection_nms_ms = timing.value("detectionNmsMs", 0.0);
        result.timing.mask_reconstruction_ms = timing.value("maskReconstructionMs", 0.0);
        result.timing.tracking_ms = timing.value("trackingMs", 0.0);
        result.timing.queue_wait_ms = timing.value("queueWaitMs", 0.0);
        result.timing.grading_ms = timing.value("gradingMs", 0.0);
        result.timing.backend_total_ms = timing.value("backendTotalMs", 0.0);
    }
    return result;
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
    if (config.threads < 1 || config.threads > 16) throw std::invalid_argument("threads must be between 1 and 16 for the Windows profile");
    if (config.format == "ncnn" && !config.use_vulkan_compute) throw std::invalid_argument("NCNN models require runtime.useVulkanCompute to be true");
    if (config.gpu_device < -1 || config.gpu_device > 15) throw std::invalid_argument("runtime.gpuDevice must be -1 (automatic) or an index between 0 and 15");
    if (config.top_k < 1 || config.top_k > 100) throw std::invalid_argument("topK must be between 1 and 100");
    if (config.max_detections < 1 || config.max_detections > 1000) throw std::invalid_argument("maxDetections must be between 1 and 1000");
    if (config.format == "ncnn" && (config.task == ModelTask::YoloDetect || config.task == ModelTask::YoloSegment) && config.labels.empty()) {
        throw std::invalid_argument("YOLO detection/segmentation requires class labels");
    }
    if (config.format == "ncnn" && config.task == ModelTask::YoloSegment && config.output_blobs.size() < 2) {
        throw std::invalid_argument("YOLO segmentation requires detection and prototype output blobs");
    }
    if (config.format == "ncnn" && config.task == ModelTask::YoloSegment && config.labels != std::vector<std::string>{"class_1", "class_2", "class_3"}) {
        throw std::invalid_argument("YOLO defect segmentation labels must be ordered class_1, class_2, class_3");
    }
    if (config.mask_threshold <= 0.0F || config.mask_threshold >= 1.0F) {
        throw std::invalid_argument("maskThreshold must be between 0 and 1");
    }
    if (config.labels.size() > 10000) throw std::invalid_argument("too many class labels");
    if (!config.legacy_unverified && config.schema_version == 2) {
        if (config.schema_version != 2) throw std::invalid_argument("schemaVersion must be 2");
        if (config.framework != "ncnn") throw std::invalid_argument("this runner imports NCNN manifests only");
        if (config.runtime_name != "ncnn") throw std::invalid_argument("runtime.name must be ncnn");
        if (config.exporter_name.empty() || config.exporter_version.empty() || config.runtime_version.empty()) {
            throw std::invalid_argument("exporter and runtime name/version are required");
        }
        for (const auto& [field, value] : std::vector<std::pair<std::string, std::string>>{
                 {"checkpointSha256", config.checkpoint_sha256}, {"datasetHash", config.dataset_hash},
                 {"splitHash", config.split_hash}, {"classSchemaHash", config.class_schema_hash}}) {
            if (!is_sha256(value)) throw std::invalid_argument(field + " must be a 64-character SHA-256 hex digest");
        }
        if (!config.artifact_verified) throw std::invalid_argument("artifactVerification.verified must be true");
    } else if (config.schema_version == 3) {
        if (config.format.empty() || config.format == "ncnn") throw std::invalid_argument("direct model imports require a non-NCNN artifact format");
        if (config.artifact_file.empty()) throw std::invalid_argument("direct model imports require an artifact file");
        if (config.backend != "ultralytics") throw std::invalid_argument("direct model imports require the Ultralytics backend");
    } else if (!config.legacy_unverified) {
        throw std::invalid_argument("schemaVersion must be 2 for NCNN packages or 3 for direct model artifacts");
    }
    if (config.format != "ncnn") return;
    if (config.precision == ModelPrecision::FP32) {
        if (config.use_int8_inference || config.use_int8_storage || config.use_fp16_storage || config.use_fp16_arithmetic) {
            throw std::invalid_argument("FP32 runtime flags must disable INT8 and FP16 storage/arithmetic");
        }
    } else if (config.precision == ModelPrecision::FP16) {
        if (!config.use_fp16_storage || config.use_int8_inference || config.use_int8_storage) {
            throw std::invalid_argument("FP16 requires FP16 storage and must disable INT8 flags");
        }
    } else if (config.precision == ModelPrecision::INT8) {
        if (config.calibration_id.empty() || !config.artifact_quantized || !config.use_int8_inference || !config.use_int8_storage) {
            throw std::invalid_argument("INT8 requires calibrationId, verified quantization, and explicit INT8 runtime flags");
        }
        if (config.use_fp16_storage || config.use_fp16_arithmetic) {
            throw std::invalid_argument("INT8 must disable FP16 storage/arithmetic flags");
        }
    }
}

}  // namespace runner
