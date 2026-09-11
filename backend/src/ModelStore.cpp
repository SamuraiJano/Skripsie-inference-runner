#include "runner/ModelStore.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <windows.h>
#include <psapi.h>

namespace runner {
namespace {

using Clock = std::chrono::steady_clock;

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path, std::ios::out | std::ios::trunc);
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

bool path_is_within(const fs::path& child, const fs::path& parent) {
    const fs::path resolved_child = fs::weakly_canonical(child);
    const fs::path resolved_parent = fs::weakly_canonical(parent);
    auto child_part = resolved_child.begin();
    for (auto parent_part = resolved_parent.begin(); parent_part != resolved_parent.end(); ++parent_part, ++child_part) {
        if (child_part == resolved_child.end() || *child_part != *parent_part) return false;
    }
    return true;
}

ModelConfig config_from_discovery(const json& item) {
    const std::string format = item.value("format", "");
    const bool ncnn = format == "ncnn";
    const bool quantized = item.value("quantized", false);
    const std::string precision = item.value("precision", "unknown");
    const std::string calibration = quantized && precision == "int8"
        ? item.value("quantizationMethod", "auto-detected") : "";
    const bool executable_checkpoint = format == "pt" || format == "torchscript" ||
        format == "executorch" || format == "locate-anything" || format == "dfine";
    json generated = {
        {"id", item.at("id")},
        {"name", item.at("name")},
        {"schemaVersion", ncnn ? 1 : 3},
        {"task", item.value("task", "raw")},
        {"precision", precision},
        {"framework", item.value("framework", format)},
        {"artifact", {
            {"format", format},
            {"file", item.value("relativePath", fs::path(item.at("path").get<std::string>()).filename().u8string())},
            {"backend", item.value("backend", ncnn ? "ncnn" : "ultralytics")},
            {"accelerator", item.value("accelerator", ncnn ? "Vulkan" : "automatic")},
            {"trustedCheckpoint", executable_checkpoint},
        }},
        {"input", {
            {"width", item.value("inputWidth", 640)}, {"height", item.value("inputHeight", 640)},
            {"blob", item.value("inputBlob", ncnn ? "in0" : "images")},
            {"mean", {0.0F, 0.0F, 0.0F}}, {"norm", {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F}},
        }},
        {"outputs", item.value("outputBlobs", json::array({ncnn ? "out0" : "auto"}))},
        {"labels", item.value("labels", json::array())},
        {"runtime", {
            {"name", ncnn ? "ncnn" : item.value("runtime", format)},
            {"version", "automatic"}, {"threads", 8},
            {"useVulkanCompute", ncnn}, {"gpuDevice", -1},
            {"useInt8Inference", ncnn && quantized}, {"useInt8Storage", ncnn && quantized},
            {"useFp16Storage", false}, {"useFp16Arithmetic", false},
        }},
        {"artifactVerification", {{"verified", item.value("compatible", true)}, {"quantized", quantized}}},
        {"calibrationId", calibration.empty() ? json(nullptr) : json(calibration)},
        {"classification", {{"applySoftmax", false}, {"topK", 5}}},
        {"maxDetections", 300}, {"maskThreshold", 0.5F},
        {"createdAt", item.value("modifiedAt", "")},
    };
    ModelConfig config = ModelConfig::from_json(generated, ncnn);
    config.legacy_unverified = false;
    return config;
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

json statistics(std::vector<double> values) {
    if (values.empty()) return nullptr;
    std::sort(values.begin(), values.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    auto percentile = [&](double value) {
        const double position = value * static_cast<double>(values.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(std::floor(position));
        const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
        const double fraction = position - lower;
        return values[lower] * (1.0 - fraction) + values[upper] * fraction;
    };
    return {{"count", values.size()}, {"mean", mean}, {"median", percentile(0.50)}, {"p95", percentile(0.95)}, {"p99", percentile(0.99)}};
}

std::uint64_t peak_rss_bytes() {
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
    }
    return 0;
}

}  // namespace

ModelStore::ModelStore(fs::path models_root, fs::path state_root)
    : models_root_(fs::absolute(std::move(models_root))), state_root_(fs::absolute(std::move(state_root))) {
    fs::create_directories(models_root_);
    fs::create_directories(state_root_);
    (void)rescan();
    restore_active_model();
}

void ModelStore::restore_active_model() {
    const fs::path active_path = state_root_ / "active-model.txt";
    if (!fs::exists(active_path)) return;
    std::string persisted_active = read_text(active_path);
    persisted_active.erase(std::remove(persisted_active.begin(), persisted_active.end(), '\n'), persisted_active.end());
    persisted_active.erase(std::remove(persisted_active.begin(), persisted_active.end(), '\r'), persisted_active.end());
    if (persisted_active.empty()) return;
    try {
        (void)activate(persisted_active);
    } catch (const std::exception& error) {
        std::cerr << "Could not restore active model '" << persisted_active << "': " << error.what() << '\n';
        persist_active_id();
    }
}

ModelStore::Record* ModelStore::find_record(const std::string& id) {
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const Record& record) { return record.config.id == id; });
    return found == records_.end() ? nullptr : &*found;
}

const ModelStore::Record* ModelStore::find_record(const std::string& id) const {
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const Record& record) { return record.config.id == id; });
    return found == records_.end() ? nullptr : &*found;
}

json ModelStore::public_record(const Record& record) const {
    json result = record.config.to_public_json(record.config.id == active_id_, record.size_bytes);
    result["family"] = record.family;
    result["relativePath"] = record.relative_path;
    result["quantization"] = {
        {"label", record.quantization}, {"method", record.quantization_method}, {"quantized", record.quantized},
    };
    result["runtimeAvailable"] = record.runtime_available;
    result["compatible"] = record.compatible;
    result["statusReason"] = record.status_reason;
    result["autoDetected"] = true;
    return result;
}

json ModelStore::list() const {
    std::scoped_lock lock(state_mutex_);
    json result = json::array();
    for (const auto& record : records_) result.push_back(public_record(record));
    return result;
}

json ModelStore::rescan() {
    std::scoped_lock discovery_lock(discovery_mutex_);
    const json discovered = runner::discover_models();
    const EngineRuntimeInfo native = engine_runtime_info();
    const bool native_available = native.ncnn_available && native.vulkan_compiled && native.gpu_count > 0;
    std::vector<Record> next_records;
    std::set<std::string> identifiers;
    for (const auto& item : discovered) {
        try {
            Record record;
            record.config = config_from_discovery(item);
            record.primary_path = fs::absolute(item.at("path").get<std::string>());
            const std::string secondary = item.value("secondaryPath", "");
            if (!secondary.empty()) record.secondary_path = fs::absolute(secondary);
            if (!path_is_within(record.primary_path, models_root_) || !fs::exists(record.primary_path)) {
                throw std::invalid_argument("discovered path is outside the models directory or no longer exists");
            }
            if (!record.secondary_path.empty() && (!path_is_within(record.secondary_path, models_root_) || !fs::exists(record.secondary_path))) {
                throw std::invalid_argument("discovered secondary artifact is invalid");
            }
            if (!identifiers.insert(record.config.id).second) throw std::invalid_argument("duplicate discovered model id");
            record.size_bytes = item.value("sizeBytes", std::uintmax_t{0});
            record.family = item.value("family", "Models");
            record.relative_path = item.value("relativePath", record.primary_path.filename().u8string());
            record.quantization = item.value("quantization", "Unknown");
            record.quantization_method = item.value("quantizationMethod", "");
            record.quantized = item.value("quantized", false);
            record.runtime_available = record.config.format == "ncnn" ? native_available : item.value("runtimeAvailable", false);
            record.compatible = item.value("compatible", true);
            record.status_reason = item.value("statusReason", "");
            next_records.push_back(std::move(record));
        } catch (const std::exception& error) {
            std::cerr << "Skipping invalid discovered model: " << error.what() << '\n';
        }
    }

    std::scoped_lock state_lock(state_mutex_);
    std::set<std::string> changed_or_removed;
    for (const auto& previous : records_) {
        const auto found = std::find_if(next_records.begin(), next_records.end(), [&](const Record& record) {
            return record.config.id == previous.config.id;
        });
        if (found == next_records.end() || found->primary_path != previous.primary_path ||
            found->size_bytes != previous.size_bytes || found->config.created_at != previous.config.created_at) {
            changed_or_removed.insert(previous.config.id);
        }
    }
    for (const auto& id : changed_or_removed) pipeline_engines_.erase(id);
    if (changed_or_removed.count(active_id_) != 0) {
        active_engine_.reset();
        active_id_.clear();
        persist_active_id();
    }
    records_ = std::move(next_records);
    json result = json::array();
    for (const auto& record : records_) result.push_back(public_record(record));
    return result;
}

json ModelStore::activate(const std::string& id) {
    std::scoped_lock lock(state_mutex_);
    Record* record = find_record(id);
    if (!record) throw std::invalid_argument("model not found; rescan the models directory");
    if (!record->compatible) {
        throw std::invalid_argument(record->status_reason.empty() ? "model inspection failed" : record->status_reason);
    }
    if (!record->runtime_available) {
        throw std::runtime_error("the " + record->config.runtime_name + " runtime is not available for this model");
    }
    if (active_id_ != id || !active_engine_) {
        auto engine = create_engine(record->config, record->primary_path, record->secondary_path);
        active_engine_ = std::shared_ptr<InferenceEngine>(std::move(engine));
        active_id_ = id;
        persist_active_id();
    }
    return public_record(*record);
}

InferenceResult ModelStore::infer(
    const std::string& requested_model_id,
    const std::vector<unsigned char>& image,
    float confidence,
    float iou,
    const GroundingRequest& grounding
) {
    std::shared_ptr<InferenceEngine> engine;
    std::string active_id;
    {
        std::scoped_lock lock(state_mutex_);
        engine = active_engine_;
        active_id = active_id_;
    }
    if (!engine) throw std::runtime_error("no model is active");
    if (!requested_model_id.empty() && requested_model_id != active_id) {
        throw std::invalid_argument("requested model is not active; activate it before inference");
    }
    confidence = std::clamp(confidence, 0.0F, 1.0F);
    iou = std::clamp(iou, 0.0F, 1.0F);
    const auto queue_start = Clock::now();
    std::scoped_lock inference_lock(inference_mutex_);
    const double queue_wait = milliseconds(queue_start, Clock::now());
    InferenceResult result = engine->infer(image, confidence, iou, grounding);
    result.timing.queue_wait_ms = queue_wait;
    result.timing.backend_total_ms += queue_wait;
    return result;
}

InferenceResult ModelStore::infer_pipeline_model(
    const std::string& model_id,
    const std::vector<unsigned char>& image,
    float confidence,
    float iou
) {
    std::shared_ptr<InferenceEngine> engine;
    std::string resolved_id;
    {
        std::scoped_lock lock(state_mutex_);
        resolved_id = model_id.empty() ? active_id_ : model_id;
        const Record* record = find_record(resolved_id);
        if (!record) throw std::invalid_argument("pipeline model was not found in the models directory");
        if (!record->compatible || !record->runtime_available) throw std::runtime_error("pipeline model is not runnable on an installed backend");
        if (resolved_id == active_id_ && active_engine_) {
            engine = active_engine_;
        } else {
            auto cached = pipeline_engines_.find(resolved_id);
            if (cached == pipeline_engines_.end()) {
                auto created = create_engine(record->config, record->primary_path, record->secondary_path);
                cached = pipeline_engines_.emplace(resolved_id, std::shared_ptr<InferenceEngine>(std::move(created))).first;
            }
            engine = cached->second;
        }
    }
    confidence = std::clamp(confidence, 0.0F, 1.0F);
    iou = std::clamp(iou, 0.0F, 1.0F);
    const auto queue_start = Clock::now();
    std::scoped_lock inference_lock(inference_mutex_);
    const double queue_wait = milliseconds(queue_start, Clock::now());
    InferenceResult result = engine->infer(image, confidence, iou);
    result.timing.queue_wait_ms = queue_wait;
    result.timing.backend_total_ms += queue_wait;
    return result;
}

json ModelStore::benchmark(
    const std::string& requested_model_id,
    const std::vector<unsigned char>& image,
    float confidence,
    float iou,
    int warmup,
    int count
) {
    if (warmup < 0 || warmup > 100) throw std::invalid_argument("warmup must be between 0 and 100");
    if (count < 1 || count > 1000) throw std::invalid_argument("count must be between 1 and 1000");
    std::uintmax_t size_bytes = 0;
    int threads = 0;
    int gpu_device = -1;
    std::string precision;
    std::string backend;
    std::string accelerator;
    {
        std::scoped_lock lock(state_mutex_);
        const Record* record = find_record(requested_model_id.empty() ? active_id_ : requested_model_id);
        if (!record) throw std::invalid_argument("benchmark model was not found");
        size_bytes = record->size_bytes;
        threads = record->config.threads;
        gpu_device = record->config.gpu_device;
        precision = precision_name(record->config.precision);
        backend = record->config.backend;
        accelerator = record->config.accelerator;
    }
    for (int index = 0; index < warmup; ++index) (void)infer(requested_model_id, image, confidence, iou);
    std::vector<InferenceResult> samples;
    samples.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) samples.push_back(infer(requested_model_id, image, confidence, iou));

    auto metric = [&](auto accessor) {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto& sample : samples) values.push_back(accessor(sample.timing));
        return values;
    };
    const auto total_values = metric([](const Timing& timing) { return timing.backend_total_ms; });
    const json total_stats = statistics(total_values);
    std::ostringstream csv;
    csv << "sample,decode_ms,preprocess_ms,inference_ms,detection_nms_ms,mask_reconstruction_ms,tracking_ms,queue_wait_ms,grading_ms,backend_total_ms\n";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto& timing = samples[index].timing;
        csv << index << ',' << timing.decode_ms << ',' << timing.preprocess_ms << ',' << timing.inference_ms << ','
            << timing.detection_nms_ms << ',' << timing.mask_reconstruction_ms << ',' << timing.tracking_ms << ','
            << timing.queue_wait_ms << ',' << timing.grading_ms << ',' << timing.backend_total_ms << '\n';
    }
    const EngineRuntimeInfo runtime = engine_runtime_info();
    const int resolved_gpu = gpu_device < 0 ? runtime.default_gpu_index : gpu_device;
    const std::string gpu_name = resolved_gpu == runtime.default_gpu_index ? runtime.default_gpu_name : "Vulkan device " + std::to_string(resolved_gpu);
    return {
        {"schemaVersion", 1}, {"mode", "sequential"}, {"warmup", warmup}, {"count", count},
        {"precision", precision}, {"modelSizeBytes", size_bytes}, {"runtimeThreads", threads},
        {"accelerator", {{"backend", backend == "ncnn" ? "NCNN" : backend}, {"api", accelerator}, {"gpuDeviceIndex", resolved_gpu}, {"gpuDevice", gpu_name}}},
        {"droppedFrames", 0},
        {"effectiveFps", total_stats["mean"].get<double>() > 0.0 ? 1000.0 / total_stats["mean"].get<double>() : 0.0},
        {"latencyMs", {
            {"decode", statistics(metric([](const Timing& value) { return value.decode_ms; }))},
            {"preprocess", statistics(metric([](const Timing& value) { return value.preprocess_ms; }))},
            {"inferenceAllOutputs", statistics(metric([](const Timing& value) { return value.inference_ms; }))},
            {"detectionNms", statistics(metric([](const Timing& value) { return value.detection_nms_ms; }))},
            {"maskReconstruction", statistics(metric([](const Timing& value) { return value.mask_reconstruction_ms; }))},
            {"tracking", statistics(metric([](const Timing& value) { return value.tracking_ms; }))},
            {"queueWait", statistics(metric([](const Timing& value) { return value.queue_wait_ms; }))},
            {"grading", statistics(metric([](const Timing& value) { return value.grading_ms; }))},
            {"backendTotal", total_stats},
        }},
        {"peakRssBytes", peak_rss_bytes()}, {"samplesCsv", csv.str()},
        {"notes", "Single-image sequential local benchmark; image drawing and public client HTTP are excluded. Camera throughput/latency require the video benchmark."},
    };
}

std::string ModelStore::active_model_id() const {
    std::scoped_lock lock(state_mutex_);
    return active_id_;
}

void ModelStore::persist_active_id() const {
    const fs::path path = state_root_ / "active-model.txt";
    if (fs::exists(path)) {
        try {
            if (read_text(path) == active_id_) return;
        } catch (...) {
            // Fall through and replace unreadable state with the current ID.
        }
    }
    write_file(path, active_id_);
}

}  // namespace runner
