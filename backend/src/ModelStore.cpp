#include "runner/ModelStore.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <windows.h>
#include <psapi.h>

namespace runner {
namespace {

using Clock = std::chrono::steady_clock;

void write_file(const fs::path& path, const std::string& content, bool binary) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path, std::ios::out | std::ios::trunc | (binary ? std::ios::binary : std::ios::openmode(0)));
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::uintmax_t model_size(const fs::path& directory) {
    std::error_code error;
    const auto param_size = fs::file_size(directory / "model.param", error);
    if (error) return 0;
    const auto bin_size = fs::file_size(directory / "model.bin", error);
    if (error) return param_size;
    return param_size + bin_size;
}

std::uintmax_t model_size(const ModelConfig& config, const fs::path& directory) {
    if (config.format == "ncnn") return model_size(directory);
    std::error_code error;
    const auto size = fs::file_size(directory / config.artifact_file, error);
    return error ? 0 : size;
}

fs::path primary_artifact(const ModelConfig& config, const fs::path& directory) {
    return config.format == "ncnn" ? directory / "model.param" : directory / config.artifact_file;
}

fs::path secondary_artifact(const ModelConfig& config, const fs::path& directory) {
    return config.format == "ncnn" ? directory / "model.bin" : fs::path{};
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string artifact_format(const std::string& filename) {
    const std::string lower = lower_copy(filename);
    const std::vector<std::pair<std::string, std::string>> suffixes = {
        {".torchscript", "torchscript"}, {".onnx", "onnx"}, {".engine", "engine"},
        {".tflite", lower.size() >= 15 && lower.rfind("_edgetpu.tflite") == lower.size() - 15 ? "edgetpu" : "litert"},
        {".mnn", "mnn"}, {".pte", "executorch"}, {".xml", "openvino"},
        {".pb", "pb"}, {".zip", "archive"}, {".pt", "pt"},
    };
    for (const auto& [suffix, format] : suffixes) {
        if (lower.size() >= suffix.size() && lower.compare(lower.size() - suffix.size(), suffix.size(), suffix) == 0) return format;
    }
    throw std::invalid_argument("unsupported model format; use .pt, .torchscript, .onnx, .engine, .tflite, .mnn, .pte, .pb, .xml, or a supported export directory in .zip");
}

std::string artifact_extension(const std::string& filename) {
    const fs::path path(filename);
    std::string extension = lower_copy(path.extension().string());
    if (extension.empty() || extension.size() > 20) throw std::invalid_argument("model artifact requires a supported file extension");
    return extension;
}

std::string framework_for_format(const std::string& format) {
    if (format == "pt" || format == "torchscript") return "pytorch";
    if (format == "onnx") return "onnx";
    if (format == "engine") return "tensorrt";
    if (format == "openvino") return "openvino";
    if (format == "pb" || format == "litert" || format == "edgetpu") return "tensorflow";
    return format == "archive" ? "ultralytics-export" : format;
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

ModelStore::ModelStore(fs::path root) : root_(std::move(root)) {
    fs::create_directories(root_);
    load_from_disk();
}

void ModelStore::load_from_disk() {
    std::string persisted_active;
    if (fs::exists(root_ / "active-model.txt")) {
        persisted_active = read_text(root_ / "active-model.txt");
        persisted_active.erase(std::remove(persisted_active.begin(), persisted_active.end(), '\n'), persisted_active.end());
        persisted_active.erase(std::remove(persisted_active.begin(), persisted_active.end(), '\r'), persisted_active.end());
    }

    for (const auto& entry : fs::directory_iterator(root_)) {
        if (!entry.is_directory() || entry.path().filename().string().rfind(".upload-", 0) == 0) continue;
        const fs::path manifest_path = entry.path() / "manifest.json";
        if (!fs::exists(manifest_path)) continue;
        try {
            ModelConfig config = ModelConfig::from_json(json::parse(read_text(manifest_path)), true);
            if (config.id.empty()) config.id = entry.path().filename().string();
            if (!fs::exists(primary_artifact(config, entry.path())) || (config.format == "ncnn" && !fs::exists(secondary_artifact(config, entry.path())))) continue;
            const auto size = model_size(config, entry.path());
            records_.push_back({std::move(config), entry.path(), size});
        } catch (const std::exception& error) {
            std::cerr << "Skipping invalid model at " << entry.path() << ": " << error.what() << '\n';
        }
    }
    std::sort(records_.begin(), records_.end(), [](const Record& first, const Record& second) {
        return first.config.created_at > second.config.created_at;
    });

    if (!persisted_active.empty()) {
        try {
            activate(persisted_active);
        } catch (const std::exception& error) {
            std::cerr << "Could not restore active model '" << persisted_active << "': " << error.what() << '\n';
        }
    }
}

json ModelStore::import_artifact(
    const std::string& requested_name,
    const std::string& task,
    const std::string& precision,
    const std::string& filename,
    const std::string& artifact,
    bool trusted_checkpoint,
    int input_width,
    int input_height
) {
    if (artifact.empty()) throw std::invalid_argument("model artifact is empty");
    const std::string format = artifact_format(filename);
    if ((format == "pt" || format == "torchscript" || format == "executorch") && !trusted_checkpoint) {
        throw std::invalid_argument("executable model checkpoints require confirmation that the file comes from a trusted source");
    }
    std::string name = requested_name;
    if (name.empty()) name = fs::path(filename).stem().string();
    const std::string extension = artifact_extension(filename);
    const std::string stored_filename = "model" + extension;
    json generated = {
        {"name", name}, {"schemaVersion", 3}, {"task", task}, {"precision", precision},
        {"framework", framework_for_format(format)},
        {"artifact", {
            {"format", format}, {"file", stored_filename}, {"backend", "ultralytics"},
            {"accelerator", "automatic"}, {"trustedCheckpoint", trusted_checkpoint},
        }},
        {"input", {
            {"width", input_width}, {"height", input_height}, {"blob", "images"},
            {"mean", {0.0F, 0.0F, 0.0F}}, {"norm", {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F}},
        }},
        {"outputs", {"auto"}}, {"labels", json::array()},
        {"runtime", {
            {"name", "ultralytics"}, {"version", "automatic"}, {"threads", 8},
            {"useVulkanCompute", false}, {"gpuDevice", -1},
        }},
        {"classification", {{"applySoftmax", false}, {"topK", 5}}},
        {"maxDetections", 300}, {"maskThreshold", 0.5F},
    };
    ModelConfig config = ModelConfig::from_json(generated);
    config.id = make_model_id(config.name);
    config.created_at = utc_now_iso8601();

    const fs::path temporary = root_ / (".upload-" + config.id);
    const fs::path destination = root_ / config.id;
    if (fs::exists(temporary) || fs::exists(destination)) throw std::runtime_error("model id collision; retry the import");
    try {
        fs::create_directories(temporary);
        write_file(temporary / stored_filename, artifact, true);
        write_file(temporary / "manifest.json", config.to_json().dump(2), false);
        fs::rename(temporary, destination);

        // Every direct artifact is loaded and warmed once during import. This
        // is where missing optional runtimes and incompatible exports fail.
        auto validated_engine = create_engine(config, destination / stored_filename, {});
        const json metadata = validated_engine->metadata();
        config.format = metadata.value("format", config.format);
        config.runtime_name = metadata.value("runtime", config.runtime_name);
        config.accelerator = metadata.value("accelerator", config.accelerator);
        config.labels = metadata.value("labels", config.labels);
        const std::string detected_task = metadata.value("task", task_name(config.task));
        if (detected_task == "yolo-detect") config.task = ModelTask::YoloDetect;
        else if (detected_task == "yolo-segment") config.task = ModelTask::YoloSegment;
        else if (detected_task == "classification") config.task = ModelTask::Classification;
        write_file(destination / "manifest.json", config.to_json().dump(2), false);
        std::scoped_lock lock(state_mutex_);
        records_.insert(records_.begin(), {config, destination, model_size(config, destination)});
        if (active_id_.empty()) {
            active_engine_ = std::shared_ptr<InferenceEngine>(std::move(validated_engine));
            active_id_ = config.id;
            persist_active_id();
        }
        return public_record(records_.front());
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        fs::remove_all(destination, ignored);
        throw;
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
    return record.config.to_public_json(record.config.id == active_id_, record.size_bytes);
}

json ModelStore::list() const {
    std::scoped_lock lock(state_mutex_);
    json result = json::array();
    for (const auto& record : records_) result.push_back(public_record(record));
    return result;
}

json ModelStore::import_model(const json& manifest, const std::string& param, const std::string& bin) {
    if (param.empty() || bin.empty()) throw std::invalid_argument("both .param and .bin files are required");
    ModelConfig config = ModelConfig::from_json(manifest);
    config.id = make_model_id(config.name);
    config.created_at = utc_now_iso8601();

    const fs::path temporary = root_ / (".upload-" + config.id);
    const fs::path destination = root_ / config.id;
    if (fs::exists(temporary) || fs::exists(destination)) throw std::runtime_error("model id collision; retry the import");

    try {
        fs::create_directories(temporary);
        write_file(temporary / "model.param", param, true);
        write_file(temporary / "model.bin", bin, true);
        write_file(temporary / "manifest.json", config.to_json().dump(2), false);
        fs::rename(temporary, destination);

        std::scoped_lock lock(state_mutex_);
        records_.insert(records_.begin(), {config, destination, model_size(destination)});
        if (active_id_.empty()) {
            try {
                auto engine = create_engine(config, destination / "model.param", destination / "model.bin");
                active_engine_ = std::shared_ptr<InferenceEngine>(std::move(engine));
                active_id_ = config.id;
                persist_active_id();
            } catch (...) {
                records_.erase(records_.begin());
                fs::remove_all(destination);
                throw;
            }
        }
        return public_record(records_.front());
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        throw;
    }
}

json ModelStore::activate(const std::string& id) {
    std::scoped_lock lock(state_mutex_);
    Record* record = find_record(id);
    if (!record) throw std::invalid_argument("model not found");
    if (active_id_ != id || !active_engine_) {
        auto engine = create_engine(record->config, primary_artifact(record->config, record->directory), secondary_artifact(record->config, record->directory));
        active_engine_ = std::shared_ptr<InferenceEngine>(std::move(engine));
        active_id_ = id;
        persist_active_id();
    }
    return public_record(*record);
}

void ModelStore::remove(const std::string& id) {
    std::scoped_lock lock(state_mutex_);
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const Record& record) { return record.config.id == id; });
    if (found == records_.end()) throw std::invalid_argument("model not found");
    const fs::path directory = found->directory;
    if (active_id_ == id) {
        active_engine_.reset();
        active_id_.clear();
        persist_active_id();
    }
    pipeline_engines_.erase(id);
    records_.erase(found);
    std::error_code error;
    fs::remove_all(directory, error);
    if (error) throw std::runtime_error("model metadata was removed but its files could not be deleted: " + error.message());
}

InferenceResult ModelStore::infer(
    const std::string& requested_model_id,
    const std::vector<unsigned char>& image,
    float confidence,
    float iou
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
    InferenceResult result = engine->infer(image, confidence, iou);
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
        if (!record) throw std::invalid_argument("pipeline model was not found; import it and set its model ID in pipeline config");
        if (resolved_id == active_id_ && active_engine_) {
            engine = active_engine_;
        } else {
            auto cached = pipeline_engines_.find(resolved_id);
            if (cached == pipeline_engines_.end()) {
                auto created = create_engine(record->config, primary_artifact(record->config, record->directory), secondary_artifact(record->config, record->directory));
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
    write_file(root_ / "active-model.txt", active_id_, false);
}

}  // namespace runner
