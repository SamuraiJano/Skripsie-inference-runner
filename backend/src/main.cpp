#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#include "runner/Engine.hpp"
#include "runner/ModelStore.hpp"
#include "runner/Pipeline.hpp"
#include "runner/Types.hpp"

#ifndef RUNNER_VERSION
#define RUNNER_VERSION "dev"
#endif

#ifndef RUNNER_FRONTEND_DIST
#define RUNNER_FRONTEND_DIST "../frontend/dist"
#endif

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Arguments {
    std::string host = "127.0.0.1";
    int port = 8080;
    fs::path data = "data/models";
    fs::path models = "models";
    fs::path static_files = RUNNER_FRONTEND_DIST;
    fs::path pipeline_config = "config/pipeline.example.json";
    std::string detector_model_id;
    std::string segmentation_model_id;
    std::string python_backend_url = "http://127.0.0.1:8091";
};

httplib::Server* active_server = nullptr;

void stop_server(int) {
    if (active_server) active_server->stop();
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help" || key == "-h") {
            std::cout
                << "Inference Runner Windows " << RUNNER_VERSION << "\n\n"
                << "Usage: inference_runner_windows [options]\n"
                << "  --host <address>   Bind address (default 127.0.0.1)\n"
                << "  --port <number>    HTTP port (default 8080)\n"
                << "  --data <path>      Runtime state directory\n"
                << "  --models <path>    Auto-discovered model directory (default models)\n"
                << "  --static <path>    Built React dist directory\n"
                << "  --pipeline-config <path>  Live conveyor pipeline JSON\n"
                << "  --detector-model <id>     Override YOLO26n detector model ID\n"
                << "  --segmenter-model <id>    Override defect segmenter model ID\n"
                << "  --python-backend <url>    Private multi-format service (default http://127.0.0.1:8091)\n";
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--host") args.host = value;
        else if (key == "--port") args.port = std::stoi(value);
        else if (key == "--data") args.data = value;
        else if (key == "--models") args.models = value;
        else if (key == "--static") args.static_files = value;
        else if (key == "--pipeline-config") args.pipeline_config = value;
        else if (key == "--detector-model") args.detector_model_id = value;
        else if (key == "--segmenter-model") args.segmentation_model_id = value;
        else if (key == "--python-backend") args.python_backend_url = value;
        else throw std::invalid_argument("unknown argument: " + key);
    }
    if (args.port < 1 || args.port > 65535) throw std::invalid_argument("port must be between 1 and 65535");
    return args;
}

std::string architecture() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

void respond_json(httplib::Response& response, const json& value, int status = 200) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_content(value.dump(), "application/json; charset=utf-8");
}

void respond_error(httplib::Response& response, const std::exception& error, int status) {
    respond_json(response, {{"error", error.what()}}, status);
}

template <typename Function>
void guarded(httplib::Response& response, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        respond_error(response, error, 400);
    } catch (const nlohmann::json::exception& error) {
        respond_error(response, error, 400);
    } catch (const std::exception& error) {
        respond_error(response, error, 500);
    }
}

std::vector<httplib::FormData> read_multipart(
    const httplib::Request& request,
    const httplib::ContentReader& content_reader
) {
    if (!request.is_multipart_form_data()) throw std::invalid_argument("request must use multipart/form-data");
    std::vector<httplib::FormData> parts;
    std::size_t received = 0;
    bool too_large = false;
    const bool read = content_reader(
        [&](const httplib::FormData& part) {
            if (parts.size() >= 16) return false;
            parts.push_back(part);
            return true;
        },
        [&](const char* data, std::size_t length) {
            received += length;
            if (received > 1024ULL * 1024ULL * 1024ULL || parts.empty()) {
                too_large = true;
                return false;
            }
            parts.back().content.append(data, length);
            return true;
        });
    if (too_large) throw std::invalid_argument("multipart payload is too large");
    if (!read) throw std::invalid_argument("could not read multipart payload");
    return parts;
}

const httplib::FormData& required_part(const std::vector<httplib::FormData>& parts, const char* name) {
    const auto found = std::find_if(parts.begin(), parts.end(), [&](const httplib::FormData& part) { return part.name == name; });
    if (found == parts.end()) throw std::invalid_argument(std::string("missing multipart field '") + name + "'");
    return *found;
}

std::string optional_part(const std::vector<httplib::FormData>& parts, const char* name) {
    const auto found = std::find_if(parts.begin(), parts.end(), [&](const httplib::FormData& part) { return part.name == name; });
    return found == parts.end() ? std::string() : found->content;
}

float float_part(const std::vector<httplib::FormData>& parts, const char* name, float fallback) {
    const std::string value = optional_part(parts, name);
    if (value.empty()) return fallback;
    try {
        return std::stof(value);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid numeric field '") + name + "'");
    }
}

std::uint64_t uint64_part(const std::vector<httplib::FormData>& parts, const char* name, std::uint64_t fallback) {
    const std::string value = optional_part(parts, name);
    if (value.empty()) return fallback;
    try {
        return std::stoull(value);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid integer field '") + name + "'");
    }
}

runner::PipelineConfig load_pipeline_config(const fs::path& path) {
    if (!fs::exists(path)) {
        runner::PipelineConfig defaults;
        defaults.validate();
        return defaults;
    }
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open pipeline config " + path.string());
    return runner::PipelineConfig::from_json(json::parse(input));
}

void append_encoded_bytes(void* context, void* data, int size) {
    auto& output = *static_cast<std::vector<unsigned char>*>(context);
    const auto* bytes = static_cast<unsigned char*>(data);
    output.insert(output.end(), bytes, bytes + size);
}

runner::LiveFrame decode_live_frame(std::uint64_t frame_id, const std::vector<unsigned char>& encoded) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &channels, 3);
    if (!raw) throw std::invalid_argument(std::string("could not decode live frame: ") + stbi_failure_reason());
    runner::LiveFrame frame;
    frame.frame_id = frame_id;
    frame.captured_at = std::chrono::steady_clock::now();
    frame.timestamp = runner::utc_now_iso8601();
    frame.width = width;
    frame.height = height;
    frame.encoded_image = encoded;
    frame.rgb_image.assign(raw, raw + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    stbi_image_free(raw);
    return frame;
}

double laplacian_variance(const std::vector<unsigned char>& rgb, int width, int height) {
    if (width < 3 || height < 3) return 0.0;
    const auto grey = [&](int x, int y) {
        const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 3;
        return 0.299 * rgb[offset] + 0.587 * rgb[offset + 1] + 0.114 * rgb[offset + 2];
    };
    double sum = 0.0;
    double squares = 0.0;
    std::size_t count = 0;
    // Variance of this four-neighbour Laplacian is a cheap focus metric: crisp
    // edges produce larger responses, while blurred/flat crops remain small.
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const double value = grey(x - 1, y) + grey(x + 1, y) + grey(x, y - 1) + grey(x, y + 1) - 4.0 * grey(x, y);
            sum += value;
            squares += value * value;
            ++count;
        }
    }
    if (count == 0) return 0.0;
    const double mean = sum / static_cast<double>(count);
    return squares / static_cast<double>(count) - mean * mean;
}

runner::CropImage crop_live_frame(const runner::LiveFrame& frame, const runner::CropBounds& bounds) {
    runner::CropImage crop;
    if (bounds.width <= 0 || bounds.height <= 0 || frame.rgb_image.empty()) return crop;
    std::vector<unsigned char> pixels(static_cast<std::size_t>(bounds.width) * static_cast<std::size_t>(bounds.height) * 3);
    for (int row = 0; row < bounds.height; ++row) {
        const std::size_t source = (static_cast<std::size_t>(bounds.y + row) * frame.width + bounds.x) * 3;
        const std::size_t destination = static_cast<std::size_t>(row) * bounds.width * 3;
        std::copy_n(frame.rgb_image.data() + source, static_cast<std::size_t>(bounds.width) * 3, pixels.data() + destination);
    }
    crop.width = bounds.width;
    crop.height = bounds.height;
    crop.sharpness = laplacian_variance(pixels, bounds.width, bounds.height);
    if (!stbi_write_jpg_to_func(append_encoded_bytes, &crop.encoded, bounds.width, bounds.height, 3, pixels.data(), 88)) {
        crop.encoded.clear();
    }
    return crop;
}

double foreground_area(const runner::MaskRle& mask) {
    bool foreground = false;
    double area = 0.0;
    for (const auto count : mask.counts) {
        if (foreground) area += count;
        foreground = !foreground;
    }
    return area;
}

runner::SegmentationResult segmentation_from_inference(const runner::InferenceResult& inference) {
    runner::SegmentationResult result;
    if (!inference.classifications.empty()) {
        const auto best = std::max_element(inference.classifications.begin(), inference.classifications.end(),
            [](const auto& left, const auto& right) { return left.score < right.score; });
        result.class_probabilities.resize(inference.classifications.size(), 0.0F);
        for (const auto& score : inference.classifications) {
            if (score.class_id >= 0 && static_cast<std::size_t>(score.class_id) < result.class_probabilities.size()) {
                result.class_probabilities[static_cast<std::size_t>(score.class_id)] = score.score;
            }
        }
        result.predicted_class = best->class_id;
        result.confidence = best->score;
        result.success = true;
    } else if (!inference.detections.empty()) {
        int maximum_class = 0;
        for (const auto& detection : inference.detections) maximum_class = std::max(maximum_class, detection.class_id);
        result.class_probabilities.assign(static_cast<std::size_t>(maximum_class + 1), 0.0F);
        for (const auto& detection : inference.detections) {
            if (detection.class_id >= 0) {
                auto& probability = result.class_probabilities[static_cast<std::size_t>(detection.class_id)];
                probability = std::max(probability, detection.confidence);
            }
        }
        // Conservative per-view selection: the worst class with a reported
        // probability is retained for the separate track-level aggregator.
        const runner::Detection* selected = nullptr;
        for (const auto& detection : inference.detections) {
            if (!selected || detection.class_id > selected->class_id ||
                (detection.class_id == selected->class_id && detection.confidence > selected->confidence)) selected = &detection;
        }
        result.predicted_class = selected->class_id;
        result.confidence = selected->confidence;
        result.defect_mask = selected->mask;
        result.defect_area = foreground_area(selected->mask);
        result.success = true;
    } else {
        result.message = "segmentation model returned no class evidence";
    }
    result.processing_ms = inference.timing.backend_total_ms;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
#ifdef _WIN32
        if (_putenv_s("RUNNER_PYTHON_BACKEND_URL", args.python_backend_url.c_str()) != 0) {
            throw std::runtime_error("could not configure the Python backend URL");
        }
#endif
        const runner::EngineRuntimeInfo runtime = runner::engine_runtime_info();
        const json initial_python_health = runner::python_backend_health();
        if ((!runtime.vulkan_compiled || runtime.gpu_count < 1) && initial_python_health.value("status", "unavailable") != "ok") {
            throw std::runtime_error("no model backend is available; start the Python model service or repair the Vulkan GPU runtime");
        }
        runner::ModelStore models(args.models, args.data);
        runner::PipelineConfig pipeline_config = load_pipeline_config(args.pipeline_config);
        if (!args.detector_model_id.empty()) pipeline_config.yolo26n_model_id = args.detector_model_id;
        if (!args.segmentation_model_id.empty()) pipeline_config.segmentation_model_id = args.segmentation_model_id;
        auto pipeline = std::make_unique<runner::MultiViewPipeline>(
            pipeline_config,
            [&](const runner::LiveFrame& frame) {
                auto result = models.infer_pipeline_model(pipeline_config.yolo26n_model_id, frame.encoded_image,
                    pipeline_config.detection_confidence, pipeline_config.detection_iou);
                result.detections.erase(std::remove_if(result.detections.begin(), result.detections.end(), [&](const runner::Detection& detection) {
                    return detection.class_id != pipeline_config.naartjie_class_id;
                }), result.detections.end());
                return result;
            },
            [&](const runner::SegmentationJobV2& job) {
                const std::string model_id = pipeline_config.segmentation_model_id.empty()
                    ? pipeline_config.yolo26n_model_id : pipeline_config.segmentation_model_id;
                return segmentation_from_inference(models.infer_pipeline_model(model_id, job.crop_image,
                    pipeline_config.class_confidence_threshold, pipeline_config.detection_iou));
            },
            crop_live_frame
        );
        pipeline->start();
        const auto started_at = std::chrono::steady_clock::now();
        const unsigned int cores = std::max(1U, std::thread::hardware_concurrency());
        const unsigned int recommended_threads = std::min(8U, cores);

        httplib::Server server;
        active_server = &server;
        std::signal(SIGINT, stop_server);
        std::signal(SIGTERM, stop_server);
        server.set_payload_max_length(1024ULL * 1024ULL * 1024ULL);
        server.set_read_timeout(120, 0);
        server.set_write_timeout(120, 0);
        server.set_default_headers({
            {"X-Content-Type-Options", "nosniff"},
            {"X-Frame-Options", "DENY"},
            {"Referrer-Policy", "no-referrer"},
        });

        server.Get("/api/health", [&](const httplib::Request&, httplib::Response& response) {
            const double uptime = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
            const std::string active_model = models.active_model_id();
            const json python = runner::python_backend_health();
            const bool python_available = python.value("status", "unavailable") == "ok";
            respond_json(response, {
                {"status", "ok"}, {"version", RUNNER_VERSION}, {"engine", python_available ? "NCNN + Ultralytics multi-backend" : runner::engine_name()},
                {"ncnnAvailable", runner::ncnn_available()}, {"architecture", architecture()},
                {"platform", "Windows"}, {"vulkanAvailable", runtime.vulkan_compiled && runtime.gpu_count > 0},
                {"gpuCount", runtime.gpu_count}, {"gpuDeviceIndex", runtime.default_gpu_index},
                {"gpuDevice", runtime.default_gpu_name},
                {"pythonBackendAvailable", python_available},
                {"cudaAvailable", python.value("cudaAvailable", false)},
                {"cudaVersion", python.value("cudaVersion", json(nullptr))},
                {"torchVersion", python.value("torch", json(nullptr))},
                {"ultralyticsVersion", python.value("ultralytics", json(nullptr))},
                {"formats", runner::model_format_capabilities()},
                {"logicalCores", cores}, {"recommendedThreads", recommended_threads},
                {"uptimeSeconds", uptime}, {"activeModelId", active_model.empty() ? json(nullptr) : json(active_model)},
            });
        });

        server.Get("/api/models", [&](const httplib::Request&, httplib::Response& response) {
            guarded(response, [&] { respond_json(response, models.rescan()); });
        });

        server.Post("/api/models/rescan", [&](const httplib::Request&, httplib::Response& response) {
            guarded(response, [&] { respond_json(response, models.rescan()); });
        });

        server.Get("/api/capabilities", [&](const httplib::Request&, httplib::Response& response) {
            respond_json(response, {{"formats", runner::model_format_capabilities()}});
        });

        server.Get("/api/pipeline", [&](const httplib::Request&, httplib::Response& response) {
            respond_json(response, pipeline->snapshot());
        });

        server.Get("/api/pipeline/config", [&](const httplib::Request&, httplib::Response& response) {
            respond_json(response, pipeline_config.to_json());
        });

        server.Get("/api/pipeline/captures", [&](const httplib::Request&, httplib::Response& response) {
            guarded(response, [&] { respond_json(response, pipeline->capture_library()); });
        });

        server.Get(R"(/api/pipeline/captures/([A-Za-z0-9_-]+)\.jpg)", [&](const httplib::Request& request, httplib::Response& response) {
            guarded(response, [&] {
                const auto image = pipeline->capture_image(request.matches[1].str());
                if (!image) {
                    respond_json(response, {{"error", "capture image not found"}}, 404);
                    return;
                }
                response.status = 200;
                response.set_header("Cache-Control", "public, max-age=31536000, immutable");
                response.set_content(
                    std::string(reinterpret_cast<const char*>(image->data()), image->size()),
                    "image/jpeg"
                );
            });
        });

        server.Post("/api/pipeline/reset", [&](const httplib::Request&, httplib::Response& response) {
            pipeline->reset();
            respond_json(response, pipeline->snapshot());
        });

        server.Post("/api/pipeline/frame", [&](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& reader) {
            guarded(response, [&] {
                const auto parts = read_multipart(request, reader);
                const auto& image_part = required_part(parts, "image");
                if (image_part.content.size() > 25 * 1024 * 1024) throw std::invalid_argument("live frame exceeds the 25 MB limit");
                const std::uint64_t frame_id = uint64_part(parts, "frameId", 0);
                if (frame_id == 0) throw std::invalid_argument("live frame requires a positive frameId");
                const std::vector<unsigned char> bytes(image_part.content.begin(), image_part.content.end());
                runner::LiveFrame frame = decode_live_frame(frame_id, bytes);
                json result = pipeline->process_frame(frame);
                result["sourceWidth"] = frame.width;
                result["sourceHeight"] = frame.height;
                respond_json(response, result);
            });
        });

        server.Post(R"(/api/models/([A-Za-z0-9_-]+)/activate)", [&](const httplib::Request& request, httplib::Response& response) {
            guarded(response, [&] { respond_json(response, models.activate(request.matches[1].str())); });
        });

        server.Post("/api/infer", [&](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& reader) {
            guarded(response, [&] {
                const auto parts = read_multipart(request, reader);
                const auto& image_part = required_part(parts, "image");
                if (image_part.content.size() > 25 * 1024 * 1024) throw std::invalid_argument("image exceeds the 25 MB limit");
                const std::string model_id = optional_part(parts, "modelId");
                const float confidence = float_part(parts, "confidence", 0.25F);
                const float iou = float_part(parts, "iou", 0.50F);
                runner::GroundingRequest grounding;
                grounding.prompt = optional_part(parts, "prompt");
                grounding.mode = optional_part(parts, "groundingMode");
                grounding.generation_mode = optional_part(parts, "generationMode");
                if (grounding.mode.empty()) grounding.mode = "detect";
                if (grounding.generation_mode.empty()) grounding.generation_mode = "hybrid";
                if (grounding.prompt.size() > 1000) throw std::invalid_argument("grounding prompt must not exceed 1000 characters");
                const std::vector<unsigned char> bytes(image_part.content.begin(), image_part.content.end());
                respond_json(response, models.infer(model_id, bytes, confidence, iou, grounding).to_json());
            });
        });

        server.Post("/api/benchmark", [&](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& reader) {
            guarded(response, [&] {
                const auto parts = read_multipart(request, reader);
                const auto& image_part = required_part(parts, "image");
                if (image_part.content.size() > 25 * 1024 * 1024) throw std::invalid_argument("image exceeds the 25 MB limit");
                const std::string model_id = optional_part(parts, "modelId");
                const float confidence = float_part(parts, "confidence", 0.25F);
                const float iou = float_part(parts, "iou", 0.50F);
                const int warmup = static_cast<int>(float_part(parts, "warmup", 5.0F));
                const int count = static_cast<int>(float_part(parts, "count", 50.0F));
                const std::vector<unsigned char> bytes(image_part.content.begin(), image_part.content.end());
                respond_json(response, models.benchmark(model_id, bytes, confidence, iou, warmup, count));
            });
        });

        if (fs::is_directory(args.static_files)) {
            if (!server.set_mount_point("/", args.static_files.string())) {
                std::cerr << "warning: could not mount frontend at " << args.static_files << '\n';
            }
        } else {
            std::cerr << "warning: frontend dist not found at " << args.static_files << "; API-only mode\n";
        }

        server.set_error_handler([](const httplib::Request& request, httplib::Response& response) {
            if (request.path.rfind("/api/", 0) == 0 && response.status == 404 && response.body.empty()) {
                respond_json(response, {{"error", "endpoint not found"}}, 404);
            }
        });

        std::cout << "Inference Runner Windows " << RUNNER_VERSION << '\n'
                  << "  engine: " << runner::engine_name() << '\n'
                  << "  gpu: " << runtime.default_gpu_name << " (Vulkan device " << runtime.default_gpu_index << ")\n"
                  << "  architecture: " << architecture() << " (" << cores << " logical cores)\n"
                  << "  models: " << fs::absolute(args.models) << " (automatic discovery)\n"
                  << "  state: " << fs::absolute(args.data) << '\n'
                  << "  pipeline config: " << fs::absolute(args.pipeline_config) << '\n'
                  << "  model service: " << args.python_backend_url << " (" << runner::python_backend_health().value("status", "unavailable") << ")\n"
                  << "  dashboard: http://" << (args.host == "0.0.0.0" ? "127.0.0.1" : args.host) << ':' << args.port << "\n";

        if (!server.listen(args.host, args.port)) throw std::runtime_error("could not bind HTTP server");
        pipeline->stop();
        active_server = nullptr;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
