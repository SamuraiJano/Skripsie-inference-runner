#include "runner/Engine.hpp"

#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <httplib.h>

namespace runner {
namespace {

struct Endpoint {
    std::string host = "127.0.0.1";
    int port = 8091;
};

Endpoint backend_endpoint() {
    std::string value = "http://127.0.0.1:8091";
#ifdef _WIN32
    char* configured = nullptr;
    std::size_t configured_size = 0;
    if (_dupenv_s(&configured, &configured_size, "RUNNER_PYTHON_BACKEND_URL") == 0 && configured && *configured) {
        value = configured;
    }
    std::free(configured);
#else
    const char* configured = std::getenv("RUNNER_PYTHON_BACKEND_URL");
    if (configured && *configured) value = configured;
#endif
    constexpr const char* http_prefix = "http://";
    if (value.rfind(http_prefix, 0) == 0) value.erase(0, std::char_traits<char>::length(http_prefix));
    if (value.find('/') != std::string::npos) throw std::runtime_error("Python backend URL must contain only an HTTP host and port");
    Endpoint endpoint;
    const auto separator = value.rfind(':');
    if (separator == std::string::npos) {
        endpoint.host = value;
    } else {
        endpoint.host = value.substr(0, separator);
        try {
            endpoint.port = std::stoi(value.substr(separator + 1));
        } catch (...) {
            throw std::runtime_error("Python backend URL has an invalid port");
        }
    }
    if (endpoint.host != "127.0.0.1" && endpoint.host != "localhost") {
        throw std::runtime_error("Python model backend must use a loopback address");
    }
    if (endpoint.port < 1 || endpoint.port > 65535) throw std::runtime_error("Python backend port is out of range");
    return endpoint;
}

std::unique_ptr<httplib::Client> client_for_backend(int read_timeout_seconds = 300) {
    const Endpoint endpoint = backend_endpoint();
    auto client = std::make_unique<httplib::Client>(endpoint.host, endpoint.port);
    client->set_connection_timeout(2, 0);
    client->set_read_timeout(read_timeout_seconds, 0);
    client->set_write_timeout(120, 0);
    return client;
}

json parse_response(const httplib::Result& response, const std::string& action) {
    if (!response) {
        throw std::runtime_error(action + " could not reach the local Python model service: " + httplib::to_string(response.error()));
    }
    json value;
    try {
        value = json::parse(response->body);
    } catch (...) {
        throw std::runtime_error(action + " received an invalid response from the Python model service");
    }
    if (response->status < 200 || response->status >= 300) {
        throw std::runtime_error(value.value("error", action + " failed in the Python model service"));
    }
    return value;
}

std::string percent_encode(const std::string& value) {
    std::ostringstream output;
    output << std::uppercase << std::hex;
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            output << static_cast<char>(ch);
        } else {
            output << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
    }
    return output.str();
}

class PythonEngine final : public InferenceEngine {
public:
    PythonEngine(ModelConfig config, const fs::path& artifact_path) : config_(std::move(config)) {
        // Model loading is a private process boundary.  The Python registry
        // returns normalized metadata and reference-counts duplicate loads.
        json request = {
            {"modelId", config_.id},
            {"path", fs::absolute(artifact_path).u8string()},
            {"task", task_name(config_.task)},
            {"inputWidth", config_.input_width},
            {"inputHeight", config_.input_height},
            {"trustedCheckpoint", config_.trusted_checkpoint},
        };
        auto client = client_for_backend(config_.format == "locate-anything" ? 1800 : 300);
        metadata_ = parse_response(client->Post("/models/load", request.dump(), "application/json"), "model activation");
    }

    ~PythonEngine() override {
        try {
            auto client = client_for_backend(2);
            const json request = {{"modelId", config_.id}};
            (void)client->Post("/models/unload", request.dump(), "application/json");
        } catch (...) {
            // The launcher may stop the private service before late engine
            // destructors run; shutdown must remain noexcept.
        }
    }

    InferenceResult infer(
        const std::vector<unsigned char>& encoded_image,
        float confidence,
        float iou,
        const GroundingRequest& grounding
    ) override {
        // The public /api/infer route parsed multipart/form-data already. Send
        // only encoded image bytes internally and put scalar options in the
        // query string; Python returns the shared InferenceResult JSON schema.
        std::ostringstream path;
        path << "/models/" << config_.id << "/infer?confidence=" << std::setprecision(6) << confidence
             << "&iou=" << iou << "&maskThreshold=" << config_.mask_threshold;
        if (config_.task == ModelTask::VisionGrounding) {
            path << "&prompt=" << percent_encode(grounding.prompt)
                 << "&groundingMode=" << percent_encode(grounding.mode)
                 << "&generationMode=" << percent_encode(grounding.generation_mode);
        }
        const std::string body(reinterpret_cast<const char*>(encoded_image.data()), encoded_image.size());
        auto client = client_for_backend(config_.task == ModelTask::VisionGrounding ? 900 : 120);
        const json response = parse_response(client->Post(path.str(), body, "application/octet-stream"), "inference");
        return InferenceResult::from_json(response);
    }

    json metadata() const override { return metadata_; }

private:
    ModelConfig config_;
    json metadata_;
};

}  // namespace

std::unique_ptr<InferenceEngine> create_python_engine(const ModelConfig& config, const fs::path& artifact_path) {
    return std::make_unique<PythonEngine>(config, artifact_path);
}

json python_backend_health() {
    try {
        auto client = client_for_backend(5);
        return parse_response(client->Get("/health"), "Python backend health check");
    } catch (const std::exception& error) {
        return {{"status", "unavailable"}, {"error", error.what()}, {"formats", json::array()}};
    }
}

json discover_models() {
    auto client = client_for_backend(120);
    const json result = parse_response(client->Get("/models/discover"), "model discovery");
    if (!result.is_array()) throw std::runtime_error("model discovery returned an invalid model list");
    return result;
}

json model_format_capabilities() {
    json formats = json::array();
    const EngineRuntimeInfo native = engine_runtime_info();
    formats.push_back({
        {"format", "ncnn"}, {"extensions", {".param", ".bin"}},
        {"available", native.ncnn_available && native.vulkan_compiled && native.gpu_count > 0},
        {"runtime", "NCNN"}, {"accelerated", native.gpu_count > 0}, {"accelerator", "Vulkan"},
    });
    const json python = python_backend_health();
    for (const auto& item : python.value("formats", json::array())) formats.push_back(item);
    return formats;
}

}  // namespace runner
