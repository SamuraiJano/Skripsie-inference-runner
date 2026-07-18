#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "runner/Engine.hpp"
#include "runner/ModelStore.hpp"
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
    std::string host = "0.0.0.0";
    int port = 8080;
    fs::path data = "data/models";
    fs::path static_files = RUNNER_FRONTEND_DIST;
};

httplib::Server* active_server = nullptr;

class InlineTaskQueue final : public httplib::TaskQueue {
public:
    bool enqueue(std::function<void()> task) override {
        task();
        return true;
    }

    void shutdown() override {}
};

void stop_server(int) {
    if (active_server) active_server->stop();
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help" || key == "-h") {
            std::cout
                << "Edge Runner " << RUNNER_VERSION << "\n\n"
                << "Usage: edge_runner [options]\n"
                << "  --host <address>   Bind address (default 0.0.0.0)\n"
                << "  --port <number>    HTTP port (default 8080)\n"
                << "  --data <path>      Model storage directory\n"
                << "  --static <path>    Built React dist directory\n";
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--host") args.host = value;
        else if (key == "--port") args.port = std::stoi(value);
        else if (key == "--data") args.data = value;
        else if (key == "--static") args.static_files = value;
        else throw std::invalid_argument("unknown argument: " + key);
    }
    if (args.port < 1 || args.port > 65535) throw std::invalid_argument("port must be between 1 and 65535");
    return args;
}

std::string architecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm32";
#elif defined(__x86_64__) || defined(_M_X64)
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
            if (received > 300ULL * 1024ULL * 1024ULL || parts.empty()) {
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        runner::ModelStore models(args.data);
        const auto started_at = std::chrono::steady_clock::now();
        const unsigned int cores = std::max(1U, std::thread::hardware_concurrency());
        const unsigned int recommended_threads = std::min(4U, cores);

        httplib::Server server;
        active_server = &server;
        std::signal(SIGINT, stop_server);
        std::signal(SIGTERM, stop_server);
        server.set_payload_max_length(300ULL * 1024ULL * 1024ULL);
        server.set_read_timeout(120, 0);
        server.set_write_timeout(120, 0);
        // The runner intentionally admits one HTTP connection at a time. This
        // prevents parallel uploads or requests from competing with NCNN's own
        // four-worker CPU pool on a Raspberry Pi 4.
        server.new_task_queue = [] { return new InlineTaskQueue(); };
        server.set_default_headers({
            {"X-Content-Type-Options", "nosniff"},
            {"X-Frame-Options", "DENY"},
            {"Referrer-Policy", "no-referrer"},
        });

        server.Get("/api/health", [&](const httplib::Request&, httplib::Response& response) {
            const double uptime = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
            const std::string active_model = models.active_model_id();
            respond_json(response, {
                {"status", "ok"}, {"version", RUNNER_VERSION}, {"engine", runner::engine_name()},
                {"ncnnAvailable", runner::ncnn_available()}, {"architecture", architecture()},
                {"logicalCores", cores}, {"recommendedThreads", recommended_threads},
                {"uptimeSeconds", uptime}, {"activeModelId", active_model.empty() ? json(nullptr) : json(active_model)},
            });
        });

        server.Get("/api/models", [&](const httplib::Request&, httplib::Response& response) {
            respond_json(response, models.list());
        });

        server.Post("/api/models", [&](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& reader) {
            guarded(response, [&] {
                const auto parts = read_multipart(request, reader);
                const auto& manifest = required_part(parts, "manifest");
                const auto& param = required_part(parts, "param");
                const auto& bin = required_part(parts, "bin");
                if (manifest.content.size() > 1024 * 1024) throw std::invalid_argument("manifest is too large");
                if (param.content.size() > 32 * 1024 * 1024) throw std::invalid_argument(".param file is too large");
                if (bin.content.size() > 256 * 1024 * 1024) throw std::invalid_argument(".bin file exceeds the 256 MB Pi profile limit");
                respond_json(response, models.import_model(json::parse(manifest.content), param.content, bin.content), 201);
            });
        });

        server.Post(R"(/api/models/([A-Za-z0-9_-]+)/activate)", [&](const httplib::Request& request, httplib::Response& response) {
            guarded(response, [&] { respond_json(response, models.activate(request.matches[1].str())); });
        });

        server.Delete(R"(/api/models/([A-Za-z0-9_-]+))", [&](const httplib::Request& request, httplib::Response& response) {
            guarded(response, [&] {
                const std::string id = request.matches[1].str();
                models.remove(id);
                respond_json(response, {{"deleted", id}});
            });
        });

        server.Post("/api/infer", [&](const httplib::Request& request, httplib::Response& response, const httplib::ContentReader& reader) {
            guarded(response, [&] {
                const auto parts = read_multipart(request, reader);
                const auto& image_part = required_part(parts, "image");
                if (image_part.content.size() > 25 * 1024 * 1024) throw std::invalid_argument("image exceeds the 25 MB limit");
                const std::string model_id = optional_part(parts, "modelId");
                const float confidence = float_part(parts, "confidence", 0.25F);
                const float iou = float_part(parts, "iou", 0.50F);
                const std::vector<unsigned char> bytes(image_part.content.begin(), image_part.content.end());
                respond_json(response, models.infer(model_id, bytes, confidence, iou).to_json());
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

        std::cout << "Edge Runner " << RUNNER_VERSION << '\n'
                  << "  engine: " << runner::engine_name() << '\n'
                  << "  architecture: " << architecture() << " (" << cores << " logical cores)\n"
                  << "  models: " << fs::absolute(args.data) << '\n'
                  << "  dashboard: http://" << (args.host == "0.0.0.0" ? "127.0.0.1" : args.host) << ':' << args.port << "\n";

        if (!server.listen(args.host, args.port)) throw std::runtime_error("could not bind HTTP server");
        active_server = nullptr;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
