#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "runner/Engine.hpp"
#include "runner/Types.hpp"

namespace runner {

class ModelStore {
public:
    ModelStore(std::filesystem::path models_root, std::filesystem::path state_root);

    nlohmann::json list() const;
    nlohmann::json rescan();
    nlohmann::json activate(const std::string& id);
    InferenceResult infer(
        const std::string& requested_model_id,
        const std::vector<unsigned char>& image,
        float confidence,
        float iou,
        const GroundingRequest& grounding = {}
    );
    InferenceResult infer_pipeline_model(const std::string& model_id, const std::vector<unsigned char>& image, float confidence, float iou);
    nlohmann::json benchmark(
        const std::string& requested_model_id,
        const std::vector<unsigned char>& image,
        float confidence,
        float iou,
        int warmup,
        int count
    );
    std::string active_model_id() const;

private:
    struct Record {
        ModelConfig config;
        std::filesystem::path primary_path;
        std::filesystem::path secondary_path;
        std::uintmax_t size_bytes = 0;
        std::string family;
        std::string relative_path;
        std::string quantization;
        std::string quantization_method;
        std::string status_reason;
        bool quantized = false;
        bool runtime_available = false;
        bool compatible = true;
    };

    std::filesystem::path models_root_;
    std::filesystem::path state_root_;
    std::vector<Record> records_;
    std::string active_id_;
    std::shared_ptr<InferenceEngine> active_engine_;
    std::unordered_map<std::string, std::shared_ptr<InferenceEngine>> pipeline_engines_;
    mutable std::mutex state_mutex_;
    std::mutex discovery_mutex_;
    std::mutex inference_mutex_;

    void restore_active_model();
    Record* find_record(const std::string& id);
    const Record* find_record(const std::string& id) const;
    nlohmann::json public_record(const Record& record) const;
    void persist_active_id() const;
};

}  // namespace runner
