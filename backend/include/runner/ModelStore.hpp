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
    explicit ModelStore(std::filesystem::path root);

    nlohmann::json list() const;
    nlohmann::json import_model(const nlohmann::json& manifest, const std::string& param, const std::string& bin);
    nlohmann::json import_artifact(
        const std::string& name,
        const std::string& task,
        const std::string& precision,
        const std::string& filename,
        const std::string& artifact,
        bool trusted_checkpoint,
        int input_width,
        int input_height
    );
    nlohmann::json activate(const std::string& id);
    void remove(const std::string& id);
    InferenceResult infer(const std::string& requested_model_id, const std::vector<unsigned char>& image, float confidence, float iou);
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
        std::filesystem::path directory;
        std::uintmax_t size_bytes = 0;
    };

    std::filesystem::path root_;
    std::vector<Record> records_;
    std::string active_id_;
    std::shared_ptr<InferenceEngine> active_engine_;
    std::unordered_map<std::string, std::shared_ptr<InferenceEngine>> pipeline_engines_;
    mutable std::mutex state_mutex_;
    std::mutex inference_mutex_;

    void load_from_disk();
    Record* find_record(const std::string& id);
    const Record* find_record(const std::string& id) const;
    nlohmann::json public_record(const Record& record) const;
    void persist_active_id() const;
};

}  // namespace runner
