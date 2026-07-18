#include "runner/ModelStore.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace runner {
namespace {

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
        if (!fs::exists(manifest_path) || !fs::exists(entry.path() / "model.param") || !fs::exists(entry.path() / "model.bin")) continue;
        try {
            ModelConfig config = ModelConfig::from_json(json::parse(read_text(manifest_path)));
            if (config.id.empty()) config.id = entry.path().filename().string();
            records_.push_back({std::move(config), entry.path(), model_size(entry.path())});
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
        auto engine = create_engine(record->config, record->directory / "model.param", record->directory / "model.bin");
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
    std::scoped_lock inference_lock(inference_mutex_);
    return engine->infer(image, confidence, iou);
}

std::string ModelStore::active_model_id() const {
    std::scoped_lock lock(state_mutex_);
    return active_id_;
}

void ModelStore::persist_active_id() const {
    write_file(root_ / "active-model.txt", active_id_, false);
}

}  // namespace runner
