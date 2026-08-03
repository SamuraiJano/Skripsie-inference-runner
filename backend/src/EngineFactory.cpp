#include "runner/Engine.hpp"

#include <stdexcept>

namespace runner {

std::unique_ptr<InferenceEngine> create_engine(
    const ModelConfig& config,
    const std::filesystem::path& primary_path,
    const std::filesystem::path& secondary_path
) {
    if (config.format == "ncnn") return create_ncnn_engine(config, primary_path, secondary_path);
    if (config.backend == "ultralytics") return create_python_engine(config, primary_path);
    throw std::runtime_error("no inference backend is registered for model format '" + config.format + "'");
}

}  // namespace runner
