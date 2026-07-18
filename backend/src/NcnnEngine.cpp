#include "runner/Engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include <stb_image.h>

#if RUNNER_WITH_NCNN
#include <cpu.h>
#include <net.h>
#endif

namespace runner {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#if RUNNER_WITH_NCNN

struct DecodedImage {
    int width = 0;
    int height = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{nullptr, stbi_image_free};
};

struct Letterbox {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
};

struct MatrixView {
    const ncnn::Mat* source = nullptr;
    int rows = 0;
    int columns = 0;
    enum class Layout { TwoDimensional, OneChannel, ChannelsAsRows } layout = Layout::TwoDimensional;

    float at(int row, int column) const {
        switch (layout) {
            case Layout::TwoDimensional: return source->row(row)[column];
            case Layout::OneChannel: return source->channel(0).row(row)[column];
            case Layout::ChannelsAsRows: return source->channel(row).row(0)[column];
        }
        return 0.0F;
    }
};

DecodedImage decode(const std::vector<unsigned char>& encoded) {
    if (encoded.empty()) throw std::invalid_argument("uploaded image is empty");
    int width = 0;
    int height = 0;
    int channels = 0;
    auto* raw = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &channels, 3);
    if (!raw) throw std::invalid_argument(std::string("could not decode image: ") + stbi_failure_reason());
    if (width <= 0 || height <= 0 || width > 12000 || height > 12000) {
        stbi_image_free(raw);
        throw std::invalid_argument("image dimensions are outside the supported range");
    }
    DecodedImage image;
    image.width = width;
    image.height = height;
    image.pixels.reset(raw);
    return image;
}

ncnn::Mat preprocess(const DecodedImage& image, const ModelConfig& config, Letterbox& letterbox) {
    letterbox.scale = std::min(
        static_cast<float>(config.input_width) / static_cast<float>(image.width),
        static_cast<float>(config.input_height) / static_cast<float>(image.height));
    const int resized_width = std::max(1, static_cast<int>(std::round(image.width * letterbox.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(image.height * letterbox.scale)));
    letterbox.pad_x = (config.input_width - resized_width) / 2;
    letterbox.pad_y = (config.input_height - resized_height) / 2;

    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        image.pixels.get(), ncnn::Mat::PIXEL_RGB,
        image.width, image.height, resized_width, resized_height);
    if (resized.empty()) throw std::runtime_error("failed to resize input image");

    ncnn::Mat input(config.input_width, config.input_height, 3);
    input.fill(114.0F);
    for (int channel = 0; channel < 3; ++channel) {
        const ncnn::Mat source_channel = resized.channel(channel);
        ncnn::Mat target_channel = input.channel(channel);
        for (int row = 0; row < resized_height; ++row) {
            const float* source = source_channel.row(row);
            float* target = target_channel.row(row + letterbox.pad_y) + letterbox.pad_x;
            std::memcpy(target, source, static_cast<std::size_t>(resized_width) * sizeof(float));
        }
    }
    input.substract_mean_normalize(config.mean.data(), config.norm.data());
    return input;
}

MatrixView as_matrix(const ncnn::Mat& value) {
    if (value.elempack != 1 || value.elembits() != 32) {
        throw std::runtime_error("YOLO output must be an unpacked float32 tensor");
    }
    if (value.dims == 2) return {&value, value.h, value.w, MatrixView::Layout::TwoDimensional};
    if (value.dims == 3 && value.c == 1) return {&value, value.h, value.w, MatrixView::Layout::OneChannel};
    if (value.dims == 3 && value.h == 1) return {&value, value.c, value.w, MatrixView::Layout::ChannelsAsRows};
    throw std::runtime_error("unsupported YOLO output shape; use raw tensors to inspect the graph outputs");
}

float box_iou(const Detection& first, const Detection& second) {
    const float left = std::max(first.x, second.x);
    const float top = std::max(first.y, second.y);
    const float right = std::min(first.x + first.width, second.x + second.width);
    const float bottom = std::min(first.y + first.height, second.y + second.height);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float area = first.width * first.height + second.width * second.height - intersection;
    return area <= 0.0F ? 0.0F : intersection / area;
}

std::vector<Detection> decode_yolo(
    const ncnn::Mat& output,
    const ModelConfig& config,
    const DecodedImage& image,
    const Letterbox& letterbox,
    float confidence_threshold,
    float iou_threshold
) {
    const MatrixView matrix = as_matrix(output);
    const int class_count = static_cast<int>(config.labels.size());
    const int minimum_features = 4 + class_count;
    bool transposed = false;

    if (matrix.columns == 6) {
        transposed = false;
    } else if (matrix.rows == 6) {
        transposed = true;
    } else if (matrix.columns >= minimum_features && matrix.columns <= 512) {
        transposed = false;
    } else if (matrix.rows >= minimum_features && matrix.rows <= 512) {
        transposed = true;
    } else {
        throw std::runtime_error("cannot identify YOLO feature axis; verify the label count and output blob");
    }

    const int rows = transposed ? matrix.columns : matrix.rows;
    const int columns = transposed ? matrix.rows : matrix.columns;
    auto feature = [&](int row, int column) {
        return transposed ? matrix.at(column, row) : matrix.at(row, column);
    };

    std::vector<Detection> candidates;
    candidates.reserve(std::min(rows, 3000));
    for (int row = 0; row < rows; ++row) {
        int class_id = 0;
        float score = 0.0F;
        float x1 = 0.0F;
        float y1 = 0.0F;
        float x2 = 0.0F;
        float y2 = 0.0F;

        if (columns == 6) {
            score = feature(row, 4);
            class_id = static_cast<int>(std::round(feature(row, 5)));
            x1 = feature(row, 0);
            y1 = feature(row, 1);
            x2 = feature(row, 2);
            y2 = feature(row, 3);
        } else {
            if (columns < minimum_features) continue;
            for (int current_class = 0; current_class < class_count; ++current_class) {
                const float current_score = feature(row, 4 + current_class);
                if (current_score > score) {
                    score = current_score;
                    class_id = current_class;
                }
            }
            const float center_x = feature(row, 0);
            const float center_y = feature(row, 1);
            const float width = feature(row, 2);
            const float height = feature(row, 3);
            x1 = center_x - width * 0.5F;
            y1 = center_y - height * 0.5F;
            x2 = center_x + width * 0.5F;
            y2 = center_y + height * 0.5F;
        }

        if (!std::isfinite(score) || score < confidence_threshold || class_id < 0 || class_id >= class_count) continue;
        x1 = (x1 - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
        y1 = (y1 - static_cast<float>(letterbox.pad_y)) / letterbox.scale;
        x2 = (x2 - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
        y2 = (y2 - static_cast<float>(letterbox.pad_y)) / letterbox.scale;
        x1 = std::clamp(x1, 0.0F, static_cast<float>(image.width));
        y1 = std::clamp(y1, 0.0F, static_cast<float>(image.height));
        x2 = std::clamp(x2, 0.0F, static_cast<float>(image.width));
        y2 = std::clamp(y2, 0.0F, static_cast<float>(image.height));
        if (x2 <= x1 || y2 <= y1) continue;
        candidates.push_back({class_id, config.labels[static_cast<std::size_t>(class_id)], score, x1, y1, x2 - x1, y2 - y1});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Detection& first, const Detection& second) {
        return first.confidence > second.confidence;
    });
    if (candidates.size() > 3000) candidates.resize(3000);

    std::vector<Detection> kept;
    kept.reserve(std::min(static_cast<std::size_t>(config.max_detections), candidates.size()));
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& previous : kept) {
            if (candidate.class_id == previous.class_id && box_iou(candidate, previous) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
        if (static_cast<int>(kept.size()) >= config.max_detections) break;
    }
    return kept;
}

std::vector<float> flat_values(const ncnn::Mat& value, std::size_t limit = std::numeric_limits<std::size_t>::max()) {
    if (value.elembits() != 32) throw std::runtime_error("only float32 output tensors are currently supported");
    const std::size_t count = value.total() * static_cast<std::size_t>(value.elempack);
    const std::size_t copied = std::min(count, limit);
    const float* data = static_cast<const float*>(value.data);
    return std::vector<float>(data, data + copied);
}

std::vector<int> tensor_shape(const ncnn::Mat& value) {
    const int packed_width = value.w * value.elempack;
    if (value.dims == 1) return {packed_width};
    if (value.dims == 2) return {value.h, packed_width};
    if (value.dims == 3) return {value.c, value.h, packed_width};
    if (value.dims == 4) return {value.c, value.d, value.h, packed_width};
    return {};
}

std::vector<ClassificationScore> classify(const ncnn::Mat& output, const ModelConfig& config) {
    auto scores = flat_values(output);
    if (scores.empty()) throw std::runtime_error("classification output is empty");
    if (config.apply_softmax) {
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0F;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            sum += score;
        }
        if (sum > 0.0F) for (float& score : scores) score /= sum;
    }

    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    const std::size_t keep = std::min(static_cast<std::size_t>(config.top_k), order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep), order.end(), [&](int first, int second) {
        return scores[static_cast<std::size_t>(first)] > scores[static_cast<std::size_t>(second)];
    });

    std::vector<ClassificationScore> result;
    result.reserve(keep);
    for (std::size_t index = 0; index < keep; ++index) {
        const int class_id = order[index];
        const std::string label = class_id < static_cast<int>(config.labels.size())
            ? config.labels[static_cast<std::size_t>(class_id)]
            : "class_" + std::to_string(class_id);
        result.push_back({class_id, label, scores[static_cast<std::size_t>(class_id)]});
    }
    return result;
}

class NcnnEngine final : public InferenceEngine {
public:
    NcnnEngine(ModelConfig config, const std::filesystem::path& param_path, const std::filesystem::path& bin_path)
        : config_(std::move(config)) {
        ncnn::set_cpu_powersave(0);
        net_.opt.num_threads = std::clamp(config_.threads, 1, 4);
        net_.opt.lightmode = true;
        net_.opt.use_vulkan_compute = false;
        net_.opt.use_packing_layout = true;
        net_.opt.use_winograd_convolution = true;
        net_.opt.use_sgemm_convolution = true;
        net_.opt.use_int8_inference = true;
        net_.opt.use_int8_storage = true;
        net_.opt.use_fp16_storage = false;
        net_.opt.use_fp16_arithmetic = false;
        net_.opt.use_bf16_storage = false;
        net_.opt.use_mapped_model_loading = true;

        if (net_.load_param(param_path.string().c_str()) != 0) throw std::runtime_error("NCNN could not load model.param");
        if (net_.load_model(bin_path.string().c_str()) != 0) throw std::runtime_error("NCNN could not load model.bin");
    }

    InferenceResult infer(const std::vector<unsigned char>& encoded_image, float confidence, float iou) override {
        const auto total_start = Clock::now();
        const auto decode_start = total_start;
        const DecodedImage image = decode(encoded_image);
        const auto decode_end = Clock::now();

        Letterbox letterbox;
        const auto preprocess_start = decode_end;
        ncnn::Mat input = preprocess(image, config_, letterbox);
        const auto preprocess_end = Clock::now();

        ncnn::Extractor extractor = net_.create_extractor();
        if (extractor.input(config_.input_blob.c_str(), input) != 0) {
            throw std::runtime_error("NCNN could not set input blob '" + config_.input_blob + "'");
        }
        std::vector<std::pair<std::string, ncnn::Mat>> outputs;
        outputs.reserve(config_.output_blobs.size());
        const auto inference_start = Clock::now();
        for (const auto& name : config_.output_blobs) {
            ncnn::Mat value;
            if (extractor.extract(name.c_str(), value) != 0) {
                throw std::runtime_error("NCNN could not extract output blob '" + name + "'");
            }
            outputs.emplace_back(name, std::move(value));
        }
        const auto inference_end = Clock::now();

        const auto postprocess_start = inference_end;
        InferenceResult result;
        result.model_id = config_.id;
        result.task = config_.task;
        result.source_width = image.width;
        result.source_height = image.height;
        if (config_.task == ModelTask::YoloDetect) {
            result.detections = decode_yolo(outputs.front().second, config_, image, letterbox, confidence, iou);
        } else if (config_.task == ModelTask::Classification) {
            result.classifications = classify(outputs.front().second, config_);
        } else {
            for (const auto& [name, value] : outputs) {
                result.tensors.push_back({name, tensor_shape(value), flat_values(value, 64), value.total() * static_cast<std::size_t>(value.elempack)});
            }
        }
        const auto postprocess_end = Clock::now();

        result.timing.decode_ms = milliseconds(decode_start, decode_end);
        result.timing.preprocess_ms = milliseconds(preprocess_start, preprocess_end);
        result.timing.inference_ms = milliseconds(inference_start, inference_end);
        result.timing.postprocess_ms = milliseconds(postprocess_start, postprocess_end);
        result.timing.total_ms = milliseconds(total_start, postprocess_end);
        return result;
    }

private:
    ModelConfig config_;
    ncnn::Net net_;
};

#endif

}  // namespace

std::unique_ptr<InferenceEngine> create_engine(
    const ModelConfig& config,
    const std::filesystem::path& param_path,
    const std::filesystem::path& bin_path
) {
#if RUNNER_WITH_NCNN
    return std::make_unique<NcnnEngine>(config, param_path, bin_path);
#else
    (void)config;
    (void)param_path;
    (void)bin_path;
    throw std::runtime_error("this binary was built without NCNN support");
#endif
}

bool ncnn_available() {
#if RUNNER_WITH_NCNN
    return true;
#else
    return false;
#endif
}

std::string engine_name() {
#if RUNNER_WITH_NCNN
    return "NCNN 20260526 / CPU";
#else
    return "NCNN unavailable";
#endif
}

}  // namespace runner
