#include "runner/DetectorDecoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace runner {
namespace {

const TensorData& tensor_for(const DetectorManifest& manifest, const std::vector<TensorData>& tensors, const std::string& role) {
    const auto spec = std::find_if(manifest.outputs.begin(), manifest.outputs.end(), [&](const DetectorOutputSpec& output) {
        return output.role == role;
    });
    if (spec == manifest.outputs.end()) throw std::invalid_argument("manifest does not define output role '" + role + "'");
    const auto tensor = std::find_if(tensors.begin(), tensors.end(), [&](const TensorData& value) { return value.name == spec->name; });
    if (tensor == tensors.end()) throw std::invalid_argument("runtime did not return required output tensor '" + spec->name + "'");
    if (!tensor_shape_matches(tensor->shape, spec->shape)) throw std::invalid_argument("output tensor '" + spec->name + "' shape does not match the manifest");
    if (tensor->values.size() != tensor_element_count(tensor->shape)) throw std::invalid_argument("output tensor '" + spec->name + "' data length does not match its shape");
    return *tensor;
}

float sigmoid(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

Detection mapped_detection(
    const DetectorManifest& manifest,
    const ImageTransform& transform,
    int class_id,
    float confidence,
    float a,
    float b,
    float c,
    float d
) {
    if (!std::isfinite(confidence) || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d)) {
        throw std::invalid_argument("detector output contains non-finite values");
    }
    if (manifest.coordinates == "normalized") {
        a *= transform.input_width;
        c *= transform.input_width;
        b *= transform.input_height;
        d *= transform.input_height;
    }

    float x1 = a;
    float y1 = b;
    float x2 = c;
    float y2 = d;
    if (manifest.box_format == "cxcywh") {
        x1 = a - c * 0.5F;
        y1 = b - d * 0.5F;
        x2 = a + c * 0.5F;
        y2 = b + d * 0.5F;
    } else if (manifest.box_format == "xywh") {
        x2 = a + c;
        y2 = b + d;
    }
    if (manifest.coordinates != "original-pixels") {
        x1 = (x1 - transform.pad_x) / transform.scale_x;
        x2 = (x2 - transform.pad_x) / transform.scale_x;
        y1 = (y1 - transform.pad_y) / transform.scale_y;
        y2 = (y2 - transform.pad_y) / transform.scale_y;
    }
    x1 = std::clamp(x1, 0.0F, static_cast<float>(transform.source_width));
    x2 = std::clamp(x2, 0.0F, static_cast<float>(transform.source_width));
    y1 = std::clamp(y1, 0.0F, static_cast<float>(transform.source_height));
    y2 = std::clamp(y2, 0.0F, static_cast<float>(transform.source_height));
    Detection detection;
    detection.class_id = class_id;
    detection.label = class_id >= 0 && static_cast<std::size_t>(class_id) < manifest.labels.size()
        ? manifest.labels[static_cast<std::size_t>(class_id)] : "class-" + std::to_string(class_id);
    detection.confidence = confidence;
    detection.x = x1;
    detection.y = y1;
    detection.width = std::max(0.0F, x2 - x1);
    detection.height = std::max(0.0F, y2 - y1);
    return detection;
}

float intersection_over_union(const Detection& left, const Detection& right) {
    const float intersection_left = std::max(left.x, right.x);
    const float intersection_top = std::max(left.y, right.y);
    const float intersection_right = std::min(left.x + left.width, right.x + right.width);
    const float intersection_bottom = std::min(left.y + left.height, right.y + right.height);
    const float intersection = std::max(0.0F, intersection_right - intersection_left)
        * std::max(0.0F, intersection_bottom - intersection_top);
    const float total = left.width * left.height + right.width * right.height - intersection;
    return total > 0.0F ? intersection / total : 0.0F;
}

std::vector<Detection> class_aware_nms(std::vector<Detection> detections, float threshold) {
    std::sort(detections.begin(), detections.end(), [](const Detection& left, const Detection& right) {
        return left.confidence > right.confidence;
    });
    std::vector<Detection> kept;
    std::vector<bool> removed(detections.size(), false);
    for (std::size_t index = 0; index < detections.size(); ++index) {
        if (removed[index]) continue;
        kept.push_back(detections[index]);
        for (std::size_t other = index + 1; other < detections.size(); ++other) {
            if (!removed[other] && detections[index].class_id == detections[other].class_id
                && intersection_over_union(detections[index], detections[other]) > threshold) removed[other] = true;
        }
    }
    return kept;
}

std::vector<Detection> decode_yolo(
    const DetectorManifest& manifest,
    const std::vector<TensorData>& tensors,
    const ImageTransform& transform,
    float floor
) {
    std::vector<Detection> detections;
    if (manifest.output_schema == "yolo-end2end") {
        const auto& tensor = tensor_for(manifest, tensors, "detections");
        if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 6) {
            throw std::invalid_argument("yolo-end2end output must be [1,predictions,6]");
        }
        for (int prediction = 0; prediction < tensor.shape[1]; ++prediction) {
            const std::size_t offset = static_cast<std::size_t>(prediction) * 6;
            const float confidence = tensor.values[offset + 4];
            if (confidence < floor) continue;
            const int class_id = static_cast<int>(std::lround(tensor.values[offset + 5]));
            if (class_id < 0 || static_cast<std::size_t>(class_id) >= manifest.labels.size()) continue;
            Detection detection = mapped_detection(manifest, transform, class_id, confidence,
                tensor.values[offset], tensor.values[offset + 1], tensor.values[offset + 2], tensor.values[offset + 3]);
            if (detection.width > 0.0F && detection.height > 0.0F) detections.push_back(std::move(detection));
        }
    } else {
        const auto& tensor = tensor_for(manifest, tensors, "predictions");
        if (tensor.shape.size() != 3 || tensor.shape[0] != 1) throw std::invalid_argument("yolo-raw output must have rank-3 batch-one shape");
        const int channels = manifest.tensor_orientation == "channels-first" ? tensor.shape[1] : tensor.shape[2];
        const int predictions = manifest.tensor_orientation == "channels-first" ? tensor.shape[2] : tensor.shape[1];
        const int score_start = manifest.has_objectness ? 5 : 4;
        const int expected_channels = score_start + static_cast<int>(manifest.labels.size());
        if (channels != expected_channels || predictions < 0) throw std::invalid_argument("yolo-raw channel count does not match box/objectness/class configuration");
        const auto at = [&](int prediction, int channel) {
            const std::size_t index = manifest.tensor_orientation == "channels-first"
                ? static_cast<std::size_t>(channel) * predictions + prediction
                : static_cast<std::size_t>(prediction) * channels + channel;
            return tensor.values[index];
        };
        for (int prediction = 0; prediction < predictions; ++prediction) {
            int class_id = 0;
            float class_score = -std::numeric_limits<float>::infinity();
            for (int candidate = 0; candidate < static_cast<int>(manifest.labels.size()); ++candidate) {
                if (at(prediction, score_start + candidate) > class_score) {
                    class_score = at(prediction, score_start + candidate);
                    class_id = candidate;
                }
            }
            const float confidence = class_score * (manifest.has_objectness ? at(prediction, 4) : 1.0F);
            if (confidence < floor) continue;
            Detection detection = mapped_detection(manifest, transform, class_id, confidence,
                at(prediction, 0), at(prediction, 1), at(prediction, 2), at(prediction, 3));
            if (detection.width > 0.0F && detection.height > 0.0F) detections.push_back(std::move(detection));
        }
    }
    return manifest.nms == "class-aware" ? class_aware_nms(std::move(detections), manifest.nms_iou) : detections;
}

std::vector<Detection> decode_dfine(
    const DetectorManifest& manifest,
    const std::vector<TensorData>& tensors,
    const ImageTransform& transform,
    float floor
) {
    std::vector<Detection> detections;
    if (manifest.output_schema == "dfine-fused") {
        const auto& labels = tensor_for(manifest, tensors, "labels");
        const auto& boxes = tensor_for(manifest, tensors, "boxes");
        const auto& scores = tensor_for(manifest, tensors, "scores");
        if (labels.shape.size() != 2 || scores.shape != labels.shape || boxes.shape.size() != 3 || boxes.shape[0] != 1
            || boxes.shape[1] != labels.shape[1] || boxes.shape[2] != 4) {
            throw std::invalid_argument("dfine-fused outputs must be labels [1,K], boxes [1,K,4], scores [1,K]");
        }
        for (int index = 0; index < labels.shape[1]; ++index) {
            const float confidence = scores.values[static_cast<std::size_t>(index)];
            if (confidence < floor) continue;
            const int class_id = static_cast<int>(std::lround(labels.values[static_cast<std::size_t>(index)]));
            if (class_id < 0 || static_cast<std::size_t>(class_id) >= manifest.labels.size()) continue;
            const std::size_t offset = static_cast<std::size_t>(index) * 4;
            Detection detection = mapped_detection(manifest, transform, class_id, confidence,
                boxes.values[offset], boxes.values[offset + 1], boxes.values[offset + 2], boxes.values[offset + 3]);
            if (detection.width > 0.0F && detection.height > 0.0F) detections.push_back(std::move(detection));
        }
    } else {
        const auto& logits = tensor_for(manifest, tensors, "logits");
        const auto& boxes = tensor_for(manifest, tensors, "boxes");
        if (logits.shape.size() != 3 || boxes.shape.size() != 3 || logits.shape[0] != 1 || boxes.shape[0] != 1
            || logits.shape[1] != boxes.shape[1] || logits.shape[2] != static_cast<int>(manifest.labels.size()) || boxes.shape[2] != 4) {
            throw std::invalid_argument("dfine-raw outputs must be logits [1,Q,classes] and boxes [1,Q,4]");
        }
        for (int query = 0; query < logits.shape[1]; ++query) {
            int class_id = 0;
            float confidence = -1.0F;
            for (int candidate = 0; candidate < logits.shape[2]; ++candidate) {
                const float score = sigmoid(logits.values[static_cast<std::size_t>(query) * logits.shape[2] + candidate]);
                if (score > confidence) {
                    confidence = score;
                    class_id = candidate;
                }
            }
            if (confidence < floor) continue;
            const std::size_t offset = static_cast<std::size_t>(query) * 4;
            Detection detection = mapped_detection(manifest, transform, class_id, confidence,
                boxes.values[offset], boxes.values[offset + 1], boxes.values[offset + 2], boxes.values[offset + 3]);
            if (detection.width > 0.0F && detection.height > 0.0F) detections.push_back(std::move(detection));
        }
    }
    return detections;
}

}  // namespace

std::vector<Detection> decode_detector_outputs(
    const DetectorManifest& manifest,
    const std::vector<TensorData>& tensors,
    const ImageTransform& transform,
    float confidence_floor
) {
    if (transform.source_width <= 0 || transform.source_height <= 0 || transform.input_width != manifest.input.width
        || transform.input_height != manifest.input.height || transform.scale_x <= 0.0F || transform.scale_y <= 0.0F) {
        throw std::invalid_argument("image transform is invalid or does not match the detector input");
    }
    if (!(confidence_floor >= manifest.score_floor && confidence_floor <= 1.0F)) {
        throw std::invalid_argument("confidence floor must be between the manifest scoreFloor and 1");
    }
    return manifest.model_family == "yolo"
        ? decode_yolo(manifest, tensors, transform, confidence_floor)
        : decode_dfine(manifest, tensors, transform, confidence_floor);
}

}  // namespace runner
