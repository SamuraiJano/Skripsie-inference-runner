#include "runner/TrackingPipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace runner {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

void NormalizedRoi::validate() const {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height)
        || x < 0.0F || y < 0.0F || width <= 0.0F || height <= 0.0F || x + width > 1.0F || y + height > 1.0F) {
        throw std::invalid_argument("ROI x, y, width, and height must describe a positive normalized rectangle inside the frame");
    }
}

nlohmann::json TrackingFrameResult::to_json() const {
    nlohmann::json track_values = nlohmann::json::array();
    for (const auto& track : tracks) {
        track_values.push_back({
            {"trackId", track.track_id}, {"classId", track.class_id}, {"label", track.label},
            {"confidence", track.confidence}, {"x", track.x}, {"y", track.y},
            {"width", track.width}, {"height", track.height}, {"progress", track.progress},
            {"state", "tracked"}, {"detectionIndex", track.detection_index},
        });
    }
    return {
        {"sessionId", session_id}, {"frameId", frame_id}, {"modelId", model_id},
        {"modelFamily", model_family}, {"backend", backend},
        {"sourceWidth", source_width}, {"sourceHeight", source_height}, {"tracks", std::move(track_values)},
        {"timing", {{"decodeMs", timing.decode_ms}, {"preprocessMs", timing.preprocess_ms},
                    {"inferenceMs", timing.inference_ms}, {"postprocessMs", timing.detection_nms_ms},
                    {"trackingMs", timing.tracking_ms}, {"totalMs", total_ms}}},
    };
}

TrackingPipeline::TrackingPipeline(
    std::string session_id,
    std::string model_id,
    std::string model_family,
    std::string backend,
    NormalizedRoi roi,
    ByteTrackConfig tracker_config
) : session_id_(std::move(session_id)), model_id_(std::move(model_id)), model_family_(std::move(model_family)),
    backend_(std::move(backend)), roi_(roi), tracker_(tracker_config) {
    if (session_id_.empty() || model_id_.empty()) throw std::invalid_argument("tracking session and model IDs are required");
    if (model_family_ != "yolo" && model_family_ != "dfine") throw std::invalid_argument("tracking requires a YOLO or D-FINE detector");
    if (backend_ != "tflite" && backend_ != "tensorrt") throw std::invalid_argument("tracking requires a TFLite or TensorRT backend");
    roi_.validate();
}

TrackingFrameResult TrackingPipeline::process(std::uint64_t frame_id, double timestamp_ms, const InferenceResult& detector_result) {
    if (frame_id == 0) throw std::invalid_argument("frameId must be positive");
    if (last_frame_id_ != 0 && frame_id <= last_frame_id_) throw std::invalid_argument("stale or out-of-order tracking frame was rejected");
    if (timestamp_ms >= 0.0 && last_timestamp_ms_ >= 0.0 && timestamp_ms < last_timestamp_ms_) {
        throw std::invalid_argument("tracking timestamp moved backwards; reset the session before continuing");
    }
    if (detector_result.source_width <= 0 || detector_result.source_height <= 0) throw std::invalid_argument("detector result has invalid source dimensions");
    if (detector_result.model_id != model_id_ || detector_result.model_family != model_family_ || detector_result.backend != backend_) {
        throw std::invalid_argument("detector result does not match the tracking session model/backend; reset or start a new session");
    }

    const auto tracking_start = Clock::now();
    const float roi_left = roi_.x * detector_result.source_width;
    const float roi_top = roi_.y * detector_result.source_height;
    const float roi_right = (roi_.x + roi_.width) * detector_result.source_width;
    const float roi_bottom = (roi_.y + roi_.height) * detector_result.source_height;
    std::vector<ByteTrackDetection> input;
    input.reserve(detector_result.detections.size());
    for (std::size_t index = 0; index < detector_result.detections.size(); ++index) {
        const auto& detection = detector_result.detections[index];
        const float centre_x = detection.x + detection.width * 0.5F;
        const float centre_y = detection.y + detection.height * 0.5F;
        if (centre_x < roi_left || centre_x > roi_right || centre_y < roi_top || centre_y > roi_bottom) continue;
        input.push_back({
            {detection.x, detection.y, detection.x + detection.width, detection.y + detection.height},
            detection.confidence, detection.class_id, static_cast<int>(index),
        });
    }
    auto tracked = tracker_.update(input, detector_result.source_width, detector_result.source_height);
    tracker_.retire_past_bottom(roi_bottom);

    TrackingFrameResult result;
    result.session_id = session_id_;
    result.frame_id = frame_id;
    result.model_id = model_id_;
    result.model_family = model_family_;
    result.backend = backend_;
    result.source_width = detector_result.source_width;
    result.source_height = detector_result.source_height;
    result.timing = detector_result.timing;
    result.tracks.reserve(tracked.size());
    for (const auto& track : tracked) {
        if (track.detection_index < 0 || static_cast<std::size_t>(track.detection_index) >= detector_result.detections.size()) continue;
        const auto& detection = detector_result.detections[static_cast<std::size_t>(track.detection_index)];
        const float centre_y = detection.y + detection.height * 0.5F;
        result.tracks.push_back({
            track.track_id, track.class_id, detection.label, track.score,
            track.xyxy[0], track.xyxy[1], std::max(0.0F, track.xyxy[2] - track.xyxy[0]),
            std::max(0.0F, track.xyxy[3] - track.xyxy[1]),
            std::clamp((centre_y - roi_top) / std::max(roi_bottom - roi_top, 1.0F), 0.0F, 1.0F),
            track.detection_index,
        });
    }
    result.timing.tracking_ms = elapsed_ms(tracking_start, Clock::now());
    result.total_ms = detector_result.timing.backend_total_ms + result.timing.tracking_ms;
    last_frame_id_ = frame_id;
    last_timestamp_ms_ = timestamp_ms;
    visible_tracks_ = result.tracks.size();
    return result;
}

void TrackingPipeline::reset() {
    tracker_.reset();
    last_frame_id_ = 0;
    last_timestamp_ms_ = -1.0;
    visible_tracks_ = 0;
}

void TrackingPipeline::set_roi(NormalizedRoi roi) {
    roi.validate();
    if (roi.x != roi_.x || roi.y != roi_.y || roi.width != roi_.width || roi.height != roi_.height) {
        roi_ = roi;
        reset();
    }
}

void TrackingPipeline::set_model(std::string model_id, std::string model_family, std::string backend) {
    if (model_id != model_id_ || model_family != model_family_ || backend != backend_) {
        model_id_ = std::move(model_id);
        model_family_ = std::move(model_family);
        backend_ = std::move(backend);
        reset();
    }
}

nlohmann::json TrackingPipeline::status() const {
    return {
        {"sessionId", session_id_}, {"modelId", model_id_}, {"modelFamily", model_family_}, {"backend", backend_},
        {"lastFrameId", last_frame_id_ == 0 ? nlohmann::json(nullptr) : nlohmann::json(last_frame_id_)},
        {"visibleTracks", visible_tracks_}, {"activeTracks", tracker_.active_track_count()}, {"lostTracks", tracker_.lost_track_count()},
        {"roi", {{"x", roi_.x}, {"y", roi_.y}, {"width", roi_.width}, {"height", roi_.height}}},
    };
}

const std::string& TrackingPipeline::session_id() const { return session_id_; }
const std::string& TrackingPipeline::model_id() const { return model_id_; }
const NormalizedRoi& TrackingPipeline::roi() const { return roi_; }

ByteTrackConfig byte_track_config_from_json(const nlohmann::json& value) {
    ByteTrackConfig config;
    if (value.is_null()) return config;
    if (!value.is_object()) throw std::invalid_argument("tracking settings must be an object");
    config.track_thresh = value.value("trackThresh", config.track_thresh);
    config.unmatched_thresh = value.value("unmatchedThresh", config.unmatched_thresh);
    config.detrack_thresh = value.value("detrackThresh", config.detrack_thresh);
    config.tracking_thresh = value.value("trackingThresh", config.tracking_thresh);
    config.track_buffer = value.value("trackBuffer", config.track_buffer);
    config.max_age = value.value("maxAge", config.max_age);
    config.min_hits = value.value("minHits", config.min_hits);
    config.class_aware = value.value("classAware", config.class_aware);
    config.iou_weight = value.value("iouWeight", config.iou_weight);
    config.drag = value.value("drag", config.drag);
    config.velocity_alpha = value.value("velocityAlpha", config.velocity_alpha);
    config.direction_gate = value.value("directionGate", config.direction_gate);
    config.maximum_backward_ratio = value.value("maximumBackwardRatio", config.maximum_backward_ratio);
    config.maximum_sideways_ratio = value.value("maximumSidewaysRatio", config.maximum_sideways_ratio);
    config.validate();
    return config;
}

NormalizedRoi roi_from_json(const nlohmann::json& value) {
    NormalizedRoi roi;
    if (!value.is_object()) throw std::invalid_argument("roi must be an object");
    roi.x = value.value("x", roi.x);
    roi.y = value.value("y", roi.y);
    roi.width = value.value("width", roi.width);
    roi.height = value.value("height", roi.height);
    roi.validate();
    return roi;
}

}  // namespace runner
