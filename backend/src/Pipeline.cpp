#include "runner/Pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace runner {
namespace {

float iou(const Detection& left, const Detection& right) {
    const float x1 = std::max(left.x, right.x);
    const float y1 = std::max(left.y, right.y);
    const float x2 = std::min(left.x + left.width, right.x + right.width);
    const float y2 = std::min(left.y + left.height, right.y + right.height);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float total = left.width * left.height + right.width * right.height - intersection;
    return total <= 0.0F ? 0.0F : intersection / total;
}

void match_stage(
    std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    const std::vector<int>& detection_indices,
    std::vector<bool>& track_used,
    std::vector<bool>& detection_used,
    float threshold,
    std::uint64_t frame_id
) {
    while (true) {
        int best_track = -1;
        int best_detection = -1;
        float best_overlap = threshold;
        for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
            if (track_used[track_index]) continue;
            for (int detection_index : detection_indices) {
                if (detection_used[static_cast<std::size_t>(detection_index)]) continue;
                const float overlap = iou(tracks[track_index].detection, detections[static_cast<std::size_t>(detection_index)]);
                if (overlap > best_overlap) {
                    best_overlap = overlap;
                    best_track = static_cast<int>(track_index);
                    best_detection = detection_index;
                }
            }
        }
        if (best_track < 0) break;
        Track& track = tracks[static_cast<std::size_t>(best_track)];
        track.detection = detections[static_cast<std::size_t>(best_detection)];
        track.detection.track_id = track.id;
        track.last_frame_id = frame_id;
        ++track.hits;
        track.missed = 0;
        track_used[static_cast<std::size_t>(best_track)] = true;
        detection_used[static_cast<std::size_t>(best_detection)] = true;
    }
}

}  // namespace

ByteTracker::ByteTracker(float high_threshold, float low_threshold, float match_iou, int max_missed)
    : high_threshold_(high_threshold), low_threshold_(low_threshold), match_iou_(match_iou), max_missed_(max_missed) {
    if (!(low_threshold_ >= 0.0F && high_threshold_ >= low_threshold_ && high_threshold_ <= 1.0F)) {
        throw std::invalid_argument("ByteTrack confidence thresholds are invalid");
    }
}

std::vector<Detection> ByteTracker::update(std::uint64_t frame_id, const std::vector<Detection>& detections) {
    std::vector<int> high;
    std::vector<int> low;
    for (std::size_t index = 0; index < detections.size(); ++index) {
        if (detections[index].confidence >= high_threshold_) high.push_back(static_cast<int>(index));
        else if (detections[index].confidence >= low_threshold_) low.push_back(static_cast<int>(index));
    }
    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> detection_used(detections.size(), false);
    match_stage(tracks_, detections, high, track_used, detection_used, match_iou_, frame_id);
    match_stage(tracks_, detections, low, track_used, detection_used, match_iou_, frame_id);
    for (std::size_t index = 0; index < tracks_.size(); ++index) if (!track_used[index]) ++tracks_[index].missed;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& track) { return track.missed > max_missed_; }), tracks_.end());
    for (int index : high) {
        if (detection_used[static_cast<std::size_t>(index)]) continue;
        Detection detection = detections[static_cast<std::size_t>(index)];
        detection.track_id = next_id_;
        tracks_.push_back({next_id_++, detection, frame_id, 1, 0});
    }
    std::vector<Detection> output;
    for (const auto& track : tracks_) if (track.last_frame_id == frame_id) output.push_back(track.detection);
    return output;
}

EvidenceAggregator::EvidenceAggregator(int minimum_observations, float minimum_margin)
    : minimum_observations_(minimum_observations), minimum_margin_(minimum_margin) {
    if (minimum_observations_ < 1 || minimum_margin_ < 0.0F) throw std::invalid_argument("grading rule is invalid");
}

GradeDecision EvidenceAggregator::add(const GradeObservation& observation) {
    if (observation.track_id < 0 || observation.class_id < 0 || !std::isfinite(observation.confidence)) {
        throw std::invalid_argument("grade observation is invalid");
    }
    auto& evidence = evidence_[observation.track_id];
    ++evidence.count;
    evidence.weights[observation.class_id] += std::clamp(observation.confidence, 0.0F, 1.0F);
    return decision(observation.track_id);
}

GradeDecision EvidenceAggregator::decision(int track_id) const {
    GradeDecision result;
    result.track_id = track_id;
    const auto found = evidence_.find(track_id);
    if (found == evidence_.end()) return result;
    result.observations = found->second.count;
    std::vector<std::pair<int, float>> weights(found->second.weights.begin(), found->second.weights.end());
    std::sort(weights.begin(), weights.end(), [](const auto& left, const auto& right) { return left.second > right.second; });
    if (weights.empty()) return result;
    result.class_id = weights.front().first;
    result.winning_weight = weights.front().second;
    const float runner_up = weights.size() > 1 ? weights[1].second : 0.0F;
    const float total = std::max(1e-6F, result.winning_weight + runner_up);
    if (result.observations >= minimum_observations_ && (result.winning_weight - runner_up) / total >= minimum_margin_) {
        result.state = "class_" + std::to_string(result.class_id + 1);
    }
    return result;
}

AsyncFruitPipeline::AsyncFruitPipeline(
    Detector detector,
    Admission admission,
    Segmenter segmenter,
    Output output,
    std::uint64_t maximum_result_lag,
    PipelineMode mode
)
    : detector_(std::move(detector)), admission_(std::move(admission)), segmenter_(std::move(segmenter)),
      output_(std::move(output)), maximum_result_lag_(maximum_result_lag), mode_(mode) {}

AsyncFruitPipeline::~AsyncFruitPipeline() { stop(); }

void AsyncFruitPipeline::start() {
    if (running_.exchange(true)) return;
    detector_thread_ = std::thread([this] { detector_loop(); });
    segmentation_thread_ = std::thread([this] { segmentation_loop(); });
}

void AsyncFruitPipeline::submit(PipelineFrame frame) {
    if (!running_) throw std::runtime_error("pipeline is not running");
    const std::uint64_t previous = latest_frame_id_.load();
    if (frame.frame_id <= previous) throw std::invalid_argument("frame IDs must be strictly monotonic");
    latest_frame_id_ = frame.frame_id;
    capture_queue_.push(std::move(frame));
}

void AsyncFruitPipeline::stop() {
    if (!running_.exchange(false)) return;
    capture_queue_.stop();
    segmentation_queue_.stop();
    if (detector_thread_.joinable()) detector_thread_.join();
    if (segmentation_thread_.joinable()) segmentation_thread_.join();
}

void AsyncFruitPipeline::detector_loop() {
    PipelineFrame frame;
    while (capture_queue_.pop(frame)) {
        std::vector<Detection> detected;
        {
            std::scoped_lock inference_lock(inference_mutex_);
            detected = detector_(frame);
        }
        auto tracked = tracker_.update(frame.frame_id, detected);
        {
            std::scoped_lock lock(latest_mutex_);
            latest_frame_ = frame;
            latest_detections_ = tracked;
        }
        if (auto job = admission_(frame, tracked)) segmentation_queue_.push(std::move(*job));
        output_(frame, tracked, {});
    }
}

void AsyncFruitPipeline::segmentation_loop() {
    SegmentationJob job;
    while (segmentation_queue_.pop(job)) {
        if (job.frame_id + maximum_result_lag_ < latest_frame_id_.load()) continue;
        if (mode_ == PipelineMode::DetectorPriorityTimeSliced && capture_queue_.has_pending()) {
            // Preserve the latest job and let the detector consume the newest frame first.
            segmentation_queue_.push(std::move(job));
            std::this_thread::yield();
            continue;
        }
        std::vector<GradeObservation> observations;
        {
            std::scoped_lock inference_lock(inference_mutex_);
            // Recheck after waiting for a detector inference. Results that are already
            // stale must not spend another model invocation.
            if (job.frame_id + maximum_result_lag_ < latest_frame_id_.load()) continue;
            observations = segmenter_(job);
        }
        std::vector<GradeDecision> decisions;
        for (auto observation : observations) {
            observation.frame_id = job.frame_id;
            if (observation.track_id < 0) observation.track_id = job.track_id;
            decisions.push_back(aggregator_.add(observation));
        }
        if (job.frame_id + maximum_result_lag_ < latest_frame_id_.load()) continue;
        PipelineFrame output_frame;
        std::vector<Detection> output_detections;
        {
            std::scoped_lock lock(latest_mutex_);
            output_frame = latest_frame_;
            output_detections = latest_detections_;
        }
        output_(output_frame, output_detections, decisions);
    }
}

std::string track_state_name(TrackState state) {
    switch (state) {
        case TrackState::New: return "NEW";
        case TrackState::Acquiring: return "ACQUIRING";
        case TrackState::WaitingForSegmentation: return "WAITING_FOR_SEGMENTATION";
        case TrackState::WaitingForClassification: return "WAITING_FOR_CLASSIFICATION";
        case TrackState::Classified: return "CLASSIFIED";
        case TrackState::Exited: return "EXITED";
        case TrackState::Failed: return "FAILED";
    }
    return "FAILED";
}

double Coordinate::pixels(int extent) const {
    return normalized ? value * static_cast<double>(extent) : value;
}

namespace {

Coordinate coordinate_from_json(const json& value, const Coordinate& fallback) {
    if (value.is_number()) {
        const double coordinate = value.get<double>();
        return {coordinate, coordinate >= 0.0 && coordinate <= 1.0};
    }
    if (!value.is_object()) return fallback;
    return {value.value("value", fallback.value), value.value("normalized", fallback.normalized)};
}

json coordinate_to_json(const Coordinate& coordinate) {
    return {{"value", coordinate.value}, {"normalized", coordinate.normalized}};
}

json detection_json(const Detection& detection) {
    return {
        {"classId", detection.class_id}, {"label", detection.label}, {"confidence", detection.confidence},
        {"x", detection.x}, {"y", detection.y}, {"width", detection.width}, {"height", detection.height},
        {"trackId", detection.track_id},
    };
}

std::vector<unsigned char> decode_mask(const MaskRle& mask) {
    if (mask.width <= 0 || mask.height <= 0) return {};
    const std::size_t count = static_cast<std::size_t>(mask.width) * static_cast<std::size_t>(mask.height);
    std::vector<unsigned char> pixels(count, 0);
    std::size_t cursor = 0;
    bool foreground = false;
    for (const std::uint32_t run : mask.counts) {
        const std::size_t end = std::min(count, cursor + static_cast<std::size_t>(run));
        if (foreground) std::fill(pixels.begin() + static_cast<std::ptrdiff_t>(cursor), pixels.begin() + static_cast<std::ptrdiff_t>(end), 255);
        cursor = end;
        foreground = !foreground;
        if (cursor >= count) break;
    }
    return pixels;
}

double seconds_between(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    if (start.time_since_epoch().count() == 0) return 0.0;
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

PipelineConfig PipelineConfig::from_json(const json& value) {
    PipelineConfig config;
    if (!value.is_object()) throw std::invalid_argument("pipeline configuration must be a JSON object");
    if (value.contains("models")) {
        const auto& models = value.at("models");
        config.yolo26n_model_id = models.value("yolo26n_model_id", config.yolo26n_model_id);
        config.segmentation_model_id = models.value("segmentation_model_id", config.segmentation_model_id);
        config.yolo26n_model_path = models.value("yolo26n_model_path", config.yolo26n_model_path);
        config.segmentation_model_path = models.value("segmentation_model_path", config.segmentation_model_path);
    }
    if (value.contains("detection")) {
        const auto& detection = value.at("detection");
        config.detection_confidence = detection.value("confidence_threshold", config.detection_confidence);
        config.detection_iou = detection.value("iou_threshold", config.detection_iou);
        config.naartjie_class_id = detection.value("naartjie_class_id", config.naartjie_class_id);
    }
    if (value.contains("bytetrack")) {
        const auto& tracker = value.at("bytetrack");
        config.bytetrack_high_threshold = tracker.value("high_threshold", config.bytetrack_high_threshold);
        config.bytetrack_low_threshold = tracker.value("low_threshold", config.bytetrack_low_threshold);
        config.bytetrack_match_iou = tracker.value("match_iou", config.bytetrack_match_iou);
        config.bytetrack_max_missed = tracker.value("max_missed_frames", config.bytetrack_max_missed);
    }
    if (value.contains("zones")) {
        const auto& zones = value.at("zones");
        config.acquisition_zone_start_x = coordinate_from_json(zones.value("acquisition_zone_start_x", json()), config.acquisition_zone_start_x);
        config.classification_line_x = coordinate_from_json(zones.value("classification_line_x", json()), config.classification_line_x);
        config.post_classification_zone = coordinate_from_json(zones.value("post_classification_zone", json()), config.post_classification_zone);
    }
    if (value.contains("capture")) {
        const auto& capture = value.at("capture");
        config.target_capture_percentages = capture.value("target_capture_percentages", config.target_capture_percentages);
        config.target_views = capture.value("target_views", config.target_views);
        config.minimum_capture_frame_gap = capture.value("minimum_frame_gap", config.minimum_capture_frame_gap);
        config.crop_margin = capture.value("crop_margin", config.crop_margin);
        config.sharpness_threshold = capture.value("sharpness_threshold", config.sharpness_threshold);
        config.minimum_crop_width = capture.value("minimum_crop_width", config.minimum_crop_width);
        config.minimum_crop_height = capture.value("minimum_crop_height", config.minimum_crop_height);
        config.maximum_permitted_overlap = capture.value("maximum_permitted_overlap", config.maximum_permitted_overlap);
    }
    if (value.contains("segmentation")) {
        const auto& segmentation = value.at("segmentation");
        config.segmentation_queue_size = segmentation.value("queue_size", config.segmentation_queue_size);
        config.segmentation_workers = segmentation.value("workers", config.segmentation_workers);
    }
    if (value.contains("classification")) {
        const auto& classification = value.at("classification");
        config.classification_delay_seconds = classification.value("stationary_delay_seconds", config.classification_delay_seconds);
        config.classification_timeout_seconds = classification.value("timeout_seconds", config.classification_timeout_seconds);
        config.minimum_views_for_classification = classification.value("minimum_views", config.minimum_views_for_classification);
        config.class_confidence_threshold = classification.value("confidence_threshold", config.class_confidence_threshold);
    }
    if (value.contains("retention")) {
        const auto& retention = value.at("retention");
        config.stale_track_frames = retention.value("stale_track_frames", config.stale_track_frames);
        config.track_retention_seconds = retention.value("release_after_seconds", config.track_retention_seconds);
    }
    if (value.contains("output")) {
        const auto& output = value.at("output");
        config.output_directory = output.value("results_directory", config.output_directory.string());
        config.logging_directory = output.value("logging_directory", config.logging_directory.string());
        config.debug_save_images = output.value("debug_save_images", config.debug_save_images);
    }
    if (value.contains("windows")) {
        const auto& windows = value.at("windows");
        config.windows_cpu_threads = windows.value("cpu_threads", config.windows_cpu_threads);
        config.live_max_fps = windows.value("live_max_fps", config.live_max_fps);
    }
    config.validate();
    return config;
}

json PipelineConfig::to_json() const {
    return {
        {"models", {
            {"yolo26n_model_id", yolo26n_model_id}, {"segmentation_model_id", segmentation_model_id},
            {"yolo26n_model_path", yolo26n_model_path}, {"segmentation_model_path", segmentation_model_path},
        }},
        {"detection", {{"confidence_threshold", detection_confidence}, {"iou_threshold", detection_iou}, {"naartjie_class_id", naartjie_class_id}}},
        {"bytetrack", {
            {"high_threshold", bytetrack_high_threshold}, {"low_threshold", bytetrack_low_threshold},
            {"match_iou", bytetrack_match_iou}, {"max_missed_frames", bytetrack_max_missed},
        }},
        {"zones", {
            {"acquisition_zone_start_x", coordinate_to_json(acquisition_zone_start_x)},
            {"classification_line_x", coordinate_to_json(classification_line_x)},
            {"post_classification_zone", coordinate_to_json(post_classification_zone)},
        }},
        {"capture", {
            {"target_capture_percentages", target_capture_percentages}, {"target_views", target_views},
            {"minimum_frame_gap", minimum_capture_frame_gap},
            {"crop_margin", crop_margin}, {"sharpness_threshold", sharpness_threshold},
            {"minimum_crop_width", minimum_crop_width}, {"minimum_crop_height", minimum_crop_height},
            {"maximum_permitted_overlap", maximum_permitted_overlap},
        }},
        {"segmentation", {{"queue_size", segmentation_queue_size}, {"workers", segmentation_workers}}},
        {"classification", {
            {"stationary_delay_seconds", classification_delay_seconds}, {"timeout_seconds", classification_timeout_seconds},
            {"minimum_views", minimum_views_for_classification}, {"confidence_threshold", class_confidence_threshold},
        }},
        {"retention", {{"stale_track_frames", stale_track_frames}, {"release_after_seconds", track_retention_seconds}}},
        {"output", {
            {"results_directory", output_directory.string()}, {"logging_directory", logging_directory.string()},
            {"debug_save_images", debug_save_images},
        }},
        {"windows", {{"cpu_threads", windows_cpu_threads}, {"live_max_fps", live_max_fps}}},
    };
}

void PipelineConfig::validate() const {
    if (!(detection_confidence >= 0.0F && detection_confidence <= 1.0F && detection_iou >= 0.0F && detection_iou <= 1.0F)) {
        throw std::invalid_argument("detection confidence and IoU thresholds must be between 0 and 1");
    }
    if (naartjie_class_id < 0) throw std::invalid_argument("naartjie detection class ID cannot be negative");
    if (target_views < 1 || static_cast<std::size_t>(target_views) != target_capture_percentages.size()) {
        throw std::invalid_argument("target_views must match target_capture_percentages");
    }
    if (minimum_views_for_classification < 1 || minimum_views_for_classification > target_views) {
        throw std::invalid_argument("minimum classification views must be between 1 and target_views");
    }
    if (!std::is_sorted(target_capture_percentages.begin(), target_capture_percentages.end()) ||
        std::any_of(target_capture_percentages.begin(), target_capture_percentages.end(), [](double value) { return value < 0.0 || value > 1.0; })) {
        throw std::invalid_argument("capture percentages must be sorted values between 0 and 1");
    }
    if (minimum_capture_frame_gap < 1 || crop_margin < 0.0 || sharpness_threshold < 0.0 || minimum_crop_width < 1 || minimum_crop_height < 1) {
        throw std::invalid_argument("crop quality configuration is invalid");
    }
    if (maximum_permitted_overlap < 0.0 || maximum_permitted_overlap > 1.0 || segmentation_queue_size < 1 || segmentation_workers < 1) {
        throw std::invalid_argument("overlap and segmentation queue configuration is invalid");
    }
    if (classification_delay_seconds < 0.0 || classification_timeout_seconds <= 0.0 ||
        class_confidence_threshold < 0.0F || class_confidence_threshold > 1.0F) {
        throw std::invalid_argument("classification configuration is invalid");
    }
    if (stale_track_frames < 1 || track_retention_seconds < 0.0 || windows_cpu_threads < 1 || windows_cpu_threads > 16 || live_max_fps < 1 || live_max_fps > 120) {
        throw std::invalid_argument("retention and Windows performance configuration is invalid");
    }
    const auto valid_coordinate = [](const Coordinate& coordinate) {
        return std::isfinite(coordinate.value) && coordinate.value >= 0.0 && (!coordinate.normalized || coordinate.value <= 1.0);
    };
    if (!valid_coordinate(acquisition_zone_start_x) || !valid_coordinate(classification_line_x) || !valid_coordinate(post_classification_zone)) {
        throw std::invalid_argument("zone coordinates must be non-negative and normalized coordinates cannot exceed 1");
    }
    if (acquisition_zone_start_x.normalized == classification_line_x.normalized &&
        classification_line_x.normalized == post_classification_zone.normalized &&
        !(acquisition_zone_start_x.value < classification_line_x.value && classification_line_x.value < post_classification_zone.value)) {
        throw std::invalid_argument("zone coordinates must be ordered acquisition, classification, post-classification");
    }
}

CropBounds clamp_crop(const Detection& box, double margin, int image_width, int image_height) {
    if (image_width <= 0 || image_height <= 0) return {};
    const double x_margin = box.width * margin;
    const double y_margin = box.height * margin;
    const int left = std::clamp(static_cast<int>(std::floor(box.x - x_margin)), 0, image_width);
    const int top = std::clamp(static_cast<int>(std::floor(box.y - y_margin)), 0, image_height);
    const int right = std::clamp(static_cast<int>(std::ceil(box.x + box.width + x_margin)), 0, image_width);
    const int bottom = std::clamp(static_cast<int>(std::ceil(box.y + box.height + y_margin)), 0, image_height);
    return {left, top, std::max(0, right - left), std::max(0, bottom - top)};
}

double acquisition_progress(double center_x, double acquisition_start_x, double classification_line_x) {
    if (classification_line_x <= acquisition_start_x) return center_x >= classification_line_x ? 1.0 : 0.0;
    return std::clamp((center_x - acquisition_start_x) / (classification_line_x - acquisition_start_x), 0.0, 1.0);
}

bool crossed_classification_line(double previous_center_x, double center_x, double classification_line_x) {
    return previous_center_x >= 0.0 && previous_center_x < classification_line_x && center_x >= classification_line_x;
}

double detection_overlap(const Detection& left, const Detection& right) {
    const double x1 = std::max(left.x, right.x);
    const double y1 = std::max(left.y, right.y);
    const double x2 = std::min(left.x + left.width, right.x + right.width);
    const double y2 = std::min(left.y + left.height, right.y + right.height);
    const double intersection = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double smaller = std::min(static_cast<double>(left.width) * left.height, static_cast<double>(right.width) * right.height);
    return smaller <= 0.0 ? 0.0 : intersection / smaller;
}

std::optional<std::size_t> next_capture_target(
    double progress,
    const std::vector<double>& targets,
    const std::vector<bool>& captured_targets
) {
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const bool captured = index < captured_targets.size() && captured_targets[index];
        if (!captured && progress >= targets[index]) return index;
    }
    return std::nullopt;
}

json SegmentationResult::to_json() const {
    return {
        {"generation", generation}, {"trackId", track_id}, {"frameId", frame_id}, {"viewId", view_id},
        {"classProbabilities", class_probabilities}, {"predictedClass", predicted_class}, {"confidence", confidence},
        {"defectArea", defect_area}, {"processingMs", processing_ms}, {"success", success}, {"message", message},
    };
}

json FinalClassification::to_json() const {
    return {
        {"finalClass", final_class}, {"confidence", confidence}, {"viewsCaptured", views_captured},
        {"successfulResults", successful_results}, {"viewClasses", view_classes},
        {"totalVisibleDefectArea", total_visible_defect_area}, {"incompleteEvidence", incomplete_evidence},
        {"success", success}, {"failureReason", failure_reason},
    };
}

FinalClassification aggregate_track_results(
    const std::vector<SegmentationResult>& results,
    float confidence_threshold,
    int views_captured,
    bool incomplete_evidence
) {
    FinalClassification final;
    final.views_captured = views_captured;
    final.incomplete_evidence = incomplete_evidence;
    std::vector<const SegmentationResult*> accepted;
    for (const auto& result : results) {
        if (!result.success || result.predicted_class < 0 || result.predicted_class > 2 ||
            !std::isfinite(result.confidence) || result.confidence < confidence_threshold) continue;
        accepted.push_back(&result);
        final.view_classes.push_back(result.predicted_class);
        final.total_visible_defect_area += result.defect_area;
    }
    final.successful_results = static_cast<int>(accepted.size());
    if (accepted.empty()) {
        final.failure_reason = "no confident segmentation results";
        return final;
    }
    // Class indices are canonical zero-based Class 1..3 values. The highest
    // confidently observed index is the conservative (worst-quality) result.
    const auto worst = *std::max_element(accepted.begin(), accepted.end(), [](const auto* left, const auto* right) {
        return left->predicted_class < right->predicted_class;
    });
    final.final_class = worst->predicted_class;
    float confidence_sum = 0.0F;
    int matching = 0;
    for (const auto* result : accepted) {
        if (result->predicted_class == final.final_class) {
            confidence_sum += result->confidence;
            ++matching;
        }
    }
    final.confidence = matching == 0 ? 0.0F : confidence_sum / static_cast<float>(matching);
    final.success = true;
    return final;
}

json TrackRecord::to_json(bool include_images) const {
    json views = json::array();
    for (const auto& view : captured_views) {
        json value = {
            {"trackId", view.track_id}, {"frameId", view.frame_id}, {"viewId", view.view_id},
            {"timestamp", view.timestamp}, {"zoneProgress", view.zone_progress},
            {"originalBoundingBox", detection_json(view.original_bounding_box)}, {"sharpness", view.sharpness},
            {"detectionConfidence", view.detection_confidence}, {"segmentationStatus", view.segmentation_status},
        };
        if (include_images) value["encodedCropBytes"] = view.crop.encoded.size();
        views.push_back(std::move(value));
    }
    json results = json::array();
    for (const auto& result : segmentation_results) results.push_back(result.to_json());
    return {
        {"trackId", track_id}, {"currentBoundingBox", detection_json(current_bounding_box)},
        {"detectionConfidence", detection_confidence}, {"firstSeenFrame", first_seen_frame}, {"lastSeenFrame", last_seen_frame},
        {"firstSeenTimestamp", first_seen_timestamp}, {"lastSeenTimestamp", last_seen_timestamp},
        {"state", track_state_name(state)}, {"acquisitionZoneProgress", acquisition_zone_progress},
        {"capturedViews", views}, {"segmentationResults", results},
        {"finalClass", final_class < 0 ? json(nullptr) : json("Class " + std::to_string(final_class + 1))},
        {"finalClassId", final_class < 0 ? json(nullptr) : json(final_class)}, {"finalConfidence", final_confidence},
        {"incompleteEvidence", incomplete_evidence}, {"classificationTimestamp", classification_timestamp},
        {"failureReason", state == TrackState::Failed && !messages.empty() ? json(messages.back()) : json(nullptr)},
        {"messages", messages}, {"viewsCaptured", captured_views.size()}, {"viewsUsed", views_used},
        {"segmentationCompleted", segmentation_completed}, {"totalVisibleDefectArea", total_visible_defect_area},
        {"detectionLatencyMs", detection_latency_ms}, {"segmentationLatencyMs", segmentation_latency_ms},
        {"classificationLatencyMs", classification_latency_ms},
    };
}

MultiViewPipeline::MultiViewPipeline(PipelineConfig config, Detector detector, Segmenter segmenter, Cropper cropper)
    : config_(std::move(config)), detector_(std::move(detector)), segmenter_(std::move(segmenter)), cropper_(std::move(cropper)),
      tracker_(std::make_unique<ByteTracker>(config_.bytetrack_high_threshold, config_.bytetrack_low_threshold,
          config_.bytetrack_match_iou, config_.bytetrack_max_missed)),
      segmentation_queue_(config_.segmentation_queue_size) {
    config_.validate();
}

MultiViewPipeline::~MultiViewPipeline() { stop(); }

void MultiViewPipeline::start() {
    if (running_.exchange(true)) return;
    for (int index = 0; index < config_.segmentation_workers; ++index) {
        segmentation_threads_.emplace_back([this] { segmentation_loop(); });
    }
}

json MultiViewPipeline::process_frame(const LiveFrame& frame) {
    if (!running_) throw std::runtime_error("multi-view pipeline is not running");
    if (frame.frame_id == 0 || frame.frame_id <= latest_frame_id_) throw std::invalid_argument("live frame IDs must be strictly monotonic");
    if (frame.width <= 0 || frame.height <= 0) throw std::invalid_argument("live frame dimensions are invalid");
    const double acquisition_x = config_.acquisition_zone_start_x.pixels(frame.width);
    const double classification_x = config_.classification_line_x.pixels(frame.width);
    const double post_x = config_.post_classification_zone.pixels(frame.width);
    if (!(acquisition_x < classification_x && classification_x < post_x)) {
        throw std::invalid_argument("resolved live zones must be ordered acquisition, classification, post-classification");
    }
    ++detector_waiting_;
    InferenceResult inference;
    {
        std::scoped_lock model_lock(model_mutex_);
        inference = detector_(frame);
    }
    --detector_waiting_;
    model_condition_.notify_all();
    const auto tracked = tracker_->update(frame.frame_id, inference.detections);
    const auto now = std::chrono::steady_clock::now();
    if (last_frame_at_.time_since_epoch().count() != 0) {
        const double elapsed = seconds_between(last_frame_at_, now);
        if (elapsed > 0.0) {
            const double instantaneous = 1.0 / elapsed;
            detector_fps_ = detector_fps_ == 0.0 ? instantaneous : detector_fps_ * 0.85 + instantaneous * 0.15;
        }
    }
    last_frame_at_ = now;
    latest_frame_id_ = frame.frame_id;
    for (const auto& detection : tracked) update_track(frame, detection, tracked, inference.timing.backend_total_ms);
    // update_track may stamp a classification-line crossing after the frame's
    // initial timing sample. Use a fresh time point so zero-delay policies do
    // not observe a negative elapsed duration and miss classification forever.
    const auto evaluation_now = std::chrono::steady_clock::now();
    evaluate_timeouts(evaluation_now);
    remove_stale_tracks(frame.frame_id, evaluation_now);
    return snapshot();
}

void MultiViewPipeline::update_track(
    const LiveFrame& frame,
    const Detection& detection,
    const std::vector<Detection>& detections,
    double detection_ms
) {
    if (detection.track_id < 0) return;
    const double acquisition_start = config_.acquisition_zone_start_x.pixels(frame.width);
    const double classification_line = config_.classification_line_x.pixels(frame.width);
    const double post_line = config_.post_classification_zone.pixels(frame.width);
    const double center_x = detection.x + detection.width * 0.5;
    std::optional<SegmentationJobV2> pending_job;
    {
        std::scoped_lock lock(tracks_mutex_);
        auto [iterator, inserted] = tracks_.try_emplace(detection.track_id);
        TrackRecord& track = iterator->second;
        if (inserted) {
            track.track_id = detection.track_id;
            track.first_seen_frame = frame.frame_id;
            track.first_seen_timestamp = frame.timestamp;
            track.first_seen_at = frame.captured_at;
            track.state = TrackState::New;
        }
        track.current_bounding_box = detection;
        track.detection_confidence = detection.confidence;
        track.last_seen_frame = frame.frame_id;
        track.last_seen_timestamp = frame.timestamp;
        track.detection_latency_ms = detection_ms;
        track.acquisition_zone_progress = acquisition_progress(center_x, acquisition_start, classification_line);
        const bool reached_line = center_x >= classification_line || crossed_classification_line(track.previous_center_x, center_x, classification_line);

        if (track.state == TrackState::New && center_x >= acquisition_start && center_x < classification_line) track.state = TrackState::Acquiring;
        if ((track.state == TrackState::New || track.state == TrackState::Acquiring) && center_x < classification_line) {
            std::vector<bool> captured(config_.target_capture_percentages.size(), false);
            for (const auto& view : track.captured_views) {
                if (view.view_id >= 0 && static_cast<std::size_t>(view.view_id) < captured.size()) captured[static_cast<std::size_t>(view.view_id)] = true;
            }
            const auto target = next_capture_target(track.acquisition_zone_progress, config_.target_capture_percentages, captured);
            const bool frame_gap_ok = track.captured_views.empty() ||
                frame.frame_id >= track.captured_views.back().frame_id + static_cast<std::uint64_t>(config_.minimum_capture_frame_gap);
            if (target && frame_gap_ok && detection.confidence >= config_.detection_confidence &&
                detection.width >= config_.minimum_crop_width && detection.height >= config_.minimum_crop_height) {
                bool overlap_ok = true;
                for (const auto& other : detections) {
                    if (other.track_id != detection.track_id && detection_overlap(detection, other) > config_.maximum_permitted_overlap) {
                        overlap_ok = false;
                        break;
                    }
                }
                if (overlap_ok) {
                    const CropBounds bounds = clamp_crop(detection, config_.crop_margin, frame.width, frame.height);
                    CropImage crop = cropper_(frame, bounds);
                    if (!crop.encoded.empty() && crop.width >= config_.minimum_crop_width && crop.height >= config_.minimum_crop_height &&
                        crop.sharpness >= config_.sharpness_threshold) {
                        CapturedView view;
                        view.track_id = track.track_id;
                        view.frame_id = frame.frame_id;
                        view.view_id = static_cast<int>(*target);
                        view.timestamp = frame.timestamp;
                        view.zone_progress = track.acquisition_zone_progress;
                        view.original_bounding_box = detection;
                        view.sharpness = crop.sharpness;
                        view.detection_confidence = detection.confidence;
                        view.crop = std::move(crop);
                        pending_job = SegmentationJobV2{generation_.load(), track.track_id, frame.frame_id, view.view_id, frame.timestamp, view.crop.encoded};
                        track.captured_views.push_back(std::move(view));
                    } else if (crop.sharpness < config_.sharpness_threshold) {
                        track.messages.push_back("view rejected: sharpness below threshold");
                    }
                } else {
                    track.messages.push_back("view rejected: overlapping fruit");
                }
            }
            if (track.captured_views.size() >= static_cast<std::size_t>(config_.target_views) && center_x < classification_line) {
                track.state = TrackState::WaitingForSegmentation;
            }
        }
        if (reached_line && track.state != TrackState::Classified && track.state != TrackState::Exited && track.state != TrackState::Failed) {
            track.state = TrackState::WaitingForClassification;
            if (track.line_reached_at.time_since_epoch().count() == 0) track.line_reached_at = std::chrono::steady_clock::now();
        }
        if (center_x >= post_line && track.state == TrackState::Classified) {
            track.state = TrackState::Exited;
            track.terminal_at = std::chrono::steady_clock::now();
        }
        track.previous_center_x = center_x;
    }
    if (pending_job && !segmentation_queue_.try_push(std::move(*pending_job))) {
        std::scoped_lock lock(tracks_mutex_);
        auto found = tracks_.find(detection.track_id);
        if (found != tracks_.end()) {
            auto& views = found->second.captured_views;
            if (!views.empty() && views.back().frame_id == frame.frame_id) views.back().segmentation_status = "dropped";
            found->second.messages.push_back("segmentation queue overflow: view dropped");
        }
    }
}

void MultiViewPipeline::segmentation_loop() {
    SegmentationJobV2 job;
    while (segmentation_queue_.pop(job)) {
        SegmentationResult result;
        const auto started = std::chrono::steady_clock::now();
        try {
            std::unique_lock model_lock(model_mutex_);
            model_condition_.wait(model_lock, [&] { return !running_ || detector_waiting_.load() == 0; });
            if (!running_) break;
            result = segmenter_(job);
        } catch (const std::exception& error) {
            result.success = false;
            result.message = error.what();
        }
        result.track_id = job.track_id;
        result.frame_id = job.frame_id;
        result.view_id = job.view_id;
        result.generation = job.generation;
        if (result.processing_ms <= 0.0) {
            result.processing_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        }
        route_segmentation_result(result);
    }
}

bool MultiViewPipeline::route_segmentation_result(const SegmentationResult& result) {
    if (result.generation != 0 && result.generation != generation_.load()) return false;
    std::scoped_lock lock(tracks_mutex_);
    auto found = tracks_.find(result.track_id);
    if (found == tracks_.end()) return false;
    TrackRecord& track = found->second;
    const auto view = std::find_if(track.captured_views.begin(), track.captured_views.end(), [&](const CapturedView& candidate) {
        return candidate.view_id == result.view_id && candidate.frame_id == result.frame_id && candidate.track_id == result.track_id;
    });
    if (view == track.captured_views.end()) return false;
    const auto duplicate = std::find_if(track.segmentation_results.begin(), track.segmentation_results.end(), [&](const SegmentationResult& candidate) {
        return candidate.view_id == result.view_id && candidate.frame_id == result.frame_id;
    });
    if (duplicate != track.segmentation_results.end()) return false;
    view->segmentation_status = result.success ? "completed" : "failed";
    track.segmentation_results.push_back(result);
    track.segmentation_completed = static_cast<int>(track.segmentation_results.size());
    const double latency_sum = std::accumulate(track.segmentation_results.begin(), track.segmentation_results.end(), 0.0,
        [](double total, const SegmentationResult& item) { return total + item.processing_ms; });
    track.segmentation_latency_ms = latency_sum / static_cast<double>(track.segmentation_results.size());
    last_segmentation_latency_ms_ = result.processing_ms;
    if (!result.success) track.messages.push_back("segmentation failed for view " + std::to_string(result.view_id + 1) + ": " + result.message);
    maybe_classify_locked(track, std::chrono::steady_clock::now());
    return true;
}

void MultiViewPipeline::evaluate_timeouts(std::chrono::steady_clock::time_point now) {
    std::scoped_lock lock(tracks_mutex_);
    for (auto& entry : tracks_) maybe_classify_locked(entry.second, now);
}

void MultiViewPipeline::maybe_classify_locked(TrackRecord& track, std::chrono::steady_clock::time_point now) {
    if (track.state != TrackState::WaitingForClassification || track.line_reached_at.time_since_epoch().count() == 0) return;
    const double elapsed = seconds_between(track.line_reached_at, now);
    if (elapsed < config_.classification_delay_seconds) return;
    const bool timed_out = elapsed >= config_.classification_timeout_seconds;
    const bool all_target_results = track.captured_views.size() >= static_cast<std::size_t>(config_.target_views) &&
        track.segmentation_results.size() >= track.captured_views.size();
    if (!timed_out && !all_target_results) return;
    FinalClassification final = aggregate_track_results(
        track.segmentation_results,
        config_.class_confidence_threshold,
        static_cast<int>(track.captured_views.size()),
        timed_out && track.captured_views.size() < static_cast<std::size_t>(config_.target_views)
    );
    if (final.successful_results < config_.minimum_views_for_classification) {
        if (!timed_out) return;
        mark_failed_locked(track, "insufficient evidence: " + std::to_string(final.successful_results) +
            " confident view(s), " + std::to_string(config_.minimum_views_for_classification) + " required", now);
        track.incomplete_evidence = true;
        return;
    }
    track.final_class = final.final_class;
    track.final_confidence = final.confidence;
    track.views_used = final.successful_results;
    track.total_visible_defect_area = final.total_visible_defect_area;
    track.incomplete_evidence = final.incomplete_evidence || final.successful_results < config_.target_views;
    track.classification_timestamp = utc_now_iso8601();
    track.classification_latency_ms = seconds_between(track.first_seen_at, now) * 1000.0;
    last_classification_latency_ms_ = track.classification_latency_ms;
    track.state = TrackState::Classified;
    track.terminal_at = now;
    save_result_locked(track);
}

void MultiViewPipeline::mark_failed_locked(TrackRecord& track, const std::string& reason, std::chrono::steady_clock::time_point now) {
    if (track.state == TrackState::Failed || track.state == TrackState::Exited) return;
    track.state = TrackState::Failed;
    track.messages.push_back(reason);
    track.classification_timestamp = utc_now_iso8601();
    track.classification_latency_ms = seconds_between(track.first_seen_at, now) * 1000.0;
    track.terminal_at = now;
    save_result_locked(track);
}

void MultiViewPipeline::save_result_locked(TrackRecord& track) {
    if (track.result_saved) return;
    try {
        std::filesystem::create_directories(config_.output_directory);
        std::filesystem::create_directories(config_.logging_directory);
        const json record = track.to_json(false);
        const auto result_path = config_.output_directory / ("track_" + std::to_string(track.track_id) + ".json");
        std::ofstream result_output(result_path);
        result_output << std::setw(2) << record << '\n';
        std::ofstream log_output(config_.logging_directory / "tracks.jsonl", std::ios::app);
        log_output << record.dump() << '\n';
        if (config_.debug_save_images) {
            const auto debug_directory = config_.output_directory / ("track_" + std::to_string(track.track_id));
            std::filesystem::create_directories(debug_directory);
            for (const auto& view : track.captured_views) {
                const int number = view.view_id + 1;
                std::ofstream image(debug_directory / ("view_" + std::to_string(number) + ".jpg"), std::ios::binary);
                image.write(reinterpret_cast<const char*>(view.crop.encoded.data()), static_cast<std::streamsize>(view.crop.encoded.size()));
                const auto result = std::find_if(track.segmentation_results.begin(), track.segmentation_results.end(), [&](const SegmentationResult& candidate) {
                    return candidate.view_id == view.view_id && candidate.frame_id == view.frame_id;
                });
                if (result != track.segmentation_results.end() && !result->defect_mask.counts.empty()) {
                    const auto pixels = decode_mask(result->defect_mask);
                    stbi_write_png((debug_directory / ("mask_" + std::to_string(number) + ".png")).string().c_str(),
                        result->defect_mask.width, result->defect_mask.height, 1, pixels.data(), result->defect_mask.width);
                }
            }
            std::ofstream debug_result(debug_directory / "result.json");
            debug_result << std::setw(2) << record << '\n';
        }
        track.result_saved = true;
    } catch (const std::exception& error) {
        track.messages.push_back(std::string("could not save result: ") + error.what());
    }
}

std::size_t MultiViewPipeline::remove_stale_tracks(std::uint64_t current_frame, std::chrono::steady_clock::time_point now) {
    std::scoped_lock lock(tracks_mutex_);
    std::size_t removed = 0;
    for (auto iterator = tracks_.begin(); iterator != tracks_.end();) {
        TrackRecord& track = iterator->second;
        const bool stale = current_frame > track.last_seen_frame && current_frame - track.last_seen_frame > static_cast<std::uint64_t>(config_.stale_track_frames);
        if (stale && track.state != TrackState::Classified && track.state != TrackState::Exited && track.state != TrackState::Failed) {
            mark_failed_locked(track, "track lost before classification completed", now);
        }
        const bool terminal = track.state == TrackState::Classified || track.state == TrackState::Exited || track.state == TrackState::Failed;
        if (terminal && seconds_between(track.terminal_at, now) >= config_.track_retention_seconds) {
            iterator = tracks_.erase(iterator);
            ++removed;
        } else {
            ++iterator;
        }
    }
    return removed;
}

json MultiViewPipeline::snapshot_locked() const {
    json records = json::array();
    std::vector<int> ids;
    ids.reserve(tracks_.size());
    for (const auto& entry : tracks_) ids.push_back(entry.first);
    std::sort(ids.begin(), ids.end());
    for (const int id : ids) records.push_back(tracks_.at(id).to_json(false));
    return {
        {"frameId", latest_frame_id_}, {"tracks", records}, {"queueSize", segmentation_queue_.size()},
        {"droppedSegmentationJobs", segmentation_queue_.dropped()}, {"detectorFps", detector_fps_},
        {"segmentationLatencyMs", last_segmentation_latency_ms_},
        {"classificationLatencyMs", last_classification_latency_ms_}, {"config", config_.to_json()},
    };
}

json MultiViewPipeline::snapshot() const {
    std::scoped_lock lock(tracks_mutex_);
    return snapshot_locked();
}

std::optional<TrackRecord> MultiViewPipeline::track(int track_id) const {
    std::scoped_lock lock(tracks_mutex_);
    const auto found = tracks_.find(track_id);
    if (found == tracks_.end()) return std::nullopt;
    return found->second;
}

void MultiViewPipeline::reset() {
    std::scoped_lock lock(tracks_mutex_);
    ++generation_;
    tracks_.clear();
    tracker_ = std::make_unique<ByteTracker>(config_.bytetrack_high_threshold, config_.bytetrack_low_threshold,
        config_.bytetrack_match_iou, config_.bytetrack_max_missed);
    latest_frame_id_ = 0;
    detector_fps_ = 0.0;
    last_segmentation_latency_ms_ = 0.0;
    last_classification_latency_ms_ = 0.0;
    last_frame_at_ = {};
}

void MultiViewPipeline::stop() {
    if (!running_.exchange(false)) return;
    segmentation_queue_.stop();
    model_condition_.notify_all();
    for (auto& worker : segmentation_threads_) if (worker.joinable()) worker.join();
    segmentation_threads_.clear();
}

}  // namespace runner
