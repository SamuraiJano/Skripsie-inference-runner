#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <filesystem>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runner/Types.hpp"

namespace runner {

template <typename T>
class BoundedLatestQueue {
public:
    explicit BoundedLatestQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) throw std::invalid_argument("queue capacity must be positive");
    }

    void push(T value) {
        std::scoped_lock lock(mutex_);
        if (stopped_) return;
        while (values_.size() >= capacity_) {
            values_.pop_front();
            ++dropped_;
        }
        values_.push_back(std::move(value));
        condition_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return stopped_ || !values_.empty(); });
        if (values_.empty()) return false;
        value = std::move(values_.back());
        if (values_.size() > 1) dropped_ += values_.size() - 1;
        values_.clear();
        return true;
    }

    void stop() {
        std::scoped_lock lock(mutex_);
        stopped_ = true;
        values_.clear();
        condition_.notify_all();
    }

    std::uint64_t dropped() const {
        std::scoped_lock lock(mutex_);
        return dropped_;
    }

    bool has_pending() const {
        std::scoped_lock lock(mutex_);
        return !values_.empty();
    }

private:
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<T> values_;
    bool stopped_ = false;
    std::uint64_t dropped_ = 0;
};

struct Track {
    int id = -1;
    Detection detection;
    Detection last_observation;
    std::uint64_t last_frame_id = 0;
    int hits = 0;
    int missed = 0;
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    bool has_velocity = false;
};

class ByteTracker {
public:
    ByteTracker(
        float high_threshold = 0.5F,
        float low_threshold = 0.1F,
        float match_iou = 0.3F,
        int max_missed = 30,
        float new_track_threshold = -1.0F
    );
    std::vector<Detection> update(
        std::uint64_t frame_id,
        const std::vector<Detection>& detections,
        const std::vector<bool>& new_track_eligible = {},
        bool include_predictions = false
    );
    const std::vector<Track>& tracks() const { return tracks_; }
    bool has_track(int track_id) const;
    bool observed_in_frame(int track_id, std::uint64_t frame_id) const;

private:
    float high_threshold_;
    float low_threshold_;
    float new_track_threshold_;
    float match_iou_;
    int max_missed_;
    int next_id_ = 1;
    std::vector<Track> tracks_;
};

struct GradeObservation {
    int track_id = -1;
    int class_id = -1;
    float confidence = 0.0F;
    std::uint64_t frame_id = 0;
};

struct GradeDecision {
    int track_id = -1;
    std::string state = "uncertain/insufficient_evidence";
    int class_id = -1;
    int observations = 0;
    float winning_weight = 0.0F;
};

class EvidenceAggregator {
public:
    EvidenceAggregator(int minimum_observations = 3, float minimum_margin = 0.15F);
    GradeDecision add(const GradeObservation& observation);
    GradeDecision decision(int track_id) const;

private:
    struct Evidence { int count = 0; std::unordered_map<int, float> weights; };
    int minimum_observations_;
    float minimum_margin_;
    std::unordered_map<int, Evidence> evidence_;
};

struct PipelineFrame {
    std::uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point captured_at;
    std::vector<unsigned char> encoded_image;
};

struct SegmentationJob {
    std::uint64_t frame_id = 0;
    int track_id = -1;
    std::vector<unsigned char> encoded_crop;
};

enum class PipelineMode {
    SerializedAsync,
    DetectorPriorityTimeSliced,
};

class AsyncFruitPipeline {
public:
    using Detector = std::function<std::vector<Detection>(const PipelineFrame&)>;
    using Admission = std::function<std::optional<SegmentationJob>(const PipelineFrame&, const std::vector<Detection>&)>;
    using Segmenter = std::function<std::vector<GradeObservation>(const SegmentationJob&)>;
    using Output = std::function<void(const PipelineFrame&, const std::vector<Detection>&, const std::vector<GradeDecision>&)>;

    AsyncFruitPipeline(
        Detector detector,
        Admission admission,
        Segmenter segmenter,
        Output output,
        std::uint64_t maximum_result_lag = 2,
        PipelineMode mode = PipelineMode::DetectorPriorityTimeSliced
    );
    ~AsyncFruitPipeline();
    void start();
    void submit(PipelineFrame frame);
    void stop();
    std::uint64_t dropped_capture_frames() const { return capture_queue_.dropped(); }
    std::uint64_t dropped_segmentation_jobs() const { return segmentation_queue_.dropped(); }

private:
    void detector_loop();
    void segmentation_loop();

    Detector detector_;
    Admission admission_;
    Segmenter segmenter_;
    Output output_;
    std::uint64_t maximum_result_lag_;
    PipelineMode mode_;
    BoundedLatestQueue<PipelineFrame> capture_queue_{1};
    BoundedLatestQueue<SegmentationJob> segmentation_queue_{1};
    ByteTracker tracker_;
    EvidenceAggregator aggregator_;
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> latest_frame_id_{0};
    std::thread detector_thread_;
    std::thread segmentation_thread_;
    // Keep the shared model command stream ordered. In detector-priority mode,
    // segmentation yields while a newer capture is queued.
    std::mutex inference_mutex_;
    std::mutex latest_mutex_;
    PipelineFrame latest_frame_;
    std::vector<Detection> latest_detections_;
};

// The live conveyor pipeline is intentionally independent from image decoding and
// model runtimes. The HTTP/camera adapter supplies RGB frames and callbacks, which
// keeps all state transitions and routing deterministic and host-testable.
enum class TrackState {
    Tentative,
    Confirmed,
    Acquiring,
    WaitingForSegmentation,
    WaitingForClassification,
    Lost,
    Completed,
    Rejected,
};

std::string track_state_name(TrackState state);

struct Coordinate {
    double value = 0.0;
    bool normalized = true;

    double pixels(int extent) const;
};

struct PipelineConfig {
    std::string yolo26n_model_id;
    std::string segmentation_model_id;
    std::string yolo26n_model_path;
    std::string segmentation_model_path;
    float detection_confidence = 0.25F;
    float detection_iou = 0.50F;
    int naartjie_class_id = 0;
    float bytetrack_high_threshold = 0.50F;
    float bytetrack_low_threshold = 0.10F;
    float bytetrack_new_track_threshold = 0.65F;
    float bytetrack_match_iou = 0.30F;
    int bytetrack_max_missed = 30;
    // Normalized vertical conveyor geometry. Entry/exit heights are measured
    // inward from the top/bottom ROI edges.
    Coordinate roi_left{0.15, true};
    Coordinate roi_right{0.85, true};
    Coordinate roi_top{0.05, true};
    Coordinate roi_bottom{0.95, true};
    Coordinate entry_zone_height{0.12, true};
    Coordinate exit_zone_height{0.12, true};
    // New-ID admission is intentionally stricter than maintenance association.
    // Ratio thresholds are relative to the complete input frame.
    bool strict_entry_gating = false;
    double roi_boundary_tolerance_ratio = 0.02;
    double minimum_bbox_width_ratio = 0.025;
    double minimum_bbox_height_ratio = 0.025;
    double minimum_bbox_area_ratio = 0.003;
    double maximum_bbox_area_ratio = 0.25;
    double maintenance_size_tolerance = 0.50;
    // Tentative candidates need repeated high-confidence observations plus
    // net downward motion before acquisition is enabled.
    int minimum_confirmation_hits = 3;
    int motion_window_frames = 6;
    double minimum_downward_displacement_ratio = 0.003;
    double maximum_upward_displacement_ratio = 0.01;
    double stationary_tolerance_ratio = 0.002;
    int maximum_stationary_frames = 12;
    std::vector<double> target_capture_percentages{0.20, 0.40, 0.60, 0.80};
    int target_views = 4;
    int minimum_capture_frame_gap = 2;
    int minimum_views_for_classification = 2;
    double crop_margin = 0.0;
    double sharpness_threshold = 60.0;
    int minimum_crop_width = 32;
    int minimum_crop_height = 32;
    double maximum_permitted_overlap = 0.25;
    std::size_t segmentation_queue_size = 16;
    int segmentation_workers = 2;
    double classification_delay_seconds = 0.10;
    double classification_timeout_seconds = 2.0;
    float class_confidence_threshold = 0.55F;
    int stale_track_frames = 30;
    double track_retention_seconds = 5.0;
    std::filesystem::path output_directory = "data/results";
    std::filesystem::path logging_directory = "data/logs";
    bool debug_save_images = false;
    bool tracking_diagnostics = false;
    int windows_cpu_threads = 8;
    int live_max_fps = 30;

    // Dense, bounded candidate collection replaces the legacy assumption that
    // four route percentages constitute a complete inspection.  The legacy
    // capture settings remain available only when surface mapping is disabled.
    bool surface_mapping_enabled = false;
    std::string surface_shape_model = "oblate";
    bool surface_robust_shape_fit = false;
    std::string surface_rotation_primary_axis = "y";
    int surface_rotation_direction = 1;
    int surface_candidate_frame_stride = 2;
    int surface_maximum_candidate_views = 30;
    int surface_maximum_mapping_views = 20;
    int surface_minimum_mapping_views = 8;
    double surface_preferred_step_degrees = 24.0;
    double surface_maximum_step_degrees = 40.0;
    double surface_rotation_search_step_degrees = 1.0;
    double surface_secondary_axis_limit_degrees = 10.0;
    int surface_downsample_factor = 4;
    double surface_texture_percentile = 97.0;
    int surface_maximum_texture_points = 80;
    int surface_minimum_texture_points = 20;
    double surface_minimum_valid_point_fraction = 0.45;
    double surface_latitude_limit_degrees = 60.0;
    double surface_pre_roll_degrees = 45.0;
    double surface_post_roll_degrees = 45.0;
    double surface_target_coverage_degrees = 350.0;
    double surface_minimum_confident_coverage = 300.0;
    int surface_maximum_map_width = 1600;
    int surface_moment_coordinate_stride = 4;
    double surface_spherical_tolerance = 0.06;
    double surface_minimum_detection_confidence = 0.25;
    double surface_minimum_focus_score = 35.0;
    double surface_minimum_mask_area_ratio = 0.35;
    double surface_maximum_mask_area_ratio = 0.92;
    double surface_maximum_secondary_component_ratio = 0.12;
    int surface_silhouette_border_pixels = 3;
    int surface_morphological_closing_pixels = 5;
    double surface_maximum_center_shift_ratio = 0.22;
    double surface_maximum_scale_change_ratio = 0.30;
    double surface_minimum_observed_area_ratio = 0.55;
    double surface_alignment_confidence_threshold = 0.18;
    double surface_minimum_reliable_pair_ratio = 0.65;
    double surface_defect_probability_threshold = 0.50;

    static PipelineConfig from_json(const json& value);
    json to_json() const;
    void validate() const;
};

struct CropBounds {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

CropBounds clamp_crop(const Detection& box, double margin, int image_width, int image_height);
double acquisition_progress(double center_x, double acquisition_start_x, double classification_line_x);
bool crossed_classification_line(double previous_center_x, double center_x, double classification_line_x);
double detection_overlap(const Detection& left, const Detection& right);
std::optional<std::size_t> next_capture_target(
    double progress,
    const std::vector<double>& targets,
    const std::vector<bool>& captured_targets
);

struct CropImage {
    std::vector<unsigned char> encoded;
    int width = 0;
    int height = 0;
    double sharpness = 0.0;
};

struct DefectInstance {
    int class_id = -1;
    std::string label;
    float confidence = 0.0F;
    MaskRle mask;

    json to_json() const;
};

struct SegmentationResult {
    std::uint64_t generation = 0;
    int track_id = -1;
    std::uint64_t frame_id = 0;
    int view_id = -1;
    std::vector<float> class_probabilities;
    int predicted_class = -1;
    float confidence = 0.0F;
    MaskRle defect_mask;
    double defect_area = 0.0;
    double processing_ms = 0.0;
    bool success = false;
    std::string message;
    std::vector<DefectInstance> defect_instances;

    json to_json() const;
};

struct CapturedView {
    std::string capture_id;
    int track_id = -1;
    std::uint64_t frame_id = 0;
    int view_id = -1;
    std::string timestamp;
    double zone_progress = 0.0;
    Detection original_bounding_box;
    CropImage crop;
    double sharpness = 0.0;
    float detection_confidence = 0.0F;
    bool used_tracking_prediction = false;
    std::string segmentation_status = "queued";
    CropBounds crop_bounds;
};

struct FinalClassification {
    int final_class = -1;
    float confidence = 0.0F;
    int views_captured = 0;
    int successful_results = 0;
    std::vector<int> view_classes;
    double total_visible_defect_area = 0.0;
    bool incomplete_evidence = false;
    bool success = false;
    std::string failure_reason;

    json to_json() const;
};

FinalClassification aggregate_track_results(
    const std::vector<SegmentationResult>& results,
    float confidence_threshold,
    int views_captured,
    bool incomplete_evidence
);

struct TrackRecord {
    int track_id = -1;
    Detection current_bounding_box;
    float detection_confidence = 0.0F;
    std::uint64_t first_seen_frame = 0;
    std::uint64_t last_seen_frame = 0;
    std::string first_seen_timestamp;
    std::string last_seen_timestamp;
    TrackState state = TrackState::Tentative;
    TrackState state_before_lost = TrackState::Tentative;
    double acquisition_zone_progress = 0.0;
    std::vector<CapturedView> captured_views;
    std::vector<SegmentationResult> segmentation_results;
    int final_class = -1;
    float final_confidence = 0.0F;
    bool incomplete_evidence = false;
    std::string classification_timestamp;
    std::vector<std::string> messages;
    int views_used = 0;
    int segmentation_completed = 0;
    double total_visible_defect_area = 0.0;
    double detection_latency_ms = 0.0;
    double segmentation_latency_ms = 0.0;
    double classification_latency_ms = 0.0;
    double previous_center_y = -1.0;
    int confirmation_hits = 0;
    int stationary_frames = 0;
    bool confirmed = false;
    bool has_current_observation = true;
    bool classification_completed = false;
    bool counted = false;
    bool exited = false;
    std::string motion_status = "PENDING";
    std::string size_status = "VALID";
    std::string rejection_reason;
    std::deque<std::pair<std::uint64_t, double>> centroid_y_history;
    std::chrono::steady_clock::time_point first_seen_at;
    std::chrono::steady_clock::time_point line_reached_at;
    std::chrono::steady_clock::time_point terminal_at;
    bool result_saved = false;
    int candidate_views = 0;
    std::uint64_t last_candidate_frame = 0;
    std::vector<std::pair<std::uint64_t, std::string>> view_rejections;
    json surface_mapping = nullptr;
    bool surface_mapping_queued = false;

    json to_json(bool include_images = false) const;
};

struct LiveFrame {
    std::uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point captured_at;
    std::string timestamp;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> encoded_image;
    std::vector<unsigned char> rgb_image;
};

struct SegmentationJobV2 {
    std::uint64_t generation = 0;
    int track_id = -1;
    std::uint64_t frame_id = 0;
    int view_id = -1;
    std::string timestamp;
    std::vector<unsigned char> crop_image;
};

struct SurfaceMappingJob {
    std::uint64_t generation = 0;
    int track_id = -1;
    json request;
    FinalClassification fallback;
};

template <typename T>
class BoundedFifoQueue {
public:
    explicit BoundedFifoQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) throw std::invalid_argument("queue capacity must be positive");
    }

    bool try_push(T value) {
        std::scoped_lock lock(mutex_);
        if (stopped_ || values_.size() >= capacity_) {
            ++dropped_;
            return false;
        }
        values_.push_back(std::move(value));
        condition_.notify_one();
        return true;
    }

    bool pop(T& value) {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return stopped_ || !values_.empty(); });
        if (values_.empty()) return false;
        value = std::move(values_.front());
        values_.pop_front();
        return true;
    }

    void stop() {
        std::scoped_lock lock(mutex_);
        stopped_ = true;
        condition_.notify_all();
    }

    std::size_t size() const {
        std::scoped_lock lock(mutex_);
        return values_.size();
    }

    std::uint64_t dropped() const {
        std::scoped_lock lock(mutex_);
        return dropped_;
    }

private:
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<T> values_;
    bool stopped_ = false;
    std::uint64_t dropped_ = 0;
};

class MultiViewPipeline {
public:
    using Detector = std::function<InferenceResult(const LiveFrame&)>;
    using Segmenter = std::function<SegmentationResult(const SegmentationJobV2&)>;
    using Cropper = std::function<CropImage(const LiveFrame&, const CropBounds&)>;
    using SurfaceMapper = std::function<json(const json&)>;

    MultiViewPipeline(PipelineConfig config, Detector detector, Segmenter segmenter, Cropper cropper, SurfaceMapper surface_mapper = {});
    ~MultiViewPipeline();
    void start();
    json process_frame(const LiveFrame& frame);
    json snapshot() const;
    json capture_library(std::size_t limit = 500) const;
    std::optional<std::vector<unsigned char>> capture_image(const std::string& capture_id) const;
    std::optional<std::vector<unsigned char>> result_asset(int track_id, const std::string& filename) const;
    void reset();
    void stop();

    // Test/simulation hooks exercise routing and timeouts without a model runtime.
    bool route_segmentation_result(const SegmentationResult& result);
    void evaluate_timeouts(std::chrono::steady_clock::time_point now);
    std::size_t remove_stale_tracks(std::uint64_t current_frame, std::chrono::steady_clock::time_point now);
    std::optional<TrackRecord> track(int track_id) const;

private:
    void segmentation_loop();
    void surface_mapping_loop();
    void update_track(
        const LiveFrame& frame,
        const Detection& detection,
        const std::vector<Detection>& detections,
        double detection_ms,
        bool has_observation
    );
    void maybe_classify_locked(TrackRecord& track, std::chrono::steady_clock::time_point now);
    void complete_track_locked(TrackRecord& track, const FinalClassification& final, std::chrono::steady_clock::time_point now);
    json surface_mapping_request_locked(const TrackRecord& track, const FinalClassification& fallback) const;
    void mark_rejected_locked(TrackRecord& track, const std::string& reason, std::chrono::steady_clock::time_point now, bool save_result);
    void mark_missing_tracks_locked(std::uint64_t current_frame);
    void add_diagnostic_locked(std::string message);
    void release_track_payload_locked(TrackRecord& track);
    void persist_capture_locked(CapturedView& view);
    void save_result_locked(TrackRecord& track);
    json snapshot_locked() const;

    PipelineConfig config_;
    Detector detector_;
    Segmenter segmenter_;
    Cropper cropper_;
    SurfaceMapper surface_mapper_;
    mutable std::mutex tracks_mutex_;
    std::unordered_map<int, TrackRecord> tracks_;
    std::unordered_set<int> terminal_track_ids_;
    std::deque<std::string> recent_tracking_diagnostics_;
    std::unique_ptr<ByteTracker> tracker_;
    BoundedFifoQueue<SegmentationJobV2> segmentation_queue_;
    BoundedFifoQueue<SurfaceMappingJob> surface_mapping_queue_{4};
    std::vector<std::thread> segmentation_threads_;
    std::thread surface_mapping_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> detector_waiting_{0};
    std::atomic<std::uint64_t> generation_{1};
    mutable std::mutex model_mutex_;
    mutable std::condition_variable model_condition_;
    std::uint64_t latest_frame_id_ = 0;
    double detector_fps_ = 0.0;
    double last_segmentation_latency_ms_ = 0.0;
    double last_classification_latency_ms_ = 0.0;
    std::uint64_t completed_count_ = 0;
    std::unordered_map<int, std::uint64_t> class_counts_;
    std::chrono::steady_clock::time_point last_frame_at_;
    std::string capture_run_id_;
};

}  // namespace runner
