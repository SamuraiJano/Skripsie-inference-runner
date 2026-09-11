#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace runner {

// Adapted from ArgoHA/D-FINE-seg dfine_seg/infer/byte_track.py at
// 786f986a7ec95083be8f085590b1ab76a10c9b97. The source is Apache-2.0.

enum class ByteTrackState { Tracked, Lost, Removed };

struct ByteTrackDetection {
    std::array<float, 4> xyxy{};
    float score = 0.0F;
    int class_id = 0;
    int detection_index = -1;
};

struct ByteTrackOutput {
    int track_id = -1;
    int class_id = 0;
    std::array<float, 4> xyxy{};
    float score = 0.0F;
    int detection_index = -1;
    ByteTrackState state = ByteTrackState::Tracked;
};

struct ByteTrackConfig {
    float track_thresh = 0.50F;
    float unmatched_thresh = 0.70F;
    float detrack_thresh = 0.10F;
    float tracking_thresh = 0.70F;
    int track_buffer = 3;
    int max_age = 0;
    int min_hits = 2;
    bool class_aware = true;
    float iou_weight = 0.75F;
    float drag = 0.85F;
    float velocity_alpha = 0.60F;
    bool direction_gate = true;
    float maximum_backward_ratio = 0.05F;
    float maximum_sideways_ratio = 0.25F;

    void validate() const;
};

class ByteTrack {
public:
    explicit ByteTrack(ByteTrackConfig config = {});
    ~ByteTrack();

    std::vector<ByteTrackOutput> update(const std::vector<ByteTrackDetection>& detections, int frame_width, int frame_height);
    void reset();
    void retire_past_bottom(float bottom_y);
    std::size_t active_track_count() const;
    std::size_t lost_track_count() const;
    int next_track_id() const;

private:
    struct Track;
    struct Association;

    ByteTrackConfig config_;
    std::vector<Track> tracks_;
    int frame_index_ = 0;
    int next_id_ = 1;

    Association associate(
        const std::vector<std::size_t>& track_indices,
        const std::vector<ByteTrackDetection>& detections,
        int frame_width,
        int frame_height
    ) const;
};

}  // namespace runner
