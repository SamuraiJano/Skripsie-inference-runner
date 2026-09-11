#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "runner/ByteTrack.hpp"
#include "runner/Types.hpp"

namespace runner {

struct NormalizedRoi {
    float x = 0.0F;
    float y = 0.0F;
    float width = 1.0F;
    float height = 1.0F;

    void validate() const;
};

struct VisibleTrack {
    int track_id = -1;
    int class_id = 0;
    std::string label;
    float confidence = 0.0F;
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    float progress = 0.0F;
    int detection_index = -1;
};

struct TrackingFrameResult {
    std::string session_id;
    std::uint64_t frame_id = 0;
    std::string model_id;
    std::string model_family;
    std::string backend;
    int source_width = 0;
    int source_height = 0;
    std::vector<VisibleTrack> tracks;
    Timing timing;
    double total_ms = 0.0;

    nlohmann::json to_json() const;
};

class TrackingPipeline {
public:
    TrackingPipeline(
        std::string session_id,
        std::string model_id,
        std::string model_family,
        std::string backend,
        NormalizedRoi roi = {},
        ByteTrackConfig tracker_config = {}
    );

    TrackingFrameResult process(std::uint64_t frame_id, double timestamp_ms, const InferenceResult& detector_result);
    void reset();
    void set_roi(NormalizedRoi roi);
    void set_model(std::string model_id, std::string model_family, std::string backend);
    nlohmann::json status() const;
    const std::string& session_id() const;
    const std::string& model_id() const;
    const NormalizedRoi& roi() const;

private:
    std::string session_id_;
    std::string model_id_;
    std::string model_family_;
    std::string backend_;
    NormalizedRoi roi_;
    ByteTrack tracker_;
    std::uint64_t last_frame_id_ = 0;
    double last_timestamp_ms_ = -1.0;
    std::size_t visible_tracks_ = 0;
};

ByteTrackConfig byte_track_config_from_json(const nlohmann::json& value);
NormalizedRoi roi_from_json(const nlohmann::json& value);

}  // namespace runner
