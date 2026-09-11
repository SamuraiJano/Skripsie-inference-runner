#include "runner/ByteTrack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace runner {
namespace {

constexpr double forbidden_cost = 1.0e6;

std::array<double, 4> xyxy_to_cxywh(const std::array<float, 4>& box) {
    const double width = std::max(static_cast<double>(box[2] - box[0]), 1.0);
    const double height = std::max(static_cast<double>(box[3] - box[1]), 1.0);
    return {box[0] + width * 0.5, box[1] + height * 0.5, width, height};
}

std::array<float, 4> cxywh_to_xyxy(const std::array<double, 4>& value) {
    return {
        static_cast<float>(value[0] - value[2] * 0.5), static_cast<float>(value[1] - value[3] * 0.5),
        static_cast<float>(value[0] + value[2] * 0.5), static_cast<float>(value[1] + value[3] * 0.5),
    };
}

double iou(const std::array<float, 4>& left, const std::array<float, 4>& right) {
    const double x1 = std::max(left[0], right[0]);
    const double y1 = std::max(left[1], right[1]);
    const double x2 = std::min(left[2], right[2]);
    const double y2 = std::min(left[3], right[3]);
    const double intersection = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double left_area = std::max(0.0F, left[2] - left[0]) * std::max(0.0F, left[3] - left[1]);
    const double right_area = std::max(0.0F, right[2] - right[0]) * std::max(0.0F, right[3] - right[1]);
    const double total = left_area + right_area - intersection;
    return total > 0.0 ? intersection / total : 0.0;
}

// Minimum-cost Hungarian assignment for a rectangular matrix. Dummy rows or
// columns carry forbidden cost, and forbidden matches are discarded by caller.
std::vector<std::pair<std::size_t, std::size_t>> hungarian(const std::vector<std::vector<double>>& input) {
    if (input.empty() || input.front().empty()) return {};
    const std::size_t rows = input.size();
    const std::size_t columns = input.front().size();
    const std::size_t size = std::max(rows, columns);
    std::vector<std::vector<double>> cost(size, std::vector<double>(size, forbidden_cost));
    for (std::size_t row = 0; row < rows; ++row) {
        if (input[row].size() != columns) throw std::invalid_argument("association cost matrix is ragged");
        std::copy(input[row].begin(), input[row].end(), cost[row].begin());
    }
    std::vector<double> u(size + 1, 0.0), v(size + 1, 0.0);
    std::vector<std::size_t> p(size + 1, 0), way(size + 1, 0);
    for (std::size_t row = 1; row <= size; ++row) {
        p[0] = row;
        std::size_t column0 = 0;
        std::vector<double> minimum(size + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(size + 1, false);
        do {
            used[column0] = true;
            const std::size_t row0 = p[column0];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t column1 = 0;
            for (std::size_t column = 1; column <= size; ++column) {
                if (used[column]) continue;
                const double current = cost[row0 - 1][column - 1] - u[row0] - v[column];
                if (current < minimum[column]) {
                    minimum[column] = current;
                    way[column] = column0;
                }
                if (minimum[column] < delta) {
                    delta = minimum[column];
                    column1 = column;
                }
            }
            for (std::size_t column = 0; column <= size; ++column) {
                if (used[column]) {
                    u[p[column]] += delta;
                    v[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            column0 = column1;
        } while (p[column0] != 0);
        do {
            const std::size_t column1 = way[column0];
            p[column0] = p[column1];
            column0 = column1;
        } while (column0 != 0);
    }
    std::vector<std::pair<std::size_t, std::size_t>> result;
    for (std::size_t column = 1; column <= size; ++column) {
        if (p[column] > 0 && p[column] <= rows && column <= columns) result.emplace_back(p[column] - 1, column - 1);
    }
    return result;
}

}  // namespace

struct ByteTrack::Track {
    int track_id = -1;
    int class_id = 0;
    float score = 0.0F;
    std::array<double, 4> mean{};
    std::array<double, 4> velocity{};
    ByteTrackState state = ByteTrackState::Tracked;
    int age = 1;
    int hits = 1;
    int time_since_update = 0;
    int detection_index = -1;

    std::array<double, 4> predicted_mean(float drag) const {
        double damping = 1.0;
        if (state == ByteTrackState::Lost && time_since_update > 1) damping = std::pow(drag, time_since_update - 1);
        std::array<double, 4> result{};
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = mean[index] + velocity[index] * time_since_update * damping;
        }
        result[2] = std::max(result[2], 1.0);
        result[3] = std::max(result[3], 1.0);
        return result;
    }

    void predict() {
        ++age;
        ++time_since_update;
        detection_index = -1;
    }

    void update(const ByteTrackDetection& detection, float velocity_alpha) {
        const auto next = xyxy_to_cxywh(detection.xyxy);
        const int gap = std::max(time_since_update, 1);
        for (std::size_t index = 0; index < velocity.size(); ++index) {
            const double observed = (next[index] - mean[index]) / gap;
            velocity[index] = hits < 2 ? observed : velocity_alpha * velocity[index] + (1.0F - velocity_alpha) * observed;
        }
        mean = next;
        score = detection.score;
        ++hits;
        time_since_update = 0;
        state = ByteTrackState::Tracked;
        detection_index = detection.detection_index;
    }
};

struct ByteTrack::Association {
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    std::vector<std::size_t> unmatched_tracks;
    std::vector<std::size_t> unmatched_detections;
};

void ByteTrackConfig::validate() const {
    const auto probability = [](float value) { return std::isfinite(value) && value >= 0.0F && value <= 1.0F; };
    if (!probability(track_thresh) || !probability(unmatched_thresh) || !probability(detrack_thresh)
        || !probability(tracking_thresh) || !probability(iou_weight) || !probability(drag) || !probability(velocity_alpha)) {
        throw std::invalid_argument("ByteTrack thresholds and weights must be between 0 and 1");
    }
    if (!(detrack_thresh < track_thresh && track_thresh <= unmatched_thresh)) {
        throw std::invalid_argument("ByteTrack requires detrack_thresh < track_thresh <= unmatched_thresh");
    }
    if (track_buffer < 0 || max_age < 0 || min_hits < 1) throw std::invalid_argument("ByteTrack age, buffer, and hit settings are invalid");
    if (maximum_backward_ratio < 0.0F || maximum_sideways_ratio < 0.0F) throw std::invalid_argument("direction gate ratios cannot be negative");
}

ByteTrack::ByteTrack(ByteTrackConfig config) : config_(config) {
    config_.validate();
}

ByteTrack::~ByteTrack() = default;

ByteTrack::Association ByteTrack::associate(
    const std::vector<std::size_t>& track_indices,
    const std::vector<ByteTrackDetection>& detections,
    int frame_width,
    int frame_height
) const {
    Association result;
    if (track_indices.empty() || detections.empty()) {
        result.unmatched_tracks.resize(track_indices.size());
        result.unmatched_detections.resize(detections.size());
        for (std::size_t index = 0; index < track_indices.size(); ++index) result.unmatched_tracks[index] = index;
        for (std::size_t index = 0; index < detections.size(); ++index) result.unmatched_detections[index] = index;
        return result;
    }
    const double diagonal = std::max(std::hypot(static_cast<double>(frame_width), static_cast<double>(frame_height)), 1.0);
    std::vector<std::vector<double>> costs(track_indices.size(), std::vector<double>(detections.size(), forbidden_cost));
    for (std::size_t row = 0; row < track_indices.size(); ++row) {
        const Track& track = tracks_[track_indices[row]];
        const auto predicted_mean = track.predicted_mean(config_.drag);
        const auto predicted_box = cxywh_to_xyxy(predicted_mean);
        for (std::size_t column = 0; column < detections.size(); ++column) {
            const auto& detection = detections[column];
            if (config_.class_aware && track.class_id != detection.class_id) continue;
            const auto detection_mean = xyxy_to_cxywh(detection.xyxy);
            const double dx = detection_mean[0] - track.mean[0];
            const double dy = detection_mean[1] - track.mean[1];
            const int gap = std::max(track.time_since_update, 1);
            if (config_.direction_gate && (dy < -config_.maximum_backward_ratio * frame_height * gap
                || std::abs(dx) > config_.maximum_sideways_ratio * frame_width * gap)) continue;
            const double centroid = std::min(1.0, std::hypot(detection_mean[0] - predicted_mean[0], detection_mean[1] - predicted_mean[1]) / diagonal);
            const double cost = config_.iou_weight * (1.0 - iou(predicted_box, detection.xyxy)) + (1.0 - config_.iou_weight) * centroid;
            if (cost <= config_.tracking_thresh) costs[row][column] = cost;
        }
    }
    std::set<std::size_t> matched_rows;
    std::set<std::size_t> matched_columns;
    for (const auto [row, column] : hungarian(costs)) {
        if (costs[row][column] >= forbidden_cost) continue;
        result.matches.emplace_back(row, column);
        matched_rows.insert(row);
        matched_columns.insert(column);
    }
    for (std::size_t row = 0; row < track_indices.size(); ++row) if (matched_rows.count(row) == 0) result.unmatched_tracks.push_back(row);
    for (std::size_t column = 0; column < detections.size(); ++column) if (matched_columns.count(column) == 0) result.unmatched_detections.push_back(column);
    return result;
}

std::vector<ByteTrackOutput> ByteTrack::update(const std::vector<ByteTrackDetection>& detections, int frame_width, int frame_height) {
    if (frame_width <= 0 || frame_height <= 0) throw std::invalid_argument("tracking frame dimensions must be positive");
    ++frame_index_;
    std::vector<ByteTrackDetection> high;
    std::vector<ByteTrackDetection> low;
    for (std::size_t index = 0; index < detections.size(); ++index) {
        ByteTrackDetection detection = detections[index];
        if (detection.detection_index < 0) detection.detection_index = static_cast<int>(index);
        if (detection.score < config_.detrack_thresh) continue;
        if (detection.xyxy[2] <= detection.xyxy[0] || detection.xyxy[3] <= detection.xyxy[1]) continue;
        (detection.score >= config_.track_thresh ? high : low).push_back(detection);
    }
    for (auto& track : tracks_) if (track.state != ByteTrackState::Removed) track.predict();
    std::vector<std::size_t> pool;
    for (std::size_t index = 0; index < tracks_.size(); ++index) if (tracks_[index].state != ByteTrackState::Removed) pool.push_back(index);

    const Association first = associate(pool, high, frame_width, frame_height);
    for (const auto [local_track, detection] : first.matches) tracks_[pool[local_track]].update(high[detection], config_.velocity_alpha);

    std::vector<std::size_t> second_pool;
    std::vector<std::size_t> second_to_first;
    for (const std::size_t local : first.unmatched_tracks) {
        if (tracks_[pool[local]].state == ByteTrackState::Tracked) {
            second_pool.push_back(pool[local]);
            second_to_first.push_back(local);
        }
    }
    const Association second = associate(second_pool, low, frame_width, frame_height);
    std::set<std::size_t> second_matched_first;
    for (const auto [local_track, detection] : second.matches) {
        tracks_[second_pool[local_track]].update(low[detection], config_.velocity_alpha);
        second_matched_first.insert(second_to_first[local_track]);
    }
    for (const std::size_t local : first.unmatched_tracks) {
        if (second_matched_first.count(local) == 0 && tracks_[pool[local]].state == ByteTrackState::Tracked) {
            tracks_[pool[local]].state = ByteTrackState::Lost;
        }
    }
    for (const std::size_t detection_index : first.unmatched_detections) {
        const auto& detection = high[detection_index];
        if (detection.score < config_.unmatched_thresh) continue;
        Track track;
        track.track_id = next_id_++;
        track.class_id = detection.class_id;
        track.score = detection.score;
        track.mean = xyxy_to_cxywh(detection.xyxy);
        track.detection_index = detection.detection_index;
        tracks_.push_back(track);
    }
    for (auto& track : tracks_) {
        if (track.state == ByteTrackState::Removed) continue;
        if (track.time_since_update > config_.track_buffer || (config_.max_age > 0 && track.age > config_.max_age)) {
            track.state = ByteTrackState::Removed;
        }
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const Track& track) {
        return track.state == ByteTrackState::Removed;
    }), tracks_.end());

    std::vector<ByteTrackOutput> output;
    for (const auto& track : tracks_) {
        if (track.state != ByteTrackState::Tracked || track.time_since_update != 0 || track.hits < config_.min_hits) continue;
        auto box = cxywh_to_xyxy(track.mean);
        box[0] = std::clamp(box[0], 0.0F, static_cast<float>(frame_width));
        box[1] = std::clamp(box[1], 0.0F, static_cast<float>(frame_height));
        box[2] = std::clamp(box[2], 0.0F, static_cast<float>(frame_width));
        box[3] = std::clamp(box[3], 0.0F, static_cast<float>(frame_height));
        output.push_back({track.track_id, track.class_id, box, track.score, track.detection_index, ByteTrackState::Tracked});
    }
    return output;
}

void ByteTrack::reset() {
    tracks_.clear();
    frame_index_ = 0;
    next_id_ = 1;
}

void ByteTrack::retire_past_bottom(float bottom_y) {
    for (auto& track : tracks_) {
        if (track.predicted_mean(config_.drag)[1] >= bottom_y) track.state = ByteTrackState::Removed;
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const Track& track) {
        return track.state == ByteTrackState::Removed;
    }), tracks_.end());
}

std::size_t ByteTrack::active_track_count() const {
    return static_cast<std::size_t>(std::count_if(tracks_.begin(), tracks_.end(), [](const Track& track) {
        return track.state == ByteTrackState::Tracked && track.time_since_update == 0;
    }));
}

std::size_t ByteTrack::lost_track_count() const {
    return static_cast<std::size_t>(std::count_if(tracks_.begin(), tracks_.end(), [](const Track& track) {
        return track.state == ByteTrackState::Lost;
    }));
}

int ByteTrack::next_track_id() const { return next_id_; }

}  // namespace runner
