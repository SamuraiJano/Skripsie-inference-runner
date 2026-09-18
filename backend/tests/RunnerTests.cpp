#include <cassert>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "runner/Types.hpp"
#include "runner/YoloPostprocess.hpp"
#include "runner/Pipeline.hpp"

namespace {

runner::json manifest(const std::string& precision) {
    const bool fp16 = precision == "fp16";
    const bool int8 = precision == "int8";
    return {
        {"schemaVersion", 2}, {"name", "fixture"}, {"task", "yolo-segment"},
        {"precision", precision}, {"framework", "ncnn"},
        {"checkpointSha256", std::string(64, 'a')}, {"datasetHash", std::string(64, 'b')},
        {"splitHash", std::string(64, 'c')}, {"classSchemaHash", std::string(64, 'd')},
        {"exporter", {{"name", "ultralytics"}, {"version", "8.4.95"}}},
        {"runtime", {
            {"name", "ncnn"}, {"version", "1.0.20260526"}, {"threads", 8},
            {"useVulkanCompute", true}, {"gpuDevice", 0},
            {"useInt8Inference", int8}, {"useInt8Storage", int8},
            {"useFp16Storage", fp16}, {"useFp16Arithmetic", false}
        }},
        {"calibrationId", int8 ? runner::json("cal-fixture") : runner::json(nullptr)},
        {"artifactVerification", {{"verified", true}, {"quantized", int8}}},
        {"input", {{"width", 640}, {"height", 640}, {"blob", "in0"},
                    {"mean", {0.0, 0.0, 0.0}}, {"norm", {1.0 / 255, 1.0 / 255, 1.0 / 255}}}},
        {"outputs", {"out0", "out1"}}, {"labels", {"class_1", "class_2", "class_3"}},
        {"maskThreshold", 0.5}, {"maxDetections", 300},
    };
}

void expect_invalid(runner::json value) {
    bool thrown = false;
    try {
        (void)runner::ModelConfig::from_json(value);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);
}

void test_precision_flags() {
    const auto fp32 = runner::ModelConfig::from_json(manifest("fp32"));
    const auto fp16 = runner::ModelConfig::from_json(manifest("fp16"));
    const auto int8 = runner::ModelConfig::from_json(manifest("int8"));
    assert(fp32.precision == runner::ModelPrecision::FP32 && !fp32.use_fp16_storage && !fp32.use_int8_storage);
    assert(fp16.precision == runner::ModelPrecision::FP16 && fp16.use_fp16_storage && !fp16.use_fp16_arithmetic);
    assert(int8.precision == runner::ModelPrecision::INT8 && int8.use_int8_storage && int8.artifact_quantized);
    assert(fp16.use_vulkan_compute && fp16.gpu_device == 0 && fp16.threads == 8);

    auto invalid_fp32 = manifest("fp32");
    invalid_fp32["runtime"]["useInt8Inference"] = true;
    expect_invalid(invalid_fp32);
    auto invalid_int8 = manifest("int8");
    invalid_int8["calibrationId"] = nullptr;
    expect_invalid(invalid_int8);
    auto missing_prototype = manifest("fp16");
    missing_prototype["outputs"] = {"out0"};
    expect_invalid(missing_prototype);
    auto cpu_runtime = manifest("fp16");
    cpu_runtime["runtime"]["useVulkanCompute"] = false;
    expect_invalid(cpu_runtime);
    auto reordered_classes = manifest("fp16");
    reordered_classes["labels"] = {"class_2", "class_1", "class_3"};
    const auto general_segmentation = runner::ModelConfig::from_json(reordered_classes);
    assert(general_segmentation.labels.front() == "class_2");
}

void test_direct_artifact_contract_and_delegated_result() {
    runner::json direct = {
        {"schemaVersion", 3}, {"name", "direct fixture"}, {"task", "yolo-detect"},
        {"precision", "fp16"}, {"framework", "pytorch"},
        {"artifact", {
            {"format", "pt"}, {"file", "model.pt"}, {"backend", "ultralytics"},
            {"accelerator", "CUDA"}, {"trustedCheckpoint", true},
        }},
        {"input", {{"width", 640}, {"height", 640}, {"blob", "images"},
                   {"mean", {0.0, 0.0, 0.0}}, {"norm", {1.0 / 255, 1.0 / 255, 1.0 / 255}}}},
        {"outputs", {"auto"}}, {"labels", runner::json::array()},
        {"runtime", {{"name", "PyTorch"}, {"threads", 8}, {"useVulkanCompute", false}, {"gpuDevice", -1}}},
    };
    const auto config = runner::ModelConfig::from_json(direct);
    assert(config.format == "pt" && config.backend == "ultralytics" && config.accelerator == "CUDA");
    assert(config.precision == runner::ModelPrecision::FP16 && !config.use_vulkan_compute);
    auto missing_artifact = direct;
    missing_artifact["artifact"]["file"] = "";
    expect_invalid(missing_artifact);

    const runner::json delegated = {
        {"modelId", "direct-fixture"}, {"task", "yolo-detect"},
        {"sourceWidth", 100}, {"sourceHeight", 80},
        {"detections", {{
            {"classId", 2}, {"label", "defect"}, {"confidence", 0.9},
            {"x", 1.0}, {"y", 2.0}, {"width", 30.0}, {"height", 40.0}, {"trackId", nullptr},
            {"mask", {{"encoding", "rle-row-major"}, {"width", 2}, {"height", 2}, {"counts", {1, 2, 1}}}},
        }}},
        {"classifications", runner::json::array()}, {"tensors", runner::json::array()},
        {"timing", {{"inferenceMs", 3.5}, {"backendTotalMs", 4.0}}},
    };
    const auto result = runner::InferenceResult::from_json(delegated);
    assert(result.model_id == "direct-fixture" && result.detections.size() == 1);
    assert(result.detections.front().class_id == 2 && result.detections.front().mask.counts.size() == 3);
    assert(result.timing.inference_ms == 3.5 && result.timing.backend_total_ms == 4.0);

    direct["task"] = "vision-grounding";
    direct["precision"] = "bf16";
    direct["framework"] = "locate-anything";
    direct["artifact"] = {
        {"format", "locate-anything"}, {"file", "model.locateanything.json"}, {"backend", "locate-anything"},
        {"accelerator", "CUDA + CPU offload"}, {"trustedCheckpoint", true},
    };
    const auto grounding = runner::ModelConfig::from_json(direct);
    assert(grounding.task == runner::ModelTask::VisionGrounding);
    assert(grounding.precision == runner::ModelPrecision::BF16 && grounding.backend == "locate-anything");

    direct["precision"] = "fp32";
    direct["framework"] = "grounding-dino";
    direct["artifact"] = {
        {"format", "grounding-dino"}, {"file", "model.groundingdino.json"}, {"backend", "grounding-dino"},
        {"accelerator", "CUDA"}, {"trustedCheckpoint", false},
    };
    const auto grounding_dino = runner::ModelConfig::from_json(direct);
    assert(grounding_dino.task == runner::ModelTask::VisionGrounding);
    assert(grounding_dino.precision == runner::ModelPrecision::FP32 && grounding_dino.backend == "grounding-dino");

    direct["task"] = "yolo-detect";
    direct["framework"] = "pytorch";
    direct["artifact"] = {
        {"format", "dfine"}, {"file", "model.pth"}, {"backend", "dfine"},
        {"accelerator", "CUDA"}, {"trustedCheckpoint", true},
    };
    direct["labels"] = {"object"};
    const auto dfine = runner::ModelConfig::from_json(direct);
    assert(dfine.task == runner::ModelTask::YoloDetect && dfine.backend == "dfine");
    assert(dfine.precision == runner::ModelPrecision::FP32 && dfine.trusted_checkpoint);

    const runner::json grounded_result = {
        {"modelId", "locate-anything"}, {"task", "vision-grounding"},
        {"sourceWidth", 200}, {"sourceHeight", 100},
        {"detections", runner::json::array()},
        {"points", {{{"classId", 0}, {"label", "button"}, {"x", 150.0}, {"y", 25.0}}}},
        {"classifications", runner::json::array()}, {"tensors", runner::json::array()},
        {"answer", "<ref>button</ref><box><750><250></box>"},
    };
    const auto grounded = runner::InferenceResult::from_json(grounded_result);
    assert(grounded.points.size() == 1 && grounded.points.front().label == "button");
    assert(!grounded.answer.empty());
}

void test_deterministic_mask_decode() {
    runner::MaskDecodeRequest request;
    request.source_width = 4;
    request.source_height = 4;
    request.input_width = 4;
    request.input_height = 4;
    request.prototype_width = 2;
    request.prototype_height = 2;
    request.prototype_channels = 1;
    request.letterbox_scale = 1.0F;
    request.box_x = 1.0F;
    request.box_y = 1.0F;
    request.box_width = 2.0F;
    request.box_height = 2.0F;
    request.threshold = 0.5F;
    request.coefficients = {1.0F};
    request.prototypes = {10.0F, 10.0F, 10.0F, 10.0F};
    const auto encoded = runner::decode_yolo_mask_rle(request);
    const auto pixels = runner::decode_rle(encoded);
    const std::vector<unsigned char> expected = {
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0,
    };
    assert(pixels == expected);
    assert(encoded.counts.size() < pixels.size());
}

void test_bytetrack_and_grade_aggregation() {
    runner::ByteTracker tracker(0.5F, 0.1F, 0.2F, 2, 0.7F);
    runner::Detection first{0, "fruit", 0.9F, 10.0F, 10.0F, 20.0F, 20.0F};
    auto frame_one = tracker.update(1, {first});
    assert(frame_one.size() == 1 && frame_one[0].track_id == 1);
    runner::Detection second{0, "fruit", 0.3F, 11.0F, 10.0F, 20.0F, 20.0F};
    auto frame_two = tracker.update(2, {second});
    assert(frame_two.size() == 1 && frame_two[0].track_id == 1);  // ByteTrack low-score second association
    second.x = 12.0F;
    auto frame_three = tracker.update(3, {second});
    assert(frame_three.size() == 1 && frame_three[0].track_id == 1);  // intermittent low confidence keeps the ID
    runner::Detection weak_new{0, "fruit", 0.6F, 80.0F, 80.0F, 10.0F, 10.0F};
    assert(tracker.update(4, {weak_new}, {true}).empty());  // association threshold is not a creation threshold
    runner::Detection gated_new{0, "fruit", 0.95F, 80.0F, 80.0F, 10.0F, 10.0F};
    assert(tracker.update(5, {gated_new}, {false}).empty());

    runner::ByteTracker predictive_tracker(0.5F, 0.1F, 0.2F, 5, 0.7F);
    runner::Detection moving{0, "fruit", 0.9F, 10.0F, 10.0F, 20.0F, 20.0F};
    (void)predictive_tracker.update(1, {moving});
    moving.y = 20.0F;
    (void)predictive_tracker.update(2, {moving});
    moving.y = 30.0F;
    (void)predictive_tracker.update(3, {moving});
    const auto predicted = predictive_tracker.update(4, {}, {}, true);
    assert(predicted.size() == 1 && predicted.front().track_id == 1);
    assert(std::abs(predicted.front().y - 40.0F) < 0.001F);
    assert(!predictive_tracker.observed_in_frame(1, 4));
    moving.y = 50.0F;
    const auto reacquired = predictive_tracker.update(5, {moving}, {}, true);
    assert(reacquired.size() == 1 && reacquired.front().track_id == 1);
    assert(predictive_tracker.observed_in_frame(1, 5));

    runner::EvidenceAggregator grading(3, 0.1F);
    assert(grading.add({1, 2, 0.8F, 1}).state == "uncertain/insufficient_evidence");
    assert(grading.add({1, 2, 0.7F, 2}).state == "uncertain/insufficient_evidence");
    const auto final = grading.add({1, 2, 0.9F, 3});
    assert(final.state == "class_3" && final.class_id == 2);
}

void test_pipeline_serializes_model_callbacks() {
    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};
    std::atomic<int> outputs{0};
    const auto enter_model = [&] {
        const int now = ++active;
        int previous = maximum_active.load();
        while (now > previous && !maximum_active.compare_exchange_weak(previous, now)) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
        --active;
    };
    runner::AsyncFruitPipeline pipeline(
        [&](const runner::PipelineFrame&) {
            enter_model();
            return std::vector<runner::Detection>{{0, "fruit", 0.9F, 0.0F, 0.0F, 10.0F, 10.0F}};
        },
        [](const runner::PipelineFrame& frame, const std::vector<runner::Detection>& detections) {
            if (detections.empty()) return std::optional<runner::SegmentationJob>{};
            return std::optional<runner::SegmentationJob>{{frame.frame_id, detections.front().track_id, {1}}};
        },
        [&](const runner::SegmentationJob& job) {
            enter_model();
            return std::vector<runner::GradeObservation>{{job.track_id, 0, 0.9F, job.frame_id}};
        },
        [&](const runner::PipelineFrame&, const std::vector<runner::Detection>&, const std::vector<runner::GradeDecision>&) {
            ++outputs;
        }
    );
    pipeline.start();
    for (std::uint64_t frame_id = 1; frame_id <= 6; ++frame_id) {
        pipeline.submit({frame_id, std::chrono::steady_clock::now(), {1}});
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    pipeline.stop();
    assert(maximum_active == 1);
    assert(outputs > 0);
}

runner::LiveFrame live_frame(std::uint64_t frame_id) {
    return {frame_id, std::chrono::steady_clock::now(), runner::utc_now_iso8601(), 1000, 1000, {1}, {1}};
}

runner::PipelineConfig pipeline_config() {
    runner::PipelineConfig config;
    config.bytetrack_match_iou = 0.1F;
    config.crop_margin = 0.0;
    config.sharpness_threshold = 0.0;
    config.minimum_crop_width = 10;
    config.minimum_crop_height = 10;
    config.classification_delay_seconds = 0.0;
    config.classification_timeout_seconds = 0.025;
    config.track_retention_seconds = 100.0;
    config.stale_track_frames = 20;
    config.output_directory = std::filesystem::path("build") / "test-output";
    config.logging_directory = std::filesystem::path("build") / "test-logs";
    return config;
}

std::unique_ptr<runner::MultiViewPipeline> simulated_pipeline(
    runner::PipelineConfig config,
    std::vector<double> centers,
    std::function<bool(std::uint64_t)> accept_crop = [](std::uint64_t) { return true; },
    std::function<runner::SegmentationResult(const runner::SegmentationJobV2&)> segment = {},
    runner::MultiViewPipeline::SurfaceMapper surface_mapper = {}
) {
    auto detector = [centers = std::move(centers)](const runner::LiveFrame& frame) {
        runner::InferenceResult result;
        result.source_width = frame.width;
        result.source_height = frame.height;
        if (frame.frame_id <= centers.size() && centers[static_cast<std::size_t>(frame.frame_id - 1)] >= 0.0) {
            const float center = static_cast<float>(centers[static_cast<std::size_t>(frame.frame_id - 1)]);
            result.detections.push_back({0, "naartjie", 0.95F, 380.0F, center - 90.0F, 240.0F, 180.0F});
        }
        result.timing.backend_total_ms = 3.0;
        return result;
    };
    if (!segment) {
        segment = [](const runner::SegmentationJobV2& job) {
            runner::SegmentationResult result;
            result.predicted_class = job.view_id == 2 ? 2 : 0;
            result.confidence = 0.9F;
            result.success = true;
            result.processing_ms = 2.0;
            return result;
        };
    }
    auto pipeline = std::make_unique<runner::MultiViewPipeline>(
        config,
        std::move(detector),
        std::move(segment),
        [accept_crop = std::move(accept_crop)](const runner::LiveFrame& frame, const runner::CropBounds& bounds) {
            if (!accept_crop(frame.frame_id)) return runner::CropImage{};
            return runner::CropImage{{1, 2, 3}, bounds.width, bounds.height, 100.0};
        },
        std::move(surface_mapper)
    );
    pipeline->start();
    return pipeline;
}

std::unique_ptr<runner::MultiViewPipeline> scripted_pipeline(
    runner::PipelineConfig config,
    std::vector<std::vector<runner::Detection>> frames
) {
    auto pipeline = std::make_unique<runner::MultiViewPipeline>(
        config,
        [frames = std::move(frames)](const runner::LiveFrame& frame) {
            runner::InferenceResult result;
            result.source_width = frame.width;
            result.source_height = frame.height;
            if (frame.frame_id <= frames.size()) result.detections = frames[static_cast<std::size_t>(frame.frame_id - 1)];
            return result;
        },
        [](const runner::SegmentationJobV2& job) {
            runner::SegmentationResult result;
            result.predicted_class = job.view_id % 3;
            result.confidence = 0.9F;
            result.success = true;
            return result;
        },
        [](const runner::LiveFrame&, const runner::CropBounds& bounds) {
            return runner::CropImage{{1}, bounds.width, bounds.height, 100.0};
        }
    );
    pipeline->start();
    return pipeline;
}

bool wait_for(const std::function<bool()>& predicate, int milliseconds = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}

void test_zone_progress_capture_and_crop_math() {
    assert(runner::acquisition_progress(100.0, 100.0, 700.0) == 0.0);
    assert(std::abs(runner::acquisition_progress(400.0, 100.0, 700.0) - 0.5) < 1e-9);
    assert(runner::acquisition_progress(800.0, 100.0, 700.0) == 1.0);
    const std::vector<double> targets{0.2, 0.4, 0.6, 0.8};
    std::vector<bool> captured(4, false);
    assert(runner::next_capture_target(0.19, targets, captured) == std::nullopt);
    assert(runner::next_capture_target(0.20, targets, captured) == 0);
    captured[0] = true;
    assert(runner::next_capture_target(0.39, targets, captured) == std::nullopt);
    assert(runner::next_capture_target(0.40, targets, captured) == 1);
    captured[1] = true;
    assert(runner::next_capture_target(0.60, targets, captured) == 2);
    captured[2] = true;
    assert(runner::next_capture_target(0.80, targets, captured) == 3);
    captured[3] = true;
    assert(runner::next_capture_target(1.0, targets, captured) == std::nullopt);  // duplicate prevention

    const runner::Detection exact{0, "fruit", 0.9F, 10.0F, 20.0F, 30.0F, 40.0F};
    const auto exact_crop = runner::clamp_crop(exact, 0.0, 100, 100);
    assert(exact_crop.x == 10 && exact_crop.y == 20 && exact_crop.width == 30 && exact_crop.height == 40);

    const runner::Detection edge{0, "fruit", 0.9F, -5.0F, 90.0F, 30.0F, 20.0F};
    const auto crop = runner::clamp_crop(edge, 0.2, 100, 100);
    assert(crop.x == 0 && crop.y == 86 && crop.width == 31 && crop.height == 14);
    assert(runner::crossed_classification_line(699.0, 700.0, 700.0));
    assert(!runner::crossed_classification_line(700.0, 701.0, 700.0));
}

void test_worst_confident_class_aggregation() {
    std::vector<runner::SegmentationResult> results(4);
    for (int index = 0; index < 4; ++index) {
        results[static_cast<std::size_t>(index)].predicted_class = index == 2 ? 2 : index % 2;
        results[static_cast<std::size_t>(index)].confidence = 0.8F;
        results[static_cast<std::size_t>(index)].success = true;
        results[static_cast<std::size_t>(index)].defect_area = 10.0 + index;
    }
    results[3].confidence = 0.2F;  // rejected by threshold
    const auto final = runner::aggregate_track_results(results, 0.55F, 4, false);
    assert(final.success && final.final_class == 2 && final.successful_results == 3);
    assert(final.view_classes.size() == 3 && final.total_visible_defect_area == 33.0);
}

void test_routing_four_view_classification_and_no_duplicate() {
    auto pipeline = simulated_pipeline(pipeline_config(), {120.0, 200.0, 310.0, 375.0, 440.0, 505.0, 570.0, 635.0, 700.0, 840.0});
    for (std::uint64_t frame = 1; frame <= 10; ++frame) (void)pipeline->process_frame(live_frame(frame));
    assert(!pipeline->route_segmentation_result({0, 999, 1, 0, {}, 0, 0, {}, 0, 0, false, "wrong track"}));
    const bool became_classified = wait_for([&] {
        const auto track = pipeline->track(1);
        return track && track->state == runner::TrackState::Completed;
    });
    if (!became_classified) {
        const auto failed_track = pipeline->track(1);
        if (failed_track) {
            std::cerr << "classification wait failed: state=" << runner::track_state_name(failed_track->state)
                      << " views=" << failed_track->captured_views.size()
                      << " results=" << failed_track->segmentation_results.size() << '\n';
            for (const auto& message : failed_track->messages) std::cerr << "  " << message << '\n';
        }
    }
    assert(became_classified);
    const auto classified = pipeline->track(1);
    assert(classified && classified->captured_views.size() == 4 && classified->views_used == 4);
    assert(classified->final_class == 2 && !classified->incomplete_evidence);
    const std::string capture_id = classified->captured_views.front().capture_id;
    assert(!capture_id.empty());
    const auto capture_image = pipeline->capture_image(capture_id);
    assert(capture_image && capture_image->size() == 3);
    const auto capture_library = pipeline->capture_library();
    assert(std::any_of(capture_library.begin(), capture_library.end(), [&](const runner::json& item) {
        return item.value("captureId", "") == capture_id && item.value("trackId", -1) == 1;
    }));
    const std::string timestamp = classified->classification_timestamp;
    const auto duplicate = classified->segmentation_results.front();
    assert(!pipeline->route_segmentation_result(duplicate));
    pipeline->evaluate_timeouts(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    assert(pipeline->track(1)->classification_timestamp == timestamp);  // no duplicate classification
    assert(pipeline->snapshot().at("completedCount") == 1);
    for (std::uint64_t frame = 11; frame <= 45; ++frame) (void)pipeline->process_frame(live_frame(frame));
    const auto empty_belt_snapshot = pipeline->snapshot();
    assert(empty_belt_snapshot.at("tracks").size() == 1 && empty_belt_snapshot.at("completedCount") == 1);
    pipeline->reset();
    const auto reset_snapshot = pipeline->snapshot();
    assert(reset_snapshot.at("tracks").empty() && reset_snapshot.at("completedCount") == 0);  // no state leak
    pipeline->stop();
}

void test_prediction_only_frames_fill_all_four_capture_targets() {
    auto pipeline = simulated_pipeline(
        pipeline_config(),
        {120.0, 180.0, 240.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 780.0}
    );
    for (std::uint64_t frame = 1; frame <= 11; ++frame) (void)pipeline->process_frame(live_frame(frame));

    const auto predicted_track = pipeline->track(1);
    assert(predicted_track && predicted_track->state == runner::TrackState::Lost);
    assert(predicted_track->last_seen_frame == 3 && !predicted_track->has_current_observation);
    assert(predicted_track->captured_views.size() == 4);
    const std::vector<std::uint64_t> expected_frames{5, 7, 9, 11};
    for (std::size_t index = 0; index < predicted_track->captured_views.size(); ++index) {
        const auto& view = predicted_track->captured_views[index];
        assert(view.view_id == static_cast<int>(index));
        assert(view.frame_id == expected_frames[index]);
        assert(view.used_tracking_prediction);
    }

    (void)pipeline->process_frame(live_frame(12));
    const auto reacquired_track = pipeline->track(1);
    assert(reacquired_track && reacquired_track->track_id == 1);
    assert(reacquired_track->last_seen_frame == 12 && reacquired_track->has_current_observation);
    assert(reacquired_track->captured_views.size() == 4);
    pipeline->stop();
}

void test_two_view_timeout_and_insufficient_evidence() {
    auto two_view = simulated_pipeline(
        pipeline_config(), {120.0, 200.0, 310.0, 375.0, 440.0, 505.0, 570.0, 635.0, 700.0, 840.0},
        [](std::uint64_t frame) { return frame <= 5; }
    );
    for (std::uint64_t frame = 1; frame <= 10; ++frame) (void)two_view->process_frame(live_frame(frame));
    assert(wait_for([&] { return two_view->track(1) && two_view->track(1)->segmentation_results.size() == 2; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    two_view->evaluate_timeouts(std::chrono::steady_clock::now());
    const auto incomplete = two_view->track(1);
    assert(incomplete && incomplete->state == runner::TrackState::Completed);
    assert(incomplete->views_used == 2 && incomplete->incomplete_evidence);
    two_view->stop();

    auto one_view = simulated_pipeline(
        pipeline_config(), {120.0, 200.0, 310.0, 375.0, 440.0, 505.0, 570.0, 635.0, 700.0, 840.0},
        [](std::uint64_t frame) { return frame <= 5; },
        [](const runner::SegmentationJobV2& job) {
            runner::SegmentationResult result;
            result.predicted_class = 0;
            result.confidence = 0.9F;
            result.success = job.view_id == 0;
            result.message = result.success ? "" : "fixture failure";
            return result;
        }
    );
    for (std::uint64_t frame = 1; frame <= 10; ++frame) (void)one_view->process_frame(live_frame(frame));
    assert(wait_for([&] { return one_view->track(1) && one_view->track(1)->segmentation_results.size() == 2; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    one_view->evaluate_timeouts(std::chrono::steady_clock::now());
    const auto failed = one_view->track(1);
    assert(failed && failed->state == runner::TrackState::Rejected && failed->incomplete_evidence);
    one_view->stop();
}

void test_surface_mapping_disabled_preserves_legacy_api_and_grading() {
    auto config = pipeline_config();
    config.surface_mapping_enabled = false;
    assert(!config.to_json().at("surface_mapping").at("enabled").get<bool>());
    auto pipeline = simulated_pipeline(config, {120.0, 200.0, 310.0, 375.0, 440.0, 505.0, 570.0, 635.0, 700.0, 840.0});
    for (std::uint64_t frame = 1; frame <= 10; ++frame) (void)pipeline->process_frame(live_frame(frame));
    assert(wait_for([&] { return pipeline->track(1) && pipeline->track(1)->state == runner::TrackState::Completed; }));
    const auto record = pipeline->track(1)->to_json();
    assert(record.at("finalClass") == "Class 3");
    assert(record.at("surfaceMapping").is_null());
    assert(record.contains("grade") && record.contains("incompleteEvidence"));
    pipeline->stop();
}

void test_dense_track_views_route_to_background_surface_mapper() {
    auto config = pipeline_config();
    config.surface_mapping_enabled = true;
    config.minimum_capture_frame_gap = 1;
    config.surface_candidate_frame_stride = 1;
    config.surface_minimum_mapping_views = 3;
    config.surface_maximum_mapping_views = 6;
    config.surface_maximum_candidate_views = 6;
    std::atomic<int> mapping_calls{0};
    auto pipeline = simulated_pipeline(
        config,
        {120.0, 200.0, 310.0, 375.0, 440.0, 505.0, 570.0, 635.0, 700.0, 840.0},
        [](std::uint64_t) { return true; },
        {},
        [&](const runner::json& request) {
            ++mapping_calls;
            assert(request.at("trackId") == 1);
            assert(request.at("views").size() > 4);  // no fixed four-view completion rule
            return runner::json{
                {"status", "complete"}, {"coverage_degrees", 354.0}, {"observed_area_ratio", 0.82},
                {"accepted_views", request.at("views").size()}, {"incomplete_evidence", false},
                {"defects", runner::json::object()},
            };
        }
    );
    // Frame 8 is still above the classification boundary. Reaching the safe
    // dense-view buffer limit must nevertheless trigger background finalisation.
    for (std::uint64_t frame = 1; frame <= 8; ++frame) (void)pipeline->process_frame(live_frame(frame));
    assert(wait_for([&] { return pipeline->track(1) && pipeline->track(1)->state == runner::TrackState::Completed; }));
    const auto track = pipeline->track(1);
    assert(track && track->captured_views.size() > 4 && mapping_calls == 1);
    assert(track->surface_mapping.at("status") == "complete" && !track->incomplete_evidence);
    pipeline->stop();
}

void test_stale_track_removal_and_recorded_simulation() {
    auto config = pipeline_config();
    config.stale_track_frames = 1;
    config.track_retention_seconds = 0.0;
    auto pipeline = simulated_pipeline(config, {120.0, -1.0, -1.0});
    for (std::uint64_t frame = 1; frame <= 3; ++frame) (void)pipeline->process_frame(live_frame(frame));
    assert(!pipeline->track(1));
    pipeline->stop();
}

void test_tracking_admission_motion_and_identity_scenarios() {
    auto config = pipeline_config();
    config.tracking_diagnostics = true;

    // Tiny flashes and detections clearly outside the ROI never reach ByteTrack.
    auto invalid = scripted_pipeline(config, {{
        {0, "naartjie", 0.95F, 490.0F, 100.0F, 8.0F, 8.0F},
        {0, "naartjie", 0.95F, 10.0F, 100.0F, 120.0F, 120.0F},
    }});
    auto invalid_snapshot = invalid->process_frame(live_frame(1));
    assert(invalid_snapshot.at("tracks").empty() && !invalid_snapshot.at("trackingDiagnostics").empty());
    for (std::uint64_t frame = 2; frame <= 60; ++frame) invalid_snapshot = invalid->process_frame(live_frame(frame));
    assert(invalid_snapshot.at("tracks").empty());  // a long empty/flickering belt cannot grow application state
    invalid->stop();

    // Strict entry gating blocks a plausible mid-belt detection from creating an ID.
    config.strict_entry_gating = true;
    auto gated = scripted_pipeline(config, {{{0, "naartjie", 0.95F, 430.0F, 440.0F, 140.0F, 140.0F}}});
    assert(gated->process_frame(live_frame(1)).at("tracks").empty());
    gated->stop();
    config.strict_entry_gating = false;

    // A persistent static feature remains tentative and is eventually rejected; it never acquires a view.
    std::vector<std::vector<runner::Detection>> static_frames(14,
        {{0, "naartjie", 0.95F, 430.0F, 250.0F, 140.0F, 140.0F}});
    auto stationary = scripted_pipeline(config, static_frames);
    for (std::uint64_t frame = 1; frame <= static_frames.size(); ++frame) (void)stationary->process_frame(live_frame(frame));
    const auto static_track = stationary->track(1);
    assert(static_track && static_track->state == runner::TrackState::Rejected && !static_track->confirmed);
    assert(static_track->captured_views.empty());
    stationary->stop();

    // Wrong-direction motion is rejected before it can become an acquisition track.
    auto upward = scripted_pipeline(config, {
        {{0, "naartjie", 0.95F, 430.0F, 360.0F, 140.0F, 140.0F}},
        {{0, "naartjie", 0.95F, 430.0F, 310.0F, 140.0F, 140.0F}},
        {{0, "naartjie", 0.95F, 430.0F, 260.0F, 140.0F, 140.0F}},
    });
    for (std::uint64_t frame = 1; frame <= 3; ++frame) (void)upward->process_frame(live_frame(frame));
    const auto upward_track = upward->track(1);
    assert(upward_track && upward_track->state == runner::TrackState::Rejected && !upward_track->confirmed);
    upward->stop();

    // A one-frame valid flash gets a tentative ID but can never enter acquisition.
    auto flash = scripted_pipeline(config, {
        {{0, "naartjie", 0.95F, 430.0F, 80.0F, 140.0F, 140.0F}},
        {},
    });
    (void)flash->process_frame(live_frame(1));
    (void)flash->process_frame(live_frame(2));
    const auto flash_track = flash->track(1);
    assert(flash_track && flash_track->state == runner::TrackState::Lost && !flash_track->confirmed);
    assert(flash_track->captured_views.empty());
    flash->stop();

    // A two-frame flicker also stays below the configurable confirmation requirement.
    auto flicker = scripted_pipeline(config, {
        {{0, "naartjie", 0.95F, 430.0F, 80.0F, 140.0F, 140.0F}},
        {{0, "naartjie", 0.95F, 430.0F, 125.0F, 140.0F, 140.0F}},
        {},
    });
    for (std::uint64_t frame = 1; frame <= 3; ++frame) (void)flicker->process_frame(live_frame(frame));
    const auto flicker_track = flicker->track(1);
    assert(flicker_track && flicker_track->state == runner::TrackState::Lost && !flicker_track->confirmed);
    assert(flicker_track->captured_views.empty());
    flicker->stop();

    // Two nearby downward-moving fruits receive two stable IDs and both confirm.
    std::vector<std::vector<runner::Detection>> pair_frames;
    for (int frame = 0; frame < 4; ++frame) {
        const float y = 70.0F + frame * 55.0F;
        pair_frames.push_back({
            {0, "naartjie", 0.95F, 250.0F, y, 120.0F, 120.0F},
            {0, "naartjie", 0.95F, 560.0F, y + 8.0F, 120.0F, 120.0F},
        });
    }
    auto pair = scripted_pipeline(config, pair_frames);
    for (std::uint64_t frame = 1; frame <= pair_frames.size(); ++frame) (void)pair->process_frame(live_frame(frame));
    assert(pair->track(1) && pair->track(1)->confirmed);
    assert(pair->track(2) && pair->track(2)->confirmed);
    pair->stop();

    // A brief miss is LOST for one frame and resumes the same ID inside the tracker buffer.
    auto missed = scripted_pipeline(config, {
        {{0, "naartjie", 0.95F, 430.0F, 70.0F, 140.0F, 140.0F}},
        {{0, "naartjie", 0.95F, 430.0F, 125.0F, 140.0F, 140.0F}},
        {{0, "naartjie", 0.95F, 430.0F, 180.0F, 140.0F, 140.0F}},
        {},
        {{0, "naartjie", 0.95F, 430.0F, 245.0F, 140.0F, 140.0F}},
    });
    for (std::uint64_t frame = 1; frame <= 3; ++frame) (void)missed->process_frame(live_frame(frame));
    assert(missed->track(1) && missed->track(1)->confirmed);
    (void)missed->process_frame(live_frame(4));
    assert(missed->track(1)->state == runner::TrackState::Lost);
    (void)missed->process_frame(live_frame(5));
    assert(missed->track(1) && missed->track(1)->track_id == 1 && missed->track(1)->state != runner::TrackState::Lost);
    missed->stop();
}

}  // namespace

int main() {
    test_precision_flags();
    test_direct_artifact_contract_and_delegated_result();
    test_deterministic_mask_decode();
    test_bytetrack_and_grade_aggregation();
    test_pipeline_serializes_model_callbacks();
    test_zone_progress_capture_and_crop_math();
    test_worst_confident_class_aggregation();
    test_routing_four_view_classification_and_no_duplicate();
    test_prediction_only_frames_fill_all_four_capture_targets();
    test_two_view_timeout_and_insufficient_evidence();
    test_surface_mapping_disabled_preserves_legacy_api_and_grading();
    test_dense_track_views_route_to_background_surface_mapper();
    test_stale_track_removal_and_recorded_simulation();
    test_tracking_admission_motion_and_identity_scenarios();
    std::cout << "runner tests passed\n";
    return 0;
}
