#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runner/ByteTrack.hpp"
#include "runner/DetectorDecoder.hpp"
#include "runner/DetectorManifest.hpp"
#include "runner/TensorUtils.hpp"
#include "runner/TrackingPipeline.hpp"
#include "runner/Types.hpp"

namespace {

using runner::json;

json base_manifest(const std::string& family, const std::string& backend) {
    json value = {
        {"schemaVersion", 4}, {"id", family + "-" + backend}, {"name", "Naartjie detector"},
        {"task", "object-detection"}, {"modelFamily", family}, {"backend", backend},
        {"modelFile", backend == "tflite" ? "model.tflite" : "model.engine"},
        {"input", {{"name", "images"}, {"width", 640}, {"height", 640}, {"layout", backend == "tflite" ? "NHWC" : "NCHW"},
                   {"colourOrder", "RGB"}, {"dataType", "float32"}, {"mean", {0.0, 0.0, 0.0}},
                   {"scale", {1.0 / 255, 1.0 / 255, 1.0 / 255}}, {"resizeMode", "letterbox"}, {"letterboxValue", 114}}},
        {"boxFormat", "cxcywh"}, {"coordinates", "input-pixels"}, {"nms", family == "dfine" ? "none" : "class-aware"},
        {"nmsIou", 0.5}, {"scoreFloor", 0.1}, {"labels", {"Naartjie"}}, {"runtime", {{"threads", 4}}},
    };
    if (family == "yolo") {
        if (backend == "tflite") {
            value["outputs"] = {{{"name", "predictions"}, {"role", "predictions"}, {"shape", {1, 5, 4}}, {"dataType", "float32"}}};
            value["outputSchema"] = "yolo-raw";
            value["tensorOrientation"] = "channels-first";
        } else {
            value["outputs"] = {{{"name", "output0"}, {"role", "detections"}, {"shape", {1, 300, 6}}, {"dataType", "float32"}}};
            value["outputSchema"] = "yolo-end2end";
            value["boxFormat"] = "xyxy";
            value["nms"] = "none";
        }
    } else if (backend == "tflite") {
        value["outputs"] = {
            {{"name", "logits"}, {"role", "logits"}, {"shape", {1, 300, 1}}, {"dataType", "float32"}},
            {{"name", "boxes"}, {"role", "boxes"}, {"shape", {1, 300, 4}}, {"dataType", "float32"}},
        };
        value["outputSchema"] = "dfine-raw";
        value["coordinates"] = "normalized";
    } else {
        value["outputs"] = {
            {{"name", "labels"}, {"role", "labels"}, {"shape", {1, 300}}, {"dataType", "int32"}},
            {{"name", "boxes"}, {"role", "boxes"}, {"shape", {1, 300, 4}}, {"dataType", "float32"}},
            {{"name", "scores"}, {"role", "scores"}, {"shape", {1, 300}}, {"dataType", "float32"}},
        };
        value["outputSchema"] = "dfine-fused";
        value["boxFormat"] = "xyxy";
    }
    return value;
}

void expect_manifest_invalid(json value) {
    bool thrown = false;
    try { (void)runner::DetectorManifest::from_json(value); } catch (const std::invalid_argument&) { thrown = true; }
    assert(thrown);
}

runner::ImageTransform transform(int source_width = 640, int source_height = 640) {
    return {source_width, source_height, 640, 640, 1.0F, 1.0F, 0.0F, 0.0F, false};
}

void test_manifest_validation() {
    assert(runner::DetectorManifest::from_json(base_manifest("yolo", "tflite")).threads == 4);
    assert(runner::DetectorManifest::from_json(base_manifest("dfine", "tflite")).output_schema == "dfine-raw");
    assert(runner::DetectorManifest::from_json(base_manifest("yolo", "tensorrt")).output_schema == "yolo-end2end");
    assert(runner::DetectorManifest::from_json(base_manifest("dfine", "tensorrt")).output_schema == "dfine-fused");

    auto unsupported_backend = base_manifest("yolo", "tflite");
    unsupported_backend["backend"] = "ncnn";
    expect_manifest_invalid(unsupported_backend);
    auto unsupported_family = base_manifest("yolo", "tflite");
    unsupported_family["modelFamily"] = "detr";
    expect_manifest_invalid(unsupported_family);
    auto wrong_extension = base_manifest("yolo", "tflite");
    wrong_extension["modelFile"] = "model.engine";
    expect_manifest_invalid(wrong_extension);
    auto ambiguous = base_manifest("yolo", "tflite");
    ambiguous.erase("tensorOrientation");
    ambiguous["outputs"][0]["shape"] = {1, 5, 5};
    expect_manifest_invalid(ambiguous);
    auto invalid_preprocess = base_manifest("yolo", "tflite");
    invalid_preprocess["input"]["scale"] = {0.0, 0.0, 0.0};
    expect_manifest_invalid(invalid_preprocess);
}

void test_yolo_decoding() {
    auto value = base_manifest("yolo", "tflite");
    value["outputs"][0]["shape"] = {1, 5, 3};
    auto manifest = runner::DetectorManifest::from_json(value);
    // channels-first: cx, cy, w, h, class score.
    runner::TensorData channels_first{"predictions", {1, 5, 3}, {
        100, 105, 400,
        100, 105, 400,
        40, 40, 20,
        40, 40, 20,
        0.90F, 0.80F, 0.09F,
    }};
    auto decoded = runner::decode_detector_outputs(manifest, {channels_first}, transform(), 0.10F);
    assert(decoded.size() == 1);  // overlapping first/second boxes are suppressed by NMS
    assert(std::abs(decoded.front().x - 80.0F) < 0.001F && decoded.front().label == "Naartjie");

    value["tensorOrientation"] = "predictions-first";
    value["outputs"][0]["shape"] = {1, -1, 5};
    manifest = runner::DetectorManifest::from_json(value);
    runner::TensorData predictions_first{"predictions", {1, 2, 5}, {100, 100, 40, 40, 0.9F, 400, 400, 20, 20, 0.09F}};
    decoded = runner::decode_detector_outputs(manifest, {predictions_first}, transform(), 0.10F);
    assert(decoded.size() == 1 && std::abs(decoded.front().width - 40.0F) < 0.001F);

    // Reverse a 640-square letterbox into a 1280x720 source: scale 0.5, pad y 140.
    const runner::ImageTransform letterbox{1280, 720, 640, 640, 0.5F, 0.5F, 0.0F, 140.0F, true};
    runner::TensorData one{"predictions", {1, 1, 5}, {320, 320, 100, 100, 0.8F}};
    decoded = runner::decode_detector_outputs(manifest, {one}, letterbox, 0.10F);
    assert(decoded.size() == 1 && std::abs(decoded[0].x - 540.0F) < 0.001F && std::abs(decoded[0].y - 260.0F) < 0.001F);

    runner::TensorData empty{"predictions", {1, 1, 5}, {320, 320, 100, 100, 0.09F}};
    assert(runner::decode_detector_outputs(manifest, {empty}, letterbox, 0.10F).empty());
    bool malformed = false;
    try { (void)runner::decode_detector_outputs(manifest, {{"predictions", {1, 1, 6}, {0, 0, 0, 0, 0, 0}}}, transform(), 0.10F); }
    catch (const std::invalid_argument&) { malformed = true; }
    assert(malformed);
}

void test_dfine_decoding() {
    auto value = base_manifest("dfine", "tflite");
    value["outputs"][0]["shape"] = {1, 2, 1};
    value["outputs"][1]["shape"] = {1, 2, 4};
    const auto raw = runner::DetectorManifest::from_json(value);
    // sigmoid(-2)=0.119: the low-confidence value must survive the 0.10 floor.
    const runner::TensorData logits{"logits", {1, 2, 1}, {-2.0F, -3.0F}};
    const runner::TensorData boxes{"boxes", {1, 2, 4}, {0.5F, 0.5F, 0.25F, 0.25F, 0.2F, 0.2F, 0.1F, 0.1F}};
    auto decoded = runner::decode_detector_outputs(raw, {logits, boxes}, transform(), 0.10F);
    assert(decoded.size() == 1 && decoded[0].confidence > 0.11F && decoded[0].confidence < 0.13F);
    assert(std::abs(decoded[0].x - 240.0F) < 0.001F && std::abs(decoded[0].width - 160.0F) < 0.001F);

    value = base_manifest("dfine", "tensorrt");
    value["outputs"][0]["shape"] = {1, -1};
    value["outputs"][1]["shape"] = {1, -1, 4};
    value["outputs"][2]["shape"] = {1, -1};
    const auto fused = runner::DetectorManifest::from_json(value);
    const runner::TensorData labels{"labels", {1, 2}, {0, 0}};
    const runner::TensorData fused_boxes{"boxes", {1, 2, 4}, {10, 20, 50, 70, 12, 22, 48, 68}};
    const runner::TensorData scores{"scores", {1, 2}, {0.8F, 0.7F}};
    decoded = runner::decode_detector_outputs(fused, {labels, fused_boxes, scores}, transform(), 0.10F);
    assert(decoded.size() == 2);  // D-FINE does not receive automatic YOLO NMS.
    assert(decoded[0].x == 10.0F && decoded[0].height == 50.0F);

    assert(runner::decode_detector_outputs(fused,
        {runner::TensorData{"labels", {1, 0}, {}}, runner::TensorData{"boxes", {1, 0, 4}, {}}, runner::TensorData{"scores", {1, 0}, {}}},
        transform(), 0.10F).empty());
    bool malformed = false;
    try { (void)runner::decode_detector_outputs(fused, {labels, fused_boxes}, transform(), 0.10F); }
    catch (const std::invalid_argument&) { malformed = true; }
    assert(malformed);
}

void test_tensor_quantization() {
    assert(std::abs(runner::dequantize_value(runner::quantize_uint8(0.5F, 0.01F, 128), 0.01F, 128) - 0.5F) < 0.011F);
    assert(std::abs(runner::dequantize_value(runner::quantize_int8(-0.5F, 0.01F, 0), 0.01F, 0) + 0.5F) < 0.011F);
    assert(std::abs(runner::half_to_float(runner::float_to_half(1.25F)) - 1.25F) < 0.001F);
    assert(runner::tensor_shape_matches({1, 300, 6}, {1, -1, 6}));
    assert(!runner::tensor_shape_matches({1, 300, 7}, {1, -1, 6}));
}

runner::ByteTrackDetection track_detection(float x, float y, float score, int index = 0) {
    return {{x, y, x + 20.0F, y + 20.0F}, score, 0, index};
}

void test_byte_track() {
    runner::ByteTrack tracker;
    assert(tracker.update({track_detection(10, 10, 0.9F)}, 200, 200).empty());
    auto visible = tracker.update({track_detection(11, 15, 0.9F)}, 200, 200);
    assert(visible.size() == 1 && visible[0].track_id == 1);
    visible = tracker.update({track_detection(12, 20, 0.3F)}, 200, 200);
    assert(visible.size() == 1 && visible[0].track_id == 1);  // low-confidence second stage

    runner::ByteTrack low_only;
    assert(low_only.update({track_detection(10, 10, 0.3F)}, 200, 200).empty());
    assert(low_only.active_track_count() == 0);

    assert(tracker.update({}, 200, 200).empty());
    assert(tracker.lost_track_count() == 1);  // internally buffered, never rendered
    visible = tracker.update({track_detection(14, 30, 0.9F)}, 200, 200);
    assert(visible.size() == 1 && visible[0].track_id == 1);
    for (int count = 0; count < 4; ++count) assert(tracker.update({}, 200, 200).empty());
    assert(tracker.lost_track_count() == 0);
    assert(tracker.update({track_detection(20, 50, 0.9F)}, 200, 200).empty());
    visible = tracker.update({track_detection(21, 55, 0.9F)}, 200, 200);
    assert(visible.size() == 1 && visible[0].track_id == 2);  // removed ID is not reused

    runner::ByteTrack two;
    two.update({track_detection(20, 10, 0.9F, 0), track_detection(100, 10, 0.9F, 1)}, 200, 200);
    visible = two.update({track_detection(22, 16, 0.9F, 0), track_detection(98, 16, 0.9F, 1)}, 200, 200);
    assert(visible.size() == 2 && visible[0].track_id != visible[1].track_id);
    assert(two.update({}, 200, 200).empty());
}

runner::InferenceResult detector_result(const std::string& id, const std::string& family, float x, float y, float score = 0.9F) {
    runner::InferenceResult result;
    result.model_id = id;
    result.model_family = family;
    result.backend = "tflite";
    result.task = runner::ModelTask::ObjectDetection;
    result.source_width = 200;
    result.source_height = 200;
    result.detections.push_back({0, "Naartjie", score, x, y, 20, 20});
    result.timing.backend_total_ms = 4.0;
    return result;
}

void test_tracking_pipeline() {
    runner::ByteTrackConfig tracker_config;
    tracker_config.tracking_thresh = 0.80F;
    runner::TrackingPipeline pipeline("session", "model", "yolo", "tflite", {0.25F, 0.0F, 0.5F, 1.0F}, tracker_config);
    assert(pipeline.process(1, 1.0, detector_result("model", "yolo", 5, 10)).tracks.empty());  // outside ROI
    assert(pipeline.process(2, 2.0, detector_result("model", "yolo", 60, 20)).tracks.empty());
    auto result = pipeline.process(3, 3.0, detector_result("model", "yolo", 61, 40));
    assert(result.tracks.size() == 1 && result.tracks[0].track_id == 1);
    assert(std::abs(result.tracks[0].progress - 0.25F) < 0.001F);
    const auto before = pipeline.status();
    bool stale = false;
    try { (void)pipeline.process(2, 2.0, detector_result("model", "yolo", 62, 45)); }
    catch (const std::invalid_argument&) { stale = true; }
    assert(stale && pipeline.status().at("lastFrameId") == before.at("lastFrameId"));

    pipeline.reset();
    assert(pipeline.status().at("lastFrameId").is_null());
    pipeline.set_model("dfine-model", "dfine", "tflite");
    assert(pipeline.process(1, 1.0, detector_result("dfine-model", "dfine", 60, 20)).tracks.empty());
    result = pipeline.process(2, 2.0, detector_result("dfine-model", "dfine", 61, 30));
    assert(result.tracks.size() == 1 && result.tracks[0].track_id == 1);  // switch reset
}

}  // namespace

int main() {
    test_manifest_validation();
    test_yolo_decoding();
    test_dfine_decoding();
    test_tensor_quantization();
    test_byte_track();
    test_tracking_pipeline();
    std::cout << "Detector/tracking tests passed\n";
    return 0;
}
