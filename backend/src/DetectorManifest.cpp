#include "runner/DetectorManifest.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace runner {
namespace {

template <typename Container>
bool contains(const Container& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::array<float, 3> triplet(const nlohmann::json& value, const char* name) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(std::string("input.") + name + " must contain exactly three numbers");
    }
    std::array<float, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = value.at(index).get<float>();
        if (!std::isfinite(result[index])) throw std::invalid_argument(std::string("input.") + name + " must be finite");
    }
    return result;
}

const DetectorOutputSpec& output_for_role(const DetectorManifest& manifest, const std::string& role) {
    const auto found = std::find_if(manifest.outputs.begin(), manifest.outputs.end(), [&](const DetectorOutputSpec& output) {
        return output.role == role;
    });
    if (found == manifest.outputs.end()) throw std::invalid_argument("output schema requires an output with role '" + role + "'");
    return *found;
}

void require_rank(const DetectorOutputSpec& output, std::size_t rank) {
    if (output.shape.size() != rank) {
        throw std::invalid_argument("output '" + output.name + "' must have rank " + std::to_string(rank));
    }
}

}  // namespace

DetectorManifest DetectorManifest::from_json(const nlohmann::json& value) {
    if (!value.is_object()) throw std::invalid_argument("model manifest must be a JSON object");
    DetectorManifest manifest;
    manifest.schema_version = value.value("schemaVersion", 4);
    manifest.id = value.value("id", "");
    manifest.name = value.value("name", "");
    manifest.task = value.value("task", "");
    manifest.model_family = value.value("modelFamily", "");
    manifest.backend = value.value("backend", "");
    manifest.model_file = value.value("modelFile", "");
    manifest.output_schema = value.value("outputSchema", "");
    manifest.tensor_orientation = value.value("tensorOrientation", "");
    manifest.box_format = value.value("boxFormat", "");
    manifest.coordinates = value.value("coordinates", "");
    manifest.nms = value.value("nms", "");
    manifest.nms_iou = value.value("nmsIou", 0.5F);
    manifest.has_objectness = value.value("hasObjectness", false);
    manifest.score_floor = value.value("scoreFloor", 0.10F);
    manifest.labels = value.value("labels", std::vector<std::string>{});
    manifest.threads = value.contains("runtime") ? value.at("runtime").value("threads", 4) : 4;

    if (!value.contains("input") || !value.at("input").is_object()) throw std::invalid_argument("input configuration is required");
    const auto& input = value.at("input");
    manifest.input.name = input.value("name", "");
    manifest.input.width = input.value("width", 0);
    manifest.input.height = input.value("height", 0);
    manifest.input.layout = input.value("layout", "");
    manifest.input.colour_order = input.value("colourOrder", "");
    manifest.input.data_type = input.value("dataType", "");
    manifest.input.mean = triplet(input.value("mean", nlohmann::json::array()), "mean");
    manifest.input.scale = triplet(input.value("scale", nlohmann::json::array()), "scale");
    manifest.input.resize_mode = input.value("resizeMode", "");
    manifest.input.letterbox_value = input.value("letterboxValue", 114.0F);

    if (!value.contains("outputs") || !value.at("outputs").is_array()) throw std::invalid_argument("outputs must be an array");
    for (const auto& item : value.at("outputs")) {
        if (!item.is_object()) throw std::invalid_argument("each output must describe its name, role, shape, and dataType");
        DetectorOutputSpec output;
        output.name = item.value("name", "");
        output.role = item.value("role", "");
        output.shape = item.value("shape", std::vector<int>{});
        output.data_type = item.value("dataType", "");
        if (item.contains("quantization")) {
            output.quantization_scale = item.at("quantization").value("scale", 0.0F);
            output.quantization_zero_point = item.at("quantization").value("zeroPoint", 0);
        }
        manifest.outputs.push_back(std::move(output));
    }

    if (manifest.schema_version != 4) throw std::invalid_argument("object-detection manifests require schemaVersion 4");
    if (manifest.name.empty() || manifest.name.size() > 100) throw std::invalid_argument("name must contain between 1 and 100 characters");
    if (manifest.task != "object-detection") throw std::invalid_argument("task must be object-detection");
    if (!contains(std::array<std::string, 2>{"yolo", "dfine"}, manifest.model_family)) {
        throw std::invalid_argument("modelFamily must be yolo or dfine");
    }
    if (!contains(std::array<std::string, 2>{"tflite", "tensorrt"}, manifest.backend)) {
        throw std::invalid_argument("backend must be tflite or tensorrt");
    }
    if (manifest.model_file.empty() || std::filesystem::path(manifest.model_file).is_absolute() ||
        manifest.model_file.find("..") != std::string::npos) {
        throw std::invalid_argument("modelFile must be a relative file name inside the model package");
    }
    if (manifest.input.name.empty()) throw std::invalid_argument("input.name is required; tensor names are never guessed");
    if (manifest.input.width < 32 || manifest.input.width > 4096 || manifest.input.height < 32 || manifest.input.height > 4096) {
        throw std::invalid_argument("input width and height must be between 32 and 4096");
    }
    if (!contains(std::array<std::string, 2>{"NHWC", "NCHW"}, manifest.input.layout)) {
        throw std::invalid_argument("input.layout must be NHWC or NCHW");
    }
    if (!contains(std::array<std::string, 2>{"RGB", "BGR"}, manifest.input.colour_order)) {
        throw std::invalid_argument("input.colourOrder must be RGB or BGR");
    }
    if (!contains(std::array<std::string, 4>{"float32", "float16", "uint8", "int8"}, manifest.input.data_type)) {
        throw std::invalid_argument("input.dataType must be float32, float16, uint8, or int8");
    }
    if (!contains(std::array<std::string, 2>{"stretch", "letterbox"}, manifest.input.resize_mode)) {
        throw std::invalid_argument("input.resizeMode must be stretch or letterbox");
    }
    if (!std::isfinite(manifest.input.letterbox_value) || manifest.input.letterbox_value < 0.0F || manifest.input.letterbox_value > 255.0F) {
        throw std::invalid_argument("input.letterboxValue must be between 0 and 255");
    }
    if (std::any_of(manifest.input.scale.begin(), manifest.input.scale.end(), [](float scale) { return scale == 0.0F; })) {
        throw std::invalid_argument("input.scale values must be non-zero");
    }
    if (manifest.outputs.empty()) throw std::invalid_argument("at least one output tensor is required");
    std::set<std::string> names;
    const std::array<std::string, 6> output_types{"float32", "float16", "uint8", "int8", "int32", "int64"};
    for (const auto& output : manifest.outputs) {
        if (output.name.empty() || output.role.empty() || output.shape.empty()) {
            throw std::invalid_argument("every output requires name, role, shape, and dataType");
        }
        if (!names.insert(output.name).second) throw std::invalid_argument("output tensor names must be unique");
        if (!contains(output_types, output.data_type)) throw std::invalid_argument("output '" + output.name + "' has an unsupported dataType");
        if (std::any_of(output.shape.begin(), output.shape.end(), [](int dimension) { return dimension == 0 || dimension < -1; })) {
            throw std::invalid_argument("output shapes may contain positive dimensions or -1 wildcards only");
        }
        if ((output.data_type == "uint8" || output.data_type == "int8") && output.quantization_scale < 0.0F) {
            throw std::invalid_argument("quantized output scale cannot be negative");
        }
    }
    if (!contains(std::array<std::string, 3>{"cxcywh", "xyxy", "xywh"}, manifest.box_format)) {
        throw std::invalid_argument("boxFormat must be cxcywh, xyxy, or xywh");
    }
    if (!contains(std::array<std::string, 3>{"normalized", "input-pixels", "original-pixels"}, manifest.coordinates)) {
        throw std::invalid_argument("coordinates must be normalized, input-pixels, or original-pixels");
    }
    if (!contains(std::array<std::string, 2>{"none", "class-aware"}, manifest.nms)) {
        throw std::invalid_argument("nms must be none or class-aware");
    }
    if (!(manifest.nms_iou > 0.0F && manifest.nms_iou < 1.0F)) throw std::invalid_argument("nmsIou must be between 0 and 1");
    if (!(manifest.score_floor >= 0.0F && manifest.score_floor <= 0.10F)) {
        throw std::invalid_argument("scoreFloor must preserve detections down to at least 0.10 for ByteTrack");
    }
    if (manifest.labels != std::vector<std::string>{"Naartjie"}) {
        throw std::invalid_argument("the whole-fruit detector must use the single label 'Naartjie'");
    }
    if (manifest.backend == "tflite" && (manifest.threads < 1 || manifest.threads > 4)) {
        throw std::invalid_argument("TFLite runtime.threads must be between 1 and 4");
    }
    if (manifest.backend == "tflite" && manifest.input.data_type == "float16") {
        throw std::invalid_argument("TFLite detector inputs support float32, uint8, or int8; float16 input is unsupported");
    }
    if (manifest.backend == "tensorrt" && manifest.threads < 1) throw std::invalid_argument("runtime.threads must be positive");

    if (manifest.model_family == "yolo") {
        if (manifest.output_schema == "yolo-raw") {
            const auto& predictions = output_for_role(manifest, "predictions");
            require_rank(predictions, 3);
            if (!contains(std::array<std::string, 2>{"channels-first", "predictions-first"}, manifest.tensor_orientation)) {
                throw std::invalid_argument("yolo-raw requires tensorOrientation; ambiguous tensor layouts are never guessed");
            }
        } else if (manifest.output_schema == "yolo-end2end") {
            const auto& detections = output_for_role(manifest, "detections");
            require_rank(detections, 3);
            if (detections.shape.back() != -1 && detections.shape.back() != 6) {
                throw std::invalid_argument("yolo-end2end detections must contain [x1,y1,x2,y2,score,class]");
            }
        } else {
            throw std::invalid_argument("YOLO outputSchema must be yolo-raw or yolo-end2end");
        }
    } else {
        if (manifest.nms != "none") throw std::invalid_argument("D-FINE set-based exports require nms: none");
        if (manifest.output_schema == "dfine-raw") {
            require_rank(output_for_role(manifest, "logits"), 3);
            require_rank(output_for_role(manifest, "boxes"), 3);
        } else if (manifest.output_schema == "dfine-fused") {
            require_rank(output_for_role(manifest, "labels"), 2);
            require_rank(output_for_role(manifest, "boxes"), 3);
            require_rank(output_for_role(manifest, "scores"), 2);
        } else {
            throw std::invalid_argument("D-FINE outputSchema must be dfine-raw or dfine-fused");
        }
    }
    validate_detector_model_extension(manifest, manifest.model_file);
    return manifest;
}

nlohmann::json DetectorManifest::to_json() const {
    nlohmann::json output_values = nlohmann::json::array();
    for (const auto& output : outputs) {
        nlohmann::json value = {{"name", output.name}, {"role", output.role}, {"shape", output.shape}, {"dataType", output.data_type}};
        if (output.quantization_scale > 0.0F) value["quantization"] = {{"scale", output.quantization_scale}, {"zeroPoint", output.quantization_zero_point}};
        output_values.push_back(std::move(value));
    }
    return {
        {"schemaVersion", schema_version}, {"id", id}, {"name", name}, {"task", task},
        {"modelFamily", model_family}, {"backend", backend}, {"modelFile", model_file},
        {"input", {{"name", input.name}, {"width", input.width}, {"height", input.height},
                   {"layout", input.layout}, {"colourOrder", input.colour_order}, {"dataType", input.data_type},
                   {"mean", input.mean}, {"scale", input.scale}, {"resizeMode", input.resize_mode},
                   {"letterboxValue", input.letterbox_value}}},
        {"outputs", std::move(output_values)}, {"outputSchema", output_schema},
        {"tensorOrientation", tensor_orientation}, {"boxFormat", box_format}, {"coordinates", coordinates},
        {"nms", nms}, {"nmsIou", nms_iou}, {"hasObjectness", has_objectness},
        {"scoreFloor", score_floor}, {"labels", labels}, {"runtime", {{"threads", threads}}},
    };
}

void validate_detector_model_extension(const DetectorManifest& manifest, const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    const std::string expected = manifest.backend == "tflite" ? ".tflite" : ".engine";
    if (extension != expected) throw std::invalid_argument(manifest.backend + " models require a " + expected + " modelFile");
}

}  // namespace runner
