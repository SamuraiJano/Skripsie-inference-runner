#include "runner/TfliteEngine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runner/DetectorDecoder.hpp"
#include "runner/DetectorManifest.hpp"
#include "runner/ImagePreprocess.hpp"
#include "runner/TensorUtils.hpp"

#if RUNNER_WITH_TFLITE
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>
#if RUNNER_TFLITE_XNNPACK
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#endif
#endif

namespace runner {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#if RUNNER_WITH_TFLITE

std::string tflite_type_name(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32: return "float32";
        case kTfLiteUInt8: return "uint8";
        case kTfLiteInt8: return "int8";
        case kTfLiteInt32: return "int32";
        case kTfLiteInt64: return "int64";
        default: return "unsupported";
    }
}

std::vector<int> tensor_shape(const TfLiteTensor& tensor) {
    std::vector<int> shape;
    if (!tensor.dims) return shape;
    shape.reserve(static_cast<std::size_t>(tensor.dims->size));
    for (int index = 0; index < tensor.dims->size; ++index) shape.push_back(tensor.dims->data[index]);
    return shape;
}

const DetectorOutputSpec& output_spec(const DetectorManifest& manifest, const std::string& name) {
    const auto found = std::find_if(manifest.outputs.begin(), manifest.outputs.end(), [&](const DetectorOutputSpec& value) {
        return value.name == name;
    });
    if (found == manifest.outputs.end()) throw std::runtime_error("TFLite returned undeclared output tensor '" + name + "'");
    return *found;
}

#if RUNNER_TFLITE_XNNPACK
struct XnnpackDelete {
    void operator()(TfLiteDelegate* delegate) const { if (delegate) TfLiteXNNPackDelegateDelete(delegate); }
};
#endif

class TfliteEngine final : public InferenceEngine {
public:
    TfliteEngine(ModelConfig config, const fs::path& path)
        : config_(std::move(config)), manifest_(DetectorManifest::from_json(config_.detector_manifest)) {
        model_ = tflite::FlatBufferModel::BuildFromFile(path.string().c_str());
        if (!model_) throw std::runtime_error("could not load TFLite model '" + path.string() + "'");
        tflite::ops::builtin::BuiltinOpResolver resolver;
        if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk || !interpreter_) {
            throw std::runtime_error("could not create the TFLite interpreter");
        }
        interpreter_->SetNumThreads(manifest_.threads);
#if RUNNER_TFLITE_XNNPACK
        TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
        options.num_threads = manifest_.threads;
        delegate_.reset(TfLiteXNNPackDelegateCreate(&options));
        if (!delegate_ || interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
            throw std::runtime_error("TFLite XNNPACK delegate initialization failed");
        }
#endif
        if (interpreter_->AllocateTensors() != kTfLiteOk) throw std::runtime_error("TFLite tensor allocation failed");
        validate_contract();
    }

    InferenceResult infer(
        const std::vector<unsigned char>& encoded_image,
        float confidence,
        float,
        const GroundingRequest&
    ) override {
        const auto total_start = Clock::now();
        const auto preprocess_start = total_start;
        PreparedImage prepared = prepare_detector_image(encoded_image, manifest_);
        TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs().front());
        const float quant_scale = input->params.scale;
        const int zero_point = input->params.zero_point;
        if (input->type == kTfLiteFloat32) {
            std::copy(prepared.values.begin(), prepared.values.end(), interpreter_->typed_input_tensor<float>(0));
        } else if (input->type == kTfLiteUInt8) {
            auto* destination = interpreter_->typed_input_tensor<std::uint8_t>(0);
            for (std::size_t index = 0; index < prepared.values.size(); ++index) destination[index] = quantize_uint8(prepared.values[index], quant_scale, zero_point);
        } else if (input->type == kTfLiteInt8) {
            auto* destination = interpreter_->typed_input_tensor<std::int8_t>(0);
            for (std::size_t index = 0; index < prepared.values.size(); ++index) destination[index] = quantize_int8(prepared.values[index], quant_scale, zero_point);
        } else {
            throw std::runtime_error("unsupported TFLite input type after model validation");
        }
        const auto preprocess_end = Clock::now();
        const auto inference_start = preprocess_end;
        if (interpreter_->Invoke() != kTfLiteOk) throw std::runtime_error("TFLite inference failed");
        const auto inference_end = Clock::now();

        std::vector<TensorData> output_values;
        output_values.reserve(interpreter_->outputs().size());
        for (int tensor_index : interpreter_->outputs()) {
            const TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
            TensorData value;
            value.name = tensor->name ? tensor->name : "";
            value.shape = tensor_shape(*tensor);
            const std::size_t count = tensor_element_count(value.shape);
            value.values.resize(count);
            if (tensor->type == kTfLiteFloat32) {
                std::copy_n(tensor->data.f, count, value.values.begin());
            } else if (tensor->type == kTfLiteUInt8) {
                for (std::size_t index = 0; index < count; ++index) value.values[index] = dequantize_value(tensor->data.uint8[index], tensor->params.scale, tensor->params.zero_point);
            } else if (tensor->type == kTfLiteInt8) {
                for (std::size_t index = 0; index < count; ++index) value.values[index] = dequantize_value(tensor->data.int8[index], tensor->params.scale, tensor->params.zero_point);
            } else if (tensor->type == kTfLiteInt32) {
                for (std::size_t index = 0; index < count; ++index) value.values[index] = static_cast<float>(tensor->data.i32[index]);
            } else if (tensor->type == kTfLiteInt64) {
                for (std::size_t index = 0; index < count; ++index) value.values[index] = static_cast<float>(tensor->data.i64[index]);
            }
            output_values.push_back(std::move(value));
        }
        const auto postprocess_start = Clock::now();
        InferenceResult result;
        result.model_id = config_.id;
        result.model_family = manifest_.model_family;
        result.backend = manifest_.backend;
        result.task = ModelTask::ObjectDetection;
        result.source_width = prepared.source_width;
        result.source_height = prepared.source_height;
        result.detections = decode_detector_outputs(manifest_, output_values, prepared.transform,
            std::clamp(confidence, manifest_.score_floor, 1.0F));
        const auto end = Clock::now();
        result.timing.preprocess_ms = milliseconds(preprocess_start, preprocess_end);
        result.timing.inference_ms = milliseconds(inference_start, inference_end);
        result.timing.detection_nms_ms = milliseconds(postprocess_start, end);
        result.timing.backend_total_ms = milliseconds(total_start, end);
        return result;
    }

private:
    void validate_contract() {
        if (interpreter_->inputs().size() != 1) throw std::runtime_error("TFLite detector must expose exactly one input tensor");
        const TfLiteTensor* input = interpreter_->tensor(interpreter_->inputs().front());
        const std::string input_name = input->name ? input->name : "";
        if (input_name != manifest_.input.name) throw std::runtime_error("TFLite input name '" + input_name + "' does not match manifest input.name");
        if (tflite_type_name(input->type) != manifest_.input.data_type) throw std::runtime_error("TFLite input data type does not match the manifest");
        const std::vector<int> expected_input = manifest_.input.layout == "NCHW"
            ? std::vector<int>{1, 3, manifest_.input.height, manifest_.input.width}
            : std::vector<int>{1, manifest_.input.height, manifest_.input.width, 3};
        if (tensor_shape(*input) != expected_input) throw std::runtime_error("TFLite input shape does not match dimensions/layout in the manifest");
        if ((input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) && !(input->params.scale > 0.0F)) {
            throw std::runtime_error("quantized TFLite input has invalid runtime scale metadata");
        }
        if (interpreter_->outputs().size() != manifest_.outputs.size()) throw std::runtime_error("TFLite output count does not match the manifest");
        for (int index : interpreter_->outputs()) {
            const TfLiteTensor* output = interpreter_->tensor(index);
            const std::string name = output->name ? output->name : "";
            const auto& spec = output_spec(manifest_, name);
            if (tflite_type_name(output->type) != spec.data_type) throw std::runtime_error("TFLite output '" + name + "' data type does not match the manifest");
            if (!tensor_shape_matches(tensor_shape(*output), spec.shape)) throw std::runtime_error("TFLite output '" + name + "' shape does not match the manifest");
            if ((output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) && !(output->params.scale > 0.0F)) {
                throw std::runtime_error("quantized TFLite output '" + name + "' has invalid runtime scale metadata");
            }
        }
    }

    ModelConfig config_;
    DetectorManifest manifest_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
#if RUNNER_TFLITE_XNNPACK
    std::unique_ptr<TfLiteDelegate, XnnpackDelete> delegate_;
#endif
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

#endif

}  // namespace

std::unique_ptr<InferenceEngine> create_tflite_engine(const ModelConfig& config, const fs::path& model_path) {
#if RUNNER_WITH_TFLITE
    return std::make_unique<TfliteEngine>(config, model_path);
#else
    (void)config;
    (void)model_path;
    throw std::runtime_error("this binary was built without TFLite/LiteRT support");
#endif
}

bool tflite_available() {
#if RUNNER_WITH_TFLITE
    return true;
#else
    return false;
#endif
}

}  // namespace runner
