#include "runner/TensorRtEngine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runner/DetectorDecoder.hpp"
#include "runner/DetectorManifest.hpp"
#include "runner/ImagePreprocess.hpp"
#include "runner/TensorUtils.hpp"

#if RUNNER_WITH_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace runner {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#if RUNNER_WITH_TENSORRT

void cuda_check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) last_error_ = message ? message : "TensorRT error";
    }
    const std::string& last_error() const { return last_error_; }
private:
    std::string last_error_;
};

std::vector<unsigned char> read_engine_plan(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open TensorRT engine '" + path.string() + "'");
    std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (bytes.size() < 4) throw std::runtime_error("TensorRT engine file is truncated");
    std::uint32_t metadata_length = 0;
    std::memcpy(&metadata_length, bytes.data(), sizeof(metadata_length));
    if (metadata_length > 1 && metadata_length < bytes.size() - 4 && bytes[4] == '{') {
        bytes.erase(bytes.begin(), bytes.begin() + 4 + metadata_length);
    }
    return bytes;
}

std::string trt_type_name(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return "int64";
#endif
        default: return "unsupported";
    }
}

std::size_t type_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: throw std::runtime_error("TensorRT tensor uses an unsupported data type");
    }
}

std::vector<int> dimensions(nvinfer1::Dims dims) {
    std::vector<int> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) shape.push_back(dims.d[index]);
    return shape;
}

const DetectorOutputSpec& find_output(const DetectorManifest& manifest, const std::string& name) {
    const auto found = std::find_if(manifest.outputs.begin(), manifest.outputs.end(), [&](const DetectorOutputSpec& output) {
        return output.name == name;
    });
    if (found == manifest.outputs.end()) throw std::runtime_error("TensorRT engine exposes undeclared output tensor '" + name + "'");
    return *found;
}

class TensorRtEngine final : public InferenceEngine {
public:
    TensorRtEngine(ModelConfig config, const fs::path& path)
        : config_(std::move(config)), manifest_(DetectorManifest::from_json(config_.detector_manifest)) {
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "could not query CUDA devices");
        if (device_count < 1) throw std::runtime_error("TensorRT requires an NVIDIA CUDA GPU; CPU fallback is not supported");
        const auto plan = read_engine_plan(path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("could not create the TensorRT runtime");
        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_) throw std::runtime_error("TensorRT engine deserialization failed" +
            (logger_.last_error().empty() ? std::string() : ": " + logger_.last_error()));
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("could not create the TensorRT execution context");
        cuda_check(cudaStreamCreate(&stream_), "could not create reusable CUDA stream");
        try {
            configure_tensors();
        } catch (...) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
            throw;
        }
    }

    ~TensorRtEngine() override {
        for (auto& tensor : tensors_) if (tensor.device) cudaFree(tensor.device);
        if (stream_) cudaStreamDestroy(stream_);
    }

    InferenceResult infer(
        const std::vector<unsigned char>& encoded_image,
        float confidence,
        float,
        const GroundingRequest&
    ) override {
        const auto total_start = Clock::now();
        PreparedImage prepared = prepare_detector_image(encoded_image, manifest_);
        IoTensor& input = tensors_.at(input_index_);
        if (input.type == nvinfer1::DataType::kFLOAT) {
            std::memcpy(input.host.data(), prepared.values.data(), prepared.values.size() * sizeof(float));
        } else if (input.type == nvinfer1::DataType::kHALF) {
            auto* destination = reinterpret_cast<std::uint16_t*>(input.host.data());
            for (std::size_t index = 0; index < prepared.values.size(); ++index) destination[index] = float_to_half(prepared.values[index]);
        } else if (input.type == nvinfer1::DataType::kINT8) {
            if (!(input.quantization_scale > 0.0F)) throw std::runtime_error("TensorRT INT8 input requires quantization scale in the manifest");
            auto* destination = reinterpret_cast<std::int8_t*>(input.host.data());
            for (std::size_t index = 0; index < prepared.values.size(); ++index) {
                destination[index] = quantize_int8(prepared.values[index], input.quantization_scale, input.quantization_zero_point);
            }
        } else {
            throw std::runtime_error("TensorRT input type is not supported for preprocessing");
        }
        const auto preprocess_end = Clock::now();
        cuda_check(cudaMemcpyAsync(input.device, input.host.data(), input.host.size(), cudaMemcpyHostToDevice, stream_), "TensorRT input upload failed");
        const auto inference_start = Clock::now();
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
        for (auto& tensor : tensors_) {
            if (tensor.input) continue;
            cuda_check(cudaMemcpyAsync(tensor.host.data(), tensor.device, tensor.host.size(), cudaMemcpyDeviceToHost, stream_), "TensorRT output download failed");
        }
        cuda_check(cudaStreamSynchronize(stream_), "TensorRT stream synchronization failed");
        const auto inference_end = Clock::now();

        std::vector<TensorData> outputs;
        for (const auto& tensor : tensors_) {
            if (tensor.input) continue;
            TensorData output{tensor.name, tensor.shape, std::vector<float>(tensor.count)};
            if (tensor.type == nvinfer1::DataType::kFLOAT) {
                std::memcpy(output.values.data(), tensor.host.data(), tensor.count * sizeof(float));
            } else if (tensor.type == nvinfer1::DataType::kHALF) {
                const auto* source = reinterpret_cast<const std::uint16_t*>(tensor.host.data());
                for (std::size_t index = 0; index < tensor.count; ++index) output.values[index] = half_to_float(source[index]);
            } else if (tensor.type == nvinfer1::DataType::kINT8) {
                if (!(tensor.quantization_scale > 0.0F)) throw std::runtime_error("TensorRT INT8 output requires quantization metadata in the manifest");
                const auto* source = reinterpret_cast<const std::int8_t*>(tensor.host.data());
                for (std::size_t index = 0; index < tensor.count; ++index) output.values[index] = dequantize_value(source[index], tensor.quantization_scale, tensor.quantization_zero_point);
            } else if (tensor.type == nvinfer1::DataType::kINT32) {
                const auto* source = reinterpret_cast<const std::int32_t*>(tensor.host.data());
                for (std::size_t index = 0; index < tensor.count; ++index) output.values[index] = static_cast<float>(source[index]);
#if NV_TENSORRT_MAJOR >= 10
            } else if (tensor.type == nvinfer1::DataType::kINT64) {
                const auto* source = reinterpret_cast<const std::int64_t*>(tensor.host.data());
                for (std::size_t index = 0; index < tensor.count; ++index) output.values[index] = static_cast<float>(source[index]);
#endif
            }
            outputs.push_back(std::move(output));
        }
        const auto postprocess_start = Clock::now();
        InferenceResult result;
        result.model_id = config_.id;
        result.model_family = manifest_.model_family;
        result.backend = manifest_.backend;
        result.task = ModelTask::ObjectDetection;
        result.source_width = prepared.source_width;
        result.source_height = prepared.source_height;
        result.detections = decode_detector_outputs(manifest_, outputs, prepared.transform,
            std::clamp(confidence, manifest_.score_floor, 1.0F));
        const auto end = Clock::now();
        result.timing.preprocess_ms = milliseconds(total_start, preprocess_end);
        result.timing.inference_ms = milliseconds(inference_start, inference_end);
        result.timing.detection_nms_ms = milliseconds(postprocess_start, end);
        result.timing.backend_total_ms = milliseconds(total_start, end);
        return result;
    }

private:
    struct IoTensor {
        std::string name;
        bool input = false;
        nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
        std::vector<int> shape;
        std::size_t count = 0;
        std::vector<unsigned char> host;
        void* device = nullptr;
        float quantization_scale = 0.0F;
        int quantization_zero_point = 0;
    };

    void configure_tensors() {
        const std::vector<int> expected_input = manifest_.input.layout == "NCHW"
            ? std::vector<int>{1, 3, manifest_.input.height, manifest_.input.width}
            : std::vector<int>{1, manifest_.input.height, manifest_.input.width, 3};
        int inputs = 0;
        int outputs = 0;
        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* raw_name = engine_->getIOTensorName(index);
            if (!raw_name) throw std::runtime_error("TensorRT engine contains an unnamed I/O tensor");
            IoTensor tensor;
            tensor.name = raw_name;
            tensor.input = engine_->getTensorIOMode(raw_name) == nvinfer1::TensorIOMode::kINPUT;
            tensor.type = engine_->getTensorDataType(raw_name);
            tensor.shape = dimensions(engine_->getTensorShape(raw_name));
            if (tensor.input) {
                ++inputs;
                if (tensor.name != manifest_.input.name) throw std::runtime_error("TensorRT input name does not match manifest input.name");
                if (trt_type_name(tensor.type) != manifest_.input.data_type) {
                    throw std::runtime_error("TensorRT input data type does not match the manifest");
                }
                if (std::any_of(tensor.shape.begin(), tensor.shape.end(), [](int value) { return value < 0; })) {
                    nvinfer1::Dims dims{};
                    dims.nbDims = static_cast<int>(expected_input.size());
                    std::copy(expected_input.begin(), expected_input.end(), dims.d);
                    if (!context_->setInputShape(raw_name, dims)) throw std::runtime_error("TensorRT dynamic input shape rejected manifest dimensions");
                    tensor.shape = expected_input;
                }
                if (tensor.shape != expected_input) throw std::runtime_error("TensorRT input shape does not match manifest dimensions/layout");
                input_index_ = tensors_.size();
            } else {
                ++outputs;
                const auto& spec = find_output(manifest_, tensor.name);
                tensor.shape = dimensions(context_->getTensorShape(raw_name));
                if (!tensor_shape_matches(tensor.shape, spec.shape)) throw std::runtime_error("TensorRT output '" + tensor.name + "' shape does not match the manifest");
                if (trt_type_name(tensor.type) != spec.data_type) throw std::runtime_error("TensorRT output '" + tensor.name + "' data type does not match the manifest");
                tensor.quantization_scale = spec.quantization_scale;
                tensor.quantization_zero_point = spec.quantization_zero_point;
            }
            tensor.count = tensor_element_count(tensor.shape);
            tensor.host.resize(tensor.count * type_size(tensor.type));
            cuda_check(cudaMalloc(&tensor.device, tensor.host.size()), "TensorRT reusable buffer allocation failed");
            if (!context_->setTensorAddress(raw_name, tensor.device)) throw std::runtime_error("TensorRT could not bind tensor '" + tensor.name + "'");
            tensors_.push_back(std::move(tensor));
        }
        if (inputs != 1 || outputs != static_cast<int>(manifest_.outputs.size())) {
            throw std::runtime_error("TensorRT input/output count does not match the manifest");
        }
    }

    ModelConfig config_;
    DetectorManifest manifest_;
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    std::vector<IoTensor> tensors_;
    std::size_t input_index_ = 0;
};

#endif

}  // namespace

std::unique_ptr<InferenceEngine> create_tensorrt_engine(const ModelConfig& config, const fs::path& model_path) {
#if RUNNER_WITH_TENSORRT
    return std::make_unique<TensorRtEngine>(config, model_path);
#else
    (void)config;
    (void)model_path;
    throw std::runtime_error("this binary was built without TensorRT/CUDA support; CPU fallback is disabled");
#endif
}

bool tensorrt_available() {
#if RUNNER_WITH_TENSORRT
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

}  // namespace runner
