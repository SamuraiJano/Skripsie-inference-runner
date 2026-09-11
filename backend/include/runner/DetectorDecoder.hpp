#pragma once

#include <vector>

#include "runner/DetectorManifest.hpp"
#include "runner/ImagePreprocess.hpp"
#include "runner/TensorUtils.hpp"
#include "runner/Types.hpp"

namespace runner {

std::vector<Detection> decode_detector_outputs(
    const DetectorManifest& manifest,
    const std::vector<TensorData>& tensors,
    const ImageTransform& transform,
    float confidence_floor
);

}  // namespace runner
