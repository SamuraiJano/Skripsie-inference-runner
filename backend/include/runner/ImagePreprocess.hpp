#pragma once

#include <vector>

#include "runner/DetectorManifest.hpp"

namespace runner {

struct ImageTransform {
    int source_width = 0;
    int source_height = 0;
    int input_width = 0;
    int input_height = 0;
    float scale_x = 1.0F;
    float scale_y = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
    bool letterboxed = false;
};

struct PreparedImage {
    int source_width = 0;
    int source_height = 0;
    std::vector<float> values;
    ImageTransform transform;
};

PreparedImage prepare_detector_image(const std::vector<unsigned char>& encoded_image, const DetectorManifest& manifest);

}  // namespace runner
