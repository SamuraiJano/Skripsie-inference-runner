#pragma once

#include <vector>

#include "runner/Types.hpp"

namespace runner {

struct MaskDecodeRequest {
    int source_width = 0;
    int source_height = 0;
    int input_width = 0;
    int input_height = 0;
    int prototype_width = 0;
    int prototype_height = 0;
    int prototype_channels = 0;
    float letterbox_scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    float box_x = 0.0F;
    float box_y = 0.0F;
    float box_width = 0.0F;
    float box_height = 0.0F;
    float threshold = 0.5F;
    std::vector<float> coefficients;
    std::vector<float> prototypes;  // channel-major CxHxW
};

MaskRle decode_yolo_mask_rle(const MaskDecodeRequest& request);
std::vector<unsigned char> decode_rle(const MaskRle& encoded);

}  // namespace runner
