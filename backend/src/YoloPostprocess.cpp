#include "runner/YoloPostprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace runner {
namespace {

float bilinear(const std::vector<float>& values, int width, int height, float x, float y) {
    x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);
    const float top = values[static_cast<std::size_t>(y0 * width + x0)] * (1.0F - dx)
        + values[static_cast<std::size_t>(y0 * width + x1)] * dx;
    const float bottom = values[static_cast<std::size_t>(y1 * width + x0)] * (1.0F - dx)
        + values[static_cast<std::size_t>(y1 * width + x1)] * dx;
    return top * (1.0F - dy) + bottom * dy;
}

}  // namespace

MaskRle decode_yolo_mask_rle(const MaskDecodeRequest& request) {
    if (request.source_width <= 0 || request.source_height <= 0 || request.input_width <= 0 || request.input_height <= 0) {
        throw std::invalid_argument("mask decode dimensions must be positive");
    }
    if (request.prototype_width <= 0 || request.prototype_height <= 0 || request.prototype_channels <= 0) {
        throw std::invalid_argument("prototype dimensions must be positive");
    }
    const std::size_t plane = static_cast<std::size_t>(request.prototype_width * request.prototype_height);
    if (request.coefficients.size() != static_cast<std::size_t>(request.prototype_channels)
        || request.prototypes.size() != plane * static_cast<std::size_t>(request.prototype_channels)) {
        throw std::invalid_argument("mask coefficients/prototypes do not match declared dimensions");
    }
    if (!(request.threshold > 0.0F && request.threshold < 1.0F) || request.letterbox_scale <= 0.0F) {
        throw std::invalid_argument("mask threshold and letterbox scale are invalid");
    }

    std::vector<float> logits(plane, 0.0F);
    for (int channel = 0; channel < request.prototype_channels; ++channel) {
        const float coefficient = request.coefficients[static_cast<std::size_t>(channel)];
        const float* source = request.prototypes.data() + static_cast<std::size_t>(channel) * plane;
        for (std::size_t index = 0; index < plane; ++index) logits[index] += coefficient * source[index];
    }
    const float threshold_logit = std::log(request.threshold / (1.0F - request.threshold));
    const int left = std::clamp(static_cast<int>(std::floor(request.box_x)), 0, request.source_width);
    const int top = std::clamp(static_cast<int>(std::floor(request.box_y)), 0, request.source_height);
    const int right = std::clamp(static_cast<int>(std::ceil(request.box_x + request.box_width)), 0, request.source_width);
    const int bottom = std::clamp(static_cast<int>(std::ceil(request.box_y + request.box_height)), 0, request.source_height);

    MaskRle encoded;
    encoded.width = request.source_width;
    encoded.height = request.source_height;
    bool current = false;
    std::uint32_t run = 0;
    for (int source_y = 0; source_y < request.source_height; ++source_y) {
        for (int source_x = 0; source_x < request.source_width; ++source_x) {
            bool active = false;
            if (source_x >= left && source_x < right && source_y >= top && source_y < bottom) {
                const float input_x = (static_cast<float>(source_x) + 0.5F) * request.letterbox_scale + request.pad_x;
                const float input_y = (static_cast<float>(source_y) + 0.5F) * request.letterbox_scale + request.pad_y;
                const float prototype_x = (input_x + 0.5F) * request.prototype_width / request.input_width - 0.5F;
                const float prototype_y = (input_y + 0.5F) * request.prototype_height / request.input_height - 0.5F;
                active = bilinear(logits, request.prototype_width, request.prototype_height, prototype_x, prototype_y) >= threshold_logit;
            }
            if (active == current) {
                ++run;
            } else {
                encoded.counts.push_back(run);
                current = active;
                run = 1;
            }
        }
    }
    encoded.counts.push_back(run);
    return encoded;
}

std::vector<unsigned char> decode_rle(const MaskRle& encoded) {
    const std::size_t total = static_cast<std::size_t>(encoded.width) * static_cast<std::size_t>(encoded.height);
    std::vector<unsigned char> pixels;
    pixels.reserve(total);
    unsigned char value = 0;
    for (std::uint32_t count : encoded.counts) {
        if (pixels.size() + count > total) throw std::invalid_argument("RLE counts exceed mask dimensions");
        pixels.insert(pixels.end(), count, value);
        value = value ? 0 : 1;
    }
    if (pixels.size() != total) throw std::invalid_argument("RLE counts do not fill mask dimensions");
    return pixels;
}

}  // namespace runner
