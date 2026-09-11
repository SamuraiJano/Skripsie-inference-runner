#include "runner/ImagePreprocess.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include <stb_image.h>

namespace runner {
namespace {

float sample_channel(const unsigned char* pixels, int width, int height, float x, float y, int channel, float fill) {
    if (x < -0.5F || y < -0.5F || x > static_cast<float>(width) - 0.5F || y > static_cast<float>(height) - 0.5F) return fill;
    x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);
    const auto at = [&](int px, int py) { return static_cast<float>(pixels[(static_cast<std::size_t>(py) * width + px) * 3 + channel]); };
    const float top = at(x0, y0) * (1.0F - dx) + at(x1, y0) * dx;
    const float bottom = at(x0, y1) * (1.0F - dx) + at(x1, y1) * dx;
    return top * (1.0F - dy) + bottom * dy;
}

}  // namespace

PreparedImage prepare_detector_image(const std::vector<unsigned char>& encoded_image, const DetectorManifest& manifest) {
    if (encoded_image.empty()) throw std::invalid_argument("encoded image is empty");
    int source_width = 0;
    int source_height = 0;
    int source_channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{
        stbi_load_from_memory(encoded_image.data(), static_cast<int>(encoded_image.size()), &source_width, &source_height, &source_channels, 3),
        stbi_image_free,
    };
    if (!pixels || source_width <= 0 || source_height <= 0) {
        throw std::invalid_argument(std::string("could not decode image: ") + stbi_failure_reason());
    }

    PreparedImage prepared;
    prepared.source_width = source_width;
    prepared.source_height = source_height;
    prepared.transform.source_width = source_width;
    prepared.transform.source_height = source_height;
    prepared.transform.input_width = manifest.input.width;
    prepared.transform.input_height = manifest.input.height;
    prepared.transform.letterboxed = manifest.input.resize_mode == "letterbox";
    if (prepared.transform.letterboxed) {
        const float scale = std::min(
            static_cast<float>(manifest.input.width) / source_width,
            static_cast<float>(manifest.input.height) / source_height);
        const int resized_width = static_cast<int>(std::round(source_width * scale));
        const int resized_height = static_cast<int>(std::round(source_height * scale));
        prepared.transform.scale_x = scale;
        prepared.transform.scale_y = scale;
        prepared.transform.pad_x = static_cast<float>((manifest.input.width - resized_width) / 2);
        prepared.transform.pad_y = static_cast<float>((manifest.input.height - resized_height) / 2);
    } else {
        prepared.transform.scale_x = static_cast<float>(manifest.input.width) / source_width;
        prepared.transform.scale_y = static_cast<float>(manifest.input.height) / source_height;
    }

    const std::size_t plane = static_cast<std::size_t>(manifest.input.width) * manifest.input.height;
    prepared.values.resize(plane * 3);
    for (int y = 0; y < manifest.input.height; ++y) {
        for (int x = 0; x < manifest.input.width; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5F - prepared.transform.pad_x) / prepared.transform.scale_x - 0.5F;
            const float source_y = (static_cast<float>(y) + 0.5F - prepared.transform.pad_y) / prepared.transform.scale_y - 0.5F;
            for (int output_channel = 0; output_channel < 3; ++output_channel) {
                const int source_channel = manifest.input.colour_order == "RGB" ? output_channel : 2 - output_channel;
                const float pixel = sample_channel(pixels.get(), source_width, source_height, source_x, source_y,
                    source_channel, manifest.input.letterbox_value);
                const float value = (pixel - manifest.input.mean[static_cast<std::size_t>(output_channel)])
                    * manifest.input.scale[static_cast<std::size_t>(output_channel)];
                const std::size_t pixel_index = static_cast<std::size_t>(y) * manifest.input.width + x;
                const std::size_t destination = manifest.input.layout == "NCHW"
                    ? static_cast<std::size_t>(output_channel) * plane + pixel_index
                    : pixel_index * 3 + static_cast<std::size_t>(output_channel);
                prepared.values[destination] = value;
            }
        }
    }
    return prepared;
}

}  // namespace runner
