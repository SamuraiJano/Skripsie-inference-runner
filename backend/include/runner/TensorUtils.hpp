#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace runner {

struct TensorData {
    std::string name;
    std::vector<int> shape;
    std::vector<float> values;
};

std::size_t tensor_element_count(const std::vector<int>& shape);
bool tensor_shape_matches(const std::vector<int>& actual, const std::vector<int>& expected);
std::uint8_t quantize_uint8(float value, float scale, int zero_point);
std::int8_t quantize_int8(float value, float scale, int zero_point);
float dequantize_value(int value, float scale, int zero_point);
std::uint16_t float_to_half(float value);
float half_to_float(std::uint16_t value);

}  // namespace runner
