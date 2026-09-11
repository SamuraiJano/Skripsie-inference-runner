#include "runner/TensorUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace runner {

std::size_t tensor_element_count(const std::vector<int>& shape) {
    if (shape.empty()) throw std::invalid_argument("tensor shape cannot be empty");
    std::size_t count = 1;
    for (int dimension : shape) {
        if (dimension <= 0) throw std::invalid_argument("concrete tensor dimensions must be positive");
        if (count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
            throw std::overflow_error("tensor element count overflows size_t");
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

bool tensor_shape_matches(const std::vector<int>& actual, const std::vector<int>& expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] <= 0 || (expected[index] != -1 && actual[index] != expected[index])) return false;
    }
    return true;
}

std::uint8_t quantize_uint8(float value, float scale, int zero_point) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) throw std::invalid_argument("quantization scale must be positive and finite");
    const long quantized = std::lround(value / scale) + zero_point;
    return static_cast<std::uint8_t>(std::clamp<long>(quantized, 0, 255));
}

std::int8_t quantize_int8(float value, float scale, int zero_point) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) throw std::invalid_argument("quantization scale must be positive and finite");
    const long quantized = std::lround(value / scale) + zero_point;
    return static_cast<std::int8_t>(std::clamp<long>(quantized, -128, 127));
}

float dequantize_value(int value, float scale, int zero_point) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) throw std::invalid_argument("quantization scale must be positive and finite");
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    int exponent = static_cast<int>((bits >> 23U) & 0xFFU) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa = (mantissa | 0x800000U) >> static_cast<unsigned int>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U | (mantissa ? 0x0200U : 0U));
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10U) | ((mantissa + 0x1000U) >> 13U));
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1FU;
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03FFU;
            bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace runner
