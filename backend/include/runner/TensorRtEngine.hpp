#pragma once

#include "runner/Engine.hpp"

namespace runner {

// The public factory is declared in Engine.hpp. TensorRT and CUDA headers stay
// private to TensorRtEngine.cpp and are only required when enabled by CMake.

}  // namespace runner
