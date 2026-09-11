#pragma once

#include "runner/Engine.hpp"

namespace runner {

// The public factory is declared in Engine.hpp. This header exists so platform
// builds can explicitly include the TFLite backend without exposing runtime
// headers to engine-independent code.

}  // namespace runner
