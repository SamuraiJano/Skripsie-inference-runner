#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="$root_dir/.deps"
ncnn_source="$deps_dir/ncnn-src"
ncnn_build="$deps_dir/ncnn-build"
ncnn_install="$deps_dir/ncnn-install"
backend_build="$root_dir/build/pi"
ncnn_tag="20260526"
jobs="${BUILD_JOBS:-4}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Warning: this build is tuned for the Pi 4 Cortex-A72 and is expected to run on aarch64."
fi

mkdir -p "$deps_dir"

if [[ ! -f "$ncnn_install/lib/cmake/ncnn/ncnnConfig.cmake" ]]; then
  rm -rf "$ncnn_source" "$ncnn_build" "$ncnn_install"
  git clone --branch "$ncnn_tag" --depth 1 --recurse-submodules https://github.com/Tencent/ncnn.git "$ncnn_source"
  cmake -S "$ncnn_source" -B "$ncnn_build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ncnn_install" \
    -DNCNN_VULKAN=OFF \
    -DNCNN_OPENMP=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_SHARED_LIB=OFF
  cmake --build "$ncnn_build" --parallel "$jobs"
  cmake --install "$ncnn_build"
fi

npm ci --prefix "$root_dir/frontend" --no-audit --no-fund
npm run build --prefix "$root_dir/frontend"

cmake -S "$root_dir/backend" -B "$backend_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRUNNER_WITH_NCNN=ON \
  -Dncnn_DIR="$ncnn_install/lib/cmake/ncnn" \
  -DRUNNER_FRONTEND_DIST="$root_dir/frontend/dist"
cmake --build "$backend_build" --parallel "$jobs"

echo
echo "Build complete: $backend_build/edge_runner"
echo "Run: $backend_build/edge_runner --data $root_dir/data/models"

