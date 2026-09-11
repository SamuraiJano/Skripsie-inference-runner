cmake_minimum_required(VERSION 3.20)
project(inference_runner VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RUNNER_BUILD_APP "Build the HTTP application" ON)
option(RUNNER_WITH_TFLITE "Build the production TFLite/LiteRT CPU engine" OFF)
option(RUNNER_WITH_TENSORRT "Build the production TensorRT/CUDA engine" OFF)
option(RUNNER_WITH_NCNN "Build the isolated legacy NCNN engine" OFF)
option(RUNNER_REQUIRE_VULKAN "Require an NCNN build with Vulkan compute" ON)
set(RUNNER_TFLITE_ROOT "" CACHE PATH "Installed TFLite prefix containing include/ and lib/")
set(RUNNER_TFLITE_SOURCE_DIR "" CACHE PATH "Pinned TensorFlow source checkout; tensorflow/lite is added when set")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root containing include/ and lib/")
set(RUNNER_FRONTEND_DIST "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/dist" CACHE PATH "Built React application")

include(FetchContent)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  stb
  URL https://github.com/nothings/stb/archive/31c1ad37456438565541f4919958214b6e762fb4.tar.gz
  URL_HASH SHA256=e4e3bba9c572a4a4148373a914d88ea0f0d11de8cc2c66739926e7eca0223319
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json stb)

add_library(runner_core
  src/Types.cpp
  src/DetectorManifest.cpp
  src/TensorUtils.cpp
  src/ImagePreprocess.cpp
  src/DetectorDecoder.cpp
  src/ByteTrack.cpp
  src/TrackingPipeline.cpp
  src/YoloPostprocess.cpp
  src/Pipeline.cpp
)
target_include_directories(runner_core PUBLIC include PRIVATE "${stb_SOURCE_DIR}")
target_link_libraries(runner_core PUBLIC nlohmann_json::nlohmann_json)

if(RUNNER_BUILD_APP)
  set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.26.0.tar.gz
    URL_HASH SHA256=a66f908f50ccb119769adce44fe1eac75f81b6ffab7c4ac0211bb663ffeb2688
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(httplib)

  if(RUNNER_WITH_NCNN)
    find_package(ncnn CONFIG REQUIRED)
    if(RUNNER_REQUIRE_VULKAN AND NOT NCNN_VULKAN)
      message(FATAL_ERROR "RUNNER_REQUIRE_VULKAN=ON requires an NCNN build with Vulkan")
    endif()
  endif()

  set(RUNNER_TFLITE_TARGET "")
  set(RUNNER_TFLITE_INCLUDE "")
  set(RUNNER_TFLITE_XNNPACK OFF)
  if(RUNNER_WITH_TFLITE)
    if(RUNNER_TFLITE_SOURCE_DIR)
      if(NOT EXISTS "${RUNNER_TFLITE_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
        message(FATAL_ERROR "RUNNER_TFLITE_SOURCE_DIR does not contain the pinned tensorflow/lite source")
      endif()
      set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
      add_subdirectory("${RUNNER_TFLITE_SOURCE_DIR}/tensorflow/lite" "${CMAKE_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
      set(RUNNER_TFLITE_TARGET tensorflow-lite)
      set(RUNNER_TFLITE_INCLUDE "${RUNNER_TFLITE_SOURCE_DIR}")
      if(EXISTS "${RUNNER_TFLITE_SOURCE_DIR}/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h")
        set(RUNNER_TFLITE_XNNPACK ON)
      endif()
    else()
      find_path(RUNNER_TFLITE_INCLUDE tensorflow/lite/interpreter.h HINTS "${RUNNER_TFLITE_ROOT}/include")
      find_library(RUNNER_TFLITE_LIBRARY NAMES tensorflow-lite tensorflowlite
        HINTS "${RUNNER_TFLITE_ROOT}/lib" "${RUNNER_TFLITE_ROOT}/lib64")
      if(NOT RUNNER_TFLITE_INCLUDE OR NOT RUNNER_TFLITE_LIBRARY)
        message(FATAL_ERROR "TFLite was enabled but its C++ headers/library were not found. Set RUNNER_TFLITE_ROOT or RUNNER_TFLITE_SOURCE_DIR.")
      endif()
      set(RUNNER_TFLITE_TARGET "${RUNNER_TFLITE_LIBRARY}")
      if(EXISTS "${RUNNER_TFLITE_INCLUDE}/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h")
        set(RUNNER_TFLITE_XNNPACK ON)
      endif()
    endif()
  endif()

  if(RUNNER_WITH_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}/include" "$ENV{TensorRT_ROOT}/include")
    find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer nvinfer_11 nvinfer_10
      HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" "$ENV{TensorRT_ROOT}/lib" "$ENV{TensorRT_ROOT}/lib64")
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
      message(FATAL_ERROR "TensorRT was enabled but the SDK headers/import library were not found. Set TENSORRT_ROOT. Python wheels alone do not provide the C++ SDK.")
    endif()
  endif()

  add_executable(inference_runner
    src/main.cpp
    src/ModelStore.cpp
    src/NcnnEngine.cpp
    src/TfliteEngine.cpp
    src/TensorRtEngine.cpp
    src/PythonEngine.cpp
    src/EngineFactory.cpp
  )
  if(WIN32)
    set_target_properties(inference_runner PROPERTIES OUTPUT_NAME inference_runner_windows)
  endif()
  target_include_directories(inference_runner PRIVATE include "${stb_SOURCE_DIR}")
  target_link_libraries(inference_runner PRIVATE runner_core httplib::httplib nlohmann_json::nlohmann_json)
  target_compile_definitions(inference_runner PRIVATE
    RUNNER_VERSION="${PROJECT_VERSION}"
    RUNNER_FRONTEND_DIST="${RUNNER_FRONTEND_DIST}"
    RUNNER_WITH_NCNN=$<BOOL:${RUNNER_WITH_NCNN}>
    RUNNER_WITH_TFLITE=$<BOOL:${RUNNER_WITH_TFLITE}>
    RUNNER_WITH_TENSORRT=$<BOOL:${RUNNER_WITH_TENSORRT}>
    RUNNER_TFLITE_XNNPACK=$<BOOL:${RUNNER_TFLITE_XNNPACK}>
  )
  if(WIN32)
    target_compile_definitions(inference_runner PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_link_libraries(inference_runner PRIVATE Psapi)
  endif()
  if(RUNNER_WITH_NCNN)
    target_link_libraries(inference_runner PRIVATE ncnn)
  endif()
  if(RUNNER_WITH_TFLITE)
    target_include_directories(inference_runner PRIVATE "${RUNNER_TFLITE_INCLUDE}")
    target_link_libraries(inference_runner PRIVATE "${RUNNER_TFLITE_TARGET}")
  endif()
  if(RUNNER_WITH_TENSORRT)
    target_include_directories(inference_runner PRIVATE "${TENSORRT_INCLUDE_DIR}")
    target_link_libraries(inference_runner PRIVATE "${TENSORRT_NVINFER_LIBRARY}" CUDA::cudart)
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(runner_tests tests/RunnerTests.cpp)
  target_link_libraries(runner_tests PRIVATE runner_core)
  add_executable(detector_tracking_tests tests/DetectorTrackingTests.cpp)
  target_link_libraries(detector_tracking_tests PRIVATE runner_core)
  if(MSVC)
    target_compile_options(runner_tests PRIVATE /UNDEBUG)
    target_compile_options(detector_tracking_tests PRIVATE /UNDEBUG)
  else()
    target_compile_options(runner_tests PRIVATE -UNDEBUG)
    target_compile_options(detector_tracking_tests PRIVATE -UNDEBUG)
  endif()
  add_test(NAME runner_tests COMMAND runner_tests)
  add_test(NAME detector_tracking_tests COMMAND detector_tracking_tests)
endif()

if(MSVC)
  target_compile_options(runner_core PRIVATE /W4 /O2 /permissive-)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    target_compile_options(runner_core PRIVATE /arch:AVX2)
  endif()
  if(RUNNER_BUILD_APP)
    target_compile_options(inference_runner PRIVATE /W4 /O2 /permissive-)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
      target_compile_options(inference_runner PRIVATE /arch:AVX2)
    endif()
  endif()
else()
  target_compile_options(runner_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
  if(RUNNER_BUILD_APP)
    target_compile_options(inference_runner PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()
endif()

if(RUNNER_BUILD_APP)
  include(GNUInstallDirs)
  install(TARGETS inference_runner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
