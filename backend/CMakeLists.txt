cmake_minimum_required(VERSION 3.20)
project(inference_runner_windows VERSION 1.0.0 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "Inference Runner Windows supports 64-bit Windows only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RUNNER_WITH_NCNN "Build the production NCNN inference engine" ON)
option(RUNNER_REQUIRE_VULKAN "Require an NCNN build with Vulkan compute" ON)
set(RUNNER_FRONTEND_DIST "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/dist" CACHE PATH "Built React application")

include(FetchContent)
set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.26.0.tar.gz
  URL_HASH SHA256=a66f908f50ccb119769adce44fe1eac75f81b6ffab7c4ac0211bb663ffeb2688
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
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
FetchContent_MakeAvailable(httplib json stb)

if(RUNNER_WITH_NCNN)
  find_package(ncnn CONFIG REQUIRED)
  if(RUNNER_REQUIRE_VULKAN AND NOT NCNN_VULKAN)
    message(FATAL_ERROR "The Windows runner requires NCNN_VULKAN=ON. Use scripts/build_windows.ps1.")
  endif()
endif()

add_library(runner_core
  src/Types.cpp
  src/YoloPostprocess.cpp
  src/Pipeline.cpp
)
target_include_directories(runner_core PUBLIC include PRIVATE "${stb_SOURCE_DIR}")
target_link_libraries(runner_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(inference_runner_windows
  src/main.cpp
  src/ModelStore.cpp
  src/NcnnEngine.cpp
  src/PythonEngine.cpp
  src/EngineFactory.cpp
)

target_include_directories(inference_runner_windows PRIVATE include "${stb_SOURCE_DIR}")
target_link_libraries(inference_runner_windows PRIVATE runner_core httplib::httplib nlohmann_json::nlohmann_json Psapi)
target_compile_definitions(inference_runner_windows PRIVATE
  RUNNER_VERSION="${PROJECT_VERSION}"
  RUNNER_FRONTEND_DIST="${RUNNER_FRONTEND_DIST}"
  RUNNER_WITH_NCNN=$<BOOL:${RUNNER_WITH_NCNN}>
  WIN32_LEAN_AND_MEAN
  NOMINMAX
)

if(RUNNER_WITH_NCNN)
  target_link_libraries(inference_runner_windows PRIVATE ncnn)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(runner_tests tests/RunnerTests.cpp)
  target_link_libraries(runner_tests PRIVATE runner_core)
  if(MSVC)
    target_compile_options(runner_tests PRIVATE /UNDEBUG)
  else()
    target_compile_options(runner_tests PRIVATE -UNDEBUG)
  endif()
  add_test(NAME runner_tests COMMAND runner_tests)
endif()

if(MSVC)
  target_compile_options(runner_core PRIVATE /W4 /O2 /arch:AVX2 /permissive-)
  target_compile_options(inference_runner_windows PRIVATE /W4 /O2 /arch:AVX2 /permissive-)
else()
  target_compile_options(runner_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_compile_options(inference_runner_windows PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)
install(TARGETS inference_runner_windows RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
