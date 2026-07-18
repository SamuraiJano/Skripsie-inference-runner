cmake_minimum_required(VERSION 3.20)
project(edge_runner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RUNNER_WITH_NCNN "Build the production NCNN inference engine" ON)
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
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(httplib json stb)

if(RUNNER_WITH_NCNN)
  # The Pi build script pins the NCNN source tag. NCNN generates its CMake
  # package version from the build date, so a conventional minimum-version
  # check here would reject an otherwise correctly pinned installation.
  find_package(ncnn CONFIG REQUIRED)
endif()

add_executable(edge_runner
  src/main.cpp
  src/Types.cpp
  src/ModelStore.cpp
  src/NcnnEngine.cpp
)

target_include_directories(edge_runner PRIVATE include "${stb_SOURCE_DIR}")
target_link_libraries(edge_runner PRIVATE httplib::httplib nlohmann_json::nlohmann_json)
target_compile_definitions(edge_runner PRIVATE
  RUNNER_VERSION="${PROJECT_VERSION}"
  RUNNER_FRONTEND_DIST="${RUNNER_FRONTEND_DIST}"
  RUNNER_WITH_NCNN=$<BOOL:${RUNNER_WITH_NCNN}>
)

if(RUNNER_WITH_NCNN)
  target_link_libraries(edge_runner PRIVATE ncnn)
endif()

# Keep host-side validation artifacts distinct from the production Pi binary.
# Some Windows security scanners retain the generic edge_runner.exe name after
# a prior test process exits; Linux/Pi installations still receive edge_runner.
if(WIN32 AND RUNNER_WITH_NCNN)
  set_target_properties(edge_runner PROPERTIES OUTPUT_NAME edge_runner_ncnn_validation)
elseif(WIN32)
  set_target_properties(edge_runner PROPERTIES OUTPUT_NAME edge_runner_validation)
endif()

if(MSVC)
  target_compile_options(edge_runner PRIVATE /W4 /O2)
else()
  target_compile_options(edge_runner PRIVATE -O3 -Wall -Wextra -Wpedantic)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(edge_runner PRIVATE -mcpu=cortex-a72 -mtune=cortex-a72)
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS edge_runner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
