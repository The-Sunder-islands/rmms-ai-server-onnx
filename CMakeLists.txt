cmake_minimum_required(VERSION 3.21)
project(rmms-ai-server-onnx
    VERSION 0.1.0
    DESCRIPTION "RMMS AI Server — ONNX Runtime + C++20 audio stem separation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies via Conan ---
if(EXISTS ${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
    message(STATUS "Using Conan toolchain")
endif()

find_package(ONNXRuntime REQUIRED)
find_package(Drogon REQUIRED)
find_package(spdlog REQUIRED)
find_package(glaze REQUIRED)
find_package(tomlplusplus REQUIRED)
find_package(GTest REQUIRED)

# --- ONNX Runtime C API ---
add_library(onnxruntime_c INTERFACE)
target_include_directories(onnxruntime_c INTERFACE ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_directories(onnxruntime_c INTERFACE ${ONNXRUNTIME_LIB_DIRS})
target_link_libraries(onnxruntime_c INTERFACE ${ONNXRUNTIME_LIBRARIES})

# --- Main executable ---
add_executable(rmms-ai-server-onnx
    src/main.cpp
)

# --- Static libraries ---
add_library(rmms_engine STATIC
    src/engine/session.cpp
    src/engine/preprocess.cpp
)

add_library(rmms_device STATIC
    src/device/registry.cpp
    src/device/cpu.cpp
    src/device/cuda.cpp
    src/device/directml.cpp
    src/device/opencl.cpp
)

add_library(rmms_protocol STATIC
    src/protocol/models.cpp
    src/protocol/errors.cpp
    src/protocol/serialize.cpp
)

add_library(rmms_util STATIC
    src/util/audio.cpp
    src/util/config.cpp
    src/util/logging.cpp
)

# --- Headers ---
target_include_directories(rmms_engine   PUBLIC include)
target_include_directories(rmms_device   PUBLIC include)
target_include_directories(rmms_protocol PUBLIC include)
target_include_directories(rmms_util     PUBLIC include)

# --- Link main ---
target_link_libraries(rmms-ai-server-onnx PRIVATE
    Drogon::Drogon
    rmms_engine
    rmms_device
    rmms_protocol
    rmms_util
)

# --- Tests ---
enable_testing()
add_executable(rmms_tests
    tests/test_session.cpp
    tests/test_config.cpp
    tests/test_models.cpp
)
target_link_libraries(rmms_tests PRIVATE
    GTest::gtest_main
    rmms_engine
    rmms_device
    rmms_protocol
    rmms_util
)
include(GoogleTest)
gtest_discover_tests(rmms_tests)

# --- Platform ---
if(WIN32)
    target_compile_definitions(rmms-ai-server-onnx PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(rmms-ai-server-onnx PRIVATE -fsanitize=address,undefined)
endif()
