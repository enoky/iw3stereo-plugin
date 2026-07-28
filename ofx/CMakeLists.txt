cmake_minimum_required(VERSION 3.20)
project(resolve_iw3_ofx CXX)

# CUDA is optional: without it the plugin still builds and runs, just on the CPU
# path. With it, Resolve's GPU render suites keep every frame in device memory.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # 5080 is sm_120; 75 and 86 cover the older cards most users will have.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)
    endif()
    set(IW3_HAS_CUDA ON)
    message(STATUS "CUDA found: building the GPU render path")
else()
    set(IW3_HAS_CUDA OFF)
    message(WARNING "No CUDA compiler: the plugin will be CPU-only")
endif()

# Resolve ships the OpenFX headers and the OFX C++ Support wrapper with the
# application, so there is nothing to vendor. Override with -DOFX_SDK_DIR=... if
# Resolve is installed somewhere unusual.
set(OFX_SDK_DIR
    "C:/ProgramData/Blackmagic Design/DaVinci Resolve/Support/Developer/OpenFX"
    CACHE PATH "Resolve's bundled OpenFX SDK")

if(NOT EXISTS "${OFX_SDK_DIR}/OpenFX-1.4/include/ofxCore.h")
    message(FATAL_ERROR "OpenFX headers not found under ${OFX_SDK_DIR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where Resolve looks for plugins on Windows.
set(OFX_PLUGIN_DIR
    "C:/Program Files/Common Files/OFX/Plugins"
    CACHE PATH "System OFX plugin directory")

# --- the OFX C++ Support wrapper, built once and shared by every plugin -------
file(GLOB OFX_SUPPORT_SOURCES "${OFX_SDK_DIR}/Support/Library/*.cpp")

# ofxsHWNDInteract.cpp cannot compile: Resolve ships the C++ wrapper for the
# HWND interact suite but not the ofxHWNDInteract.h C header it is written
# against. Nothing here needs HWND interacts, so drop it.
list(FILTER OFX_SUPPORT_SOURCES EXCLUDE REGEX "ofxsHWNDInteract\\.cpp$")

add_library(ofx_support STATIC ${OFX_SUPPORT_SOURCES})
target_include_directories(ofx_support PUBLIC
    "${OFX_SDK_DIR}/OpenFX-1.4/include"
    "${OFX_SDK_DIR}/Support/include")
target_compile_definitions(ofx_support PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    # The wrapper is old code; it is not our job to make it warning-clean.
    target_compile_options(ofx_support PRIVATE /w)
endif()

# ofx_bundle(<target> <sources...>)
#
# Builds a DLL, renames it to <target>.ofx, and lays it out in the directory
# structure Resolve expects:
#   <target>.ofx.bundle/Contents/Win64/<target>.ofx
function(ofx_bundle TARGET)
    add_library(${TARGET} MODULE ${ARGN})
    target_link_libraries(${TARGET} PRIVATE ofx_support)
    set(BUNDLE "${CMAKE_BINARY_DIR}/bundles/${TARGET}.ofx.bundle/Contents/Win64")
    set_target_properties(${TARGET} PROPERTIES
        SUFFIX ".ofx"
        LIBRARY_OUTPUT_DIRECTORY "${BUNDLE}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BUNDLE}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BUNDLE}")
endfunction()

# Shared headers: the ORT loader and the log live in common/.
set(IW3_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/common")

ofx_bundle(iw3probe probe/iw3probe.cpp)
target_include_directories(iw3probe PRIVATE "${IW3_COMMON}")

# --- the ONNX Runtime probe --------------------------------------------------
# There is deliberately no import library and no target_link_libraries for ONNX
# Runtime: Resolve ships its own onnxruntime.dll (1.13, CPU-only) in the
# application directory, which Windows would resolve first. The runtime is
# loaded by absolute path at run time instead. See probe/ort_runtime.h.
set(ORT_PYTHON_DIR
    "F:/_AI_PROJECTS_/nunif/venv/Lib/site-packages/onnxruntime/capi"
    CACHE PATH "Directory holding onnxruntime.dll and its provider DLLs")

ofx_bundle(iw3ort probe/iw3ort.cpp common/ort_runtime.cpp)
target_include_directories(iw3ort PRIVATE "${IW3_COMMON}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")

# The runtime goes in a subdirectory of the bundle, so LOAD_WITH_ALTERED_SEARCH_PATH
# picks up its sibling provider DLLs from there rather than from Resolve.
function(stage_runtime TARGET)
    set(BUNDLE "${CMAKE_BINARY_DIR}/bundles/${TARGET}.ofx.bundle/Contents/Win64")
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE}/ort"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_PYTHON_DIR}/onnxruntime.dll"
                "${ORT_PYTHON_DIR}/onnxruntime_providers_shared.dll"
                "${ORT_PYTHON_DIR}/onnxruntime_providers_cuda.dll"
                "${BUNDLE}/ort/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/../models/stereo_warp.onnx"
                "${BUNDLE}/stereo_warp.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/../models/stereo_warp_v3.onnx"
                "${BUNDLE}/stereo_warp_v3.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/../models/light_inpaint_v1.onnx"
                "${BUNDLE}/light_inpaint_v1.onnx"
        COMMENT "Staging ONNX Runtime and the models into the ${TARGET} bundle")
endfunction()

stage_runtime(iw3ort)

# --- the plugin --------------------------------------------------------------
set(IW3STEREO_SOURCES plugin/iw3stereo.cpp plugin/stereo_pipeline.cpp common/ort_runtime.cpp)
if(IW3_HAS_CUDA)
    list(APPEND IW3STEREO_SOURCES plugin/stereo_gpu.cu plugin/monobw_gpu.cu)
endif()

ofx_bundle(iw3stereo ${IW3STEREO_SOURCES})
target_include_directories(iw3stereo PRIVATE
    "${IW3_COMMON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
if(IW3_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(iw3stereo PRIVATE CUDA::cudart_static)
    set_target_properties(iw3stereo PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
stage_runtime(iw3stereo)

# The user-facing README goes at the bundle root, which is where someone poking
# around in the OFX plugins directory will actually look.
add_custom_command(TARGET iw3stereo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/plugin/README.txt"
            "${CMAKE_BINARY_DIR}/bundles/iw3stereo.ofx.bundle/README.txt"
    COMMENT "Staging the user README into the iw3stereo bundle")

# --- the numeric core's test, built as a plain executable --------------------
add_executable(test_pipeline ../tests/cpp/test_pipeline.cpp plugin/stereo_pipeline.cpp)
target_include_directories(test_pipeline PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp")
set_target_properties(test_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests")

# The MonoBW kernels against the same reference data. The kernels share
# monobw_math.h with the CPU, so this checks the wiring the header cannot.
if(IW3_HAS_CUDA)
    add_executable(test_monobw_gpu ../tests/cpp/test_monobw_gpu.cu plugin/monobw_gpu.cu)
    target_include_directories(test_monobw_gpu PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/plugin"
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp")
    set_target_properties(test_monobw_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests"
        CUDA_SEPARABLE_COMPILATION OFF)
endif()

# `cmake --install` copies the bundles next to Resolve's other OFX plugins.
install(DIRECTORY "${CMAKE_BINARY_DIR}/bundles/" DESTINATION "${OFX_PLUGIN_DIR}")
