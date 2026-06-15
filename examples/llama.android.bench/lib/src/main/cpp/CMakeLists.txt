cmake_minimum_required(VERSION 3.31.6)

project("llama-bench-lib" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# llama.cpp + llama-bench executable
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

# Vulkan headers (vulkan.hpp) come from the host Vulkan SDK; libvulkan.so from the NDK sysroot.
if(ANDROID)
    if(DEFINED ENV{VULKAN_SDK})
        set(_vulkan_include "$ENV{VULKAN_SDK}/include")
    elseif(EXISTS "/opt/homebrew/include/vulkan/vulkan.hpp")
        set(_vulkan_include "/opt/homebrew/include")
    elseif(EXISTS "/usr/local/include/vulkan/vulkan.hpp")
        set(_vulkan_include "/usr/local/include")
    endif()
    if(_vulkan_include)
        set(Vulkan_INCLUDE_DIR "${_vulkan_include}" CACHE PATH "Vulkan include directory")
        message(STATUS "Using Vulkan headers from ${Vulkan_INCLUDE_DIR}")
    else()
        message(WARNING "Vulkan headers not found; install Vulkan SDK or set VULKAN_SDK")
    endif()

    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(_android_triple "aarch64-linux-android")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(_android_triple "x86_64-linux-android")
    endif()
    if(CMAKE_SYSROOT AND _android_triple)
        set(Vulkan_LIBRARY "${CMAKE_SYSROOT}/usr/lib/${_android_triple}/${ANDROID_NATIVE_API_LEVEL}/libvulkan.so"
            CACHE FILEPATH "Vulkan library")
        message(STATUS "Using Vulkan library ${Vulkan_LIBRARY}")
    endif()

    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc
        HINTS ENV VULKAN_SDK
        PATH_SUFFIXES bin
        /usr/local/bin /opt/homebrew/bin)
    if(Vulkan_GLSLC_EXECUTABLE)
        message(STATUS "Using glslc ${Vulkan_GLSLC_EXECUTABLE}")
    endif()
endif()

# Bonsai GGUF models require prism-llama.cpp (qvac cannot load them).
# Path: Tether/prism-llama.cpp (seven levels up from this cpp dir to repo root).
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../prism-llama.cpp)
add_subdirectory(${LLAMA_SRC} build-llama)

# Minimal shared library so AGP packages llama/ggml backend .so files.
add_library(llama-bench-stub SHARED bench_stub.cpp)
target_link_libraries(llama-bench-stub PRIVATE llama android log)

set(TARGET llama-bench)
add_executable(${TARGET} ${LLAMA_SRC}/tools/llama-bench/llama-bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Package the per-ABI binary into app assets during native build.
set(ASSET_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../app/src/main/assets/bin/${ANDROID_ABI})
add_custom_command(TARGET ${TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSET_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET}> ${ASSET_DIR}/llama-bench
    COMMENT "Copying llama-bench to assets for ${ANDROID_ABI}"
)
