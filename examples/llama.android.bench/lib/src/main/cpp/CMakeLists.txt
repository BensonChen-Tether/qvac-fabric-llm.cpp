cmake_minimum_required(VERSION 3.31.6)

project("llama-bench" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        if(GGML_VULKAN)
            set(GGML_CPU_KLEIDIAI OFF)
            set(GGML_OPENMP OFF)
        else()
            set(GGML_CPU_KLEIDIAI ON)
            set(GGML_OPENMP ON)
        endif()
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

if(GGML_VULKAN)
    message(STATUS "llama-bench Android: Vulkan GPU backend enabled")
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

set(LLAMA_BENCH_SRC ${LLAMA_SRC}/tools/llama-bench/llama-bench.cpp)

add_library(${CMAKE_PROJECT_NAME} SHARED
        llama_bench_jni.cpp
        ${LLAMA_BENCH_SRC})

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
        -include ${CMAKE_CURRENT_LIST_DIR}/bench_android_shim.h)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

if(GGML_VULKAN)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_VULKAN_BACKEND=1)
endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        log)
