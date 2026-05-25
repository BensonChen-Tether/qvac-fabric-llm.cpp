#pragma once
#include <android/log.h>

#ifndef LOG_TAG
#define LOG_TAG "llama-bench"
#endif

#ifndef LOG_MIN_LEVEL
#if defined(NDEBUG)
#define LOG_MIN_LEVEL ANDROID_LOG_INFO
#else
#define LOG_MIN_LEVEL ANDROID_LOG_VERBOSE
#endif
#endif

static inline int bench_should_log(int prio) {
    return __android_log_is_loggable(prio, LOG_TAG, LOG_MIN_LEVEL);
}

#define LOGi(...) do { if (bench_should_log(ANDROID_LOG_INFO)) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGw(...) do { if (bench_should_log(ANDROID_LOG_WARN)) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGe(...) do { if (bench_should_log(ANDROID_LOG_ERROR)) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); } while (0)

static inline int android_log_prio_from_ggml(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: return ANDROID_LOG_ERROR;
        case GGML_LOG_LEVEL_WARN:  return ANDROID_LOG_WARN;
        case GGML_LOG_LEVEL_INFO:  return ANDROID_LOG_INFO;
        case GGML_LOG_LEVEL_DEBUG: return ANDROID_LOG_DEBUG;
        default:                   return ANDROID_LOG_DEFAULT;
    }
}

static inline void bench_android_log_callback(enum ggml_log_level level,
                                              const char * text,
                                              void * /*user*/) {
    const int prio = android_log_prio_from_ggml(level);
    if (!bench_should_log(prio)) {
        return;
    }
    __android_log_write(prio, LOG_TAG, text);
}
