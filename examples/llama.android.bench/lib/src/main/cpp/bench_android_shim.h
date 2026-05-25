#pragma once

#include <csetjmp>
#include <cstdlib>

// Redirect exit() in llama-bench.cpp back to the JNI caller instead of killing the app.
extern thread_local int g_bench_exit_code;
extern thread_local jmp_buf g_bench_jmp_buf;

#define exit(status)                                                                 \
    do {                                                                             \
        g_bench_exit_code = (status);                                                \
        longjmp(g_bench_jmp_buf, 1);                                                 \
    } while (0)
