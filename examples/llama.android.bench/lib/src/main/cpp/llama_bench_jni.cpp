#include "bench_android_shim.h"

#include <android/log.h>
#include <jni.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"
#include "logging.h"

thread_local int     g_bench_exit_code = 0;
thread_local jmp_buf g_bench_jmp_buf;

extern int main(int argc, char ** argv);

static std::string read_pipe_to_string(int fd) {
    std::string out;
    char        buf[4096];
    ssize_t     n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<size_t>(n));
    }
    return out;
}

static std::string capture_stdio(int (*fn)(int, char **), int argc, char ** argv) {
    int stdout_pipe[2];
    int stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        return "error: failed to create capture pipes\n";
    }

    const int saved_stdout = dup(STDOUT_FILENO);
    const int saved_stderr = dup(STDERR_FILENO);

    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    g_bench_exit_code = 0;
    if (setjmp(g_bench_jmp_buf) == 0) {
        fn(argc, argv);
    }

    fflush(stdout);
    fflush(stderr);
    dup2(saved_stdout, STDOUT_FILENO);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stdout);
    close(saved_stderr);

    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    std::string result = read_pipe_to_string(stdout_pipe[0]);
    close(stdout_pipe[0]);

    const std::string err = read_pipe_to_string(stderr_pipe[0]);
    close(stderr_pipe[0]);

    if (!err.empty()) {
        result += "\n--- stderr ---\n";
        result += err;
    }
    if (g_bench_exit_code != 0) {
        result += "\nllama-bench exited with code " + std::to_string(g_bench_exit_code) + "\n";
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_llamabench_internal_BenchEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring native_lib_dir) {
    llama_log_set(bench_android_log_callback, nullptr);

#if defined(BENCH_VULKAN_BACKEND)
    (void) native_lib_dir;
    (void) env;
    LOGi("Loading statically linked GGML backends (Vulkan + CPU)");
    ggml_backend_load_all();
#else
    const auto *path_to_backend = env->GetStringUTFChars(native_lib_dir, 0);
    LOGi("Loading CPU backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(native_lib_dir, path_to_backend);
#endif
    LOGi("GGML backends loaded");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_llamabench_internal_BenchEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_llamabench_internal_BenchEngineImpl_runBenchNative(JNIEnv *env, jobject /*unused*/, jstring jmodel_path) {
    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);

    std::vector<std::string> arg_storage = {
        "llama-bench",
        "-m", model_path,
        "-p", "64",
        "-n", "32",
        "-ngl", "99",
        "-r", "20",
    };
    env->ReleaseStringUTFChars(jmodel_path, model_path);

    std::vector<char *> argv;
    argv.reserve(arg_storage.size());
    for (auto & s : arg_storage) {
        argv.push_back(s.data());
    }

    LOGi("Running llama-bench: -p 64 -n 32 -ngl 99 -r 20");
    const std::string output = capture_stdio(main, static_cast<int>(argv.size()), argv.data());
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_llamabench_internal_BenchEngineImpl_shutdown(JNIEnv * /*env*/, jobject /*unused*/) {
    llama_backend_free();
}
