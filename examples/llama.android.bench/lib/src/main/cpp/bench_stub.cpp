#include <jni.h>

// Stub shared library to ensure llama/ggml backend .so files are packaged in the APK.
extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM * /*vm*/, void * /*reserved*/) {
    return JNI_VERSION_1_6;
}
