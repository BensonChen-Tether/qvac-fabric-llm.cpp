plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.jetbrains.kotlin.android)
}

import java.io.File

val useVulkan = providers.gradleProperty("llamaBench.backend")
    .map { it.equals("vulkan", ignoreCase = true) }
    .getOrElse(true)

fun firstExisting(vararg paths: String): String? =
    paths.firstOrNull { File(it).exists() }

fun vulkanIncludeDir(): String =
    System.getenv("VULKAN_SDK")?.let { "$it/Include" }?.takeIf { File(it).isDirectory }
        ?: firstExisting("/opt/homebrew/include", "/usr/local/include")
        ?: "/opt/homebrew/include"

fun vulkanGlslc(): String =
    System.getenv("VULKAN_SDK")?.let { "$it/bin/glslc" }?.takeIf { File(it).canExecute() }
        ?: firstExisting("/usr/local/bin/glslc", "/opt/homebrew/bin/glslc")
        ?: "glslc"

android {
    namespace = "com.arm.llamabench"
    compileSdk = 36

    ndkVersion = "27.0.12077973"

    defaultConfig {
        minSdk = 33

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += if (useVulkan) {
                listOf("arm64-v8a")
            } else {
                listOf("arm64-v8a", "x86_64")
            }
        }
        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DCMAKE_MESSAGE_LOG_LEVEL=DEBUG"
                arguments += "-DCMAKE_VERBOSE_MAKEFILE=ON"

                arguments += "-DBUILD_SHARED_LIBS=ON"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DLLAMA_OPENSSL=OFF"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_LLAMAFILE=OFF"

                if (useVulkan) {
                    arguments += "-DGGML_VULKAN=ON"
                    arguments += "-DGGML_BACKEND_DL=OFF"
                    arguments += "-DGGML_CPU_ALL_VARIANTS=OFF"
                    arguments += "-DGGML_OPENMP=OFF"
                    arguments += "-DGGML_VULKAN_BUILD_ADRENO_SHADERS=ON"
                    arguments += "-DVulkan_INCLUDE_DIR=${vulkanIncludeDir()}"
                    arguments += "-DVulkan_GLSLC_EXECUTABLE=${vulkanGlslc()}"
                    arguments += "-DCMAKE_C_FLAGS=-march=armv8.7a"
                    arguments += "-DCMAKE_CXX_FLAGS=-march=armv8.7a"
                } else {
                    arguments += "-DGGML_BACKEND_DL=ON"
                    arguments += "-DGGML_CPU_ALL_VARIANTS=ON"
                }
            }
        }
        aarMetadata {
            minCompileSdk = 35
        }
    }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        jvmToolchain(17)

        compileOptions {
            targetCompatibility = JavaVersion.VERSION_17
        }
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    publishing {
        singleVariant("release") {
            withJavadocJar()
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.datastore.preferences)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
}
