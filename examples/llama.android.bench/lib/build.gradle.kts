plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.jetbrains.kotlin.android)
}

val vulkanInclude: String = listOf("/usr/local/include", "/opt/homebrew/include")
    .firstOrNull {
        file("$it/vulkan/vulkan.hpp").exists() && file("$it/spirv/unified1/spirv.hpp").exists()
    }
    ?: System.getenv("VULKAN_SDK")?.let { "$it/include" }
    ?: listOf("/opt/homebrew/include", "/usr/local/include")
        .firstOrNull { file("$it/vulkan/vulkan.hpp").exists() }
    ?: "/usr/local/include"

val glslcPath: String = System.getenv("VULKAN_GLSLC")
    ?: listOf("/usr/local/bin/glslc", "/opt/homebrew/bin/glslc")
        .firstOrNull { file(it).exists() }
    ?: "glslc"

android {
    namespace = "com.arm.aichat"
    compileSdk = 36

    ndkVersion = "29.0.13113456"

    defaultConfig {
        minSdk = 33

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
             abiFilters += listOf("arm64-v8a", "x86_64")
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
                arguments += "-DGGML_BACKEND_DL=ON"
                arguments += "-DGGML_CPU_ALL_VARIANTS=ON"
                arguments += "-DGGML_LLAMAFILE=OFF"

                // GPU offload via Vulkan (required for -ngl to use GPU on Android).
                arguments += "-DGGML_VULKAN=ON"
                arguments += "-DGGML_VULKAN_BUILD_ADRENO_SHADERS=ON"
                arguments += "-DVulkan_INCLUDE_DIR=$vulkanInclude"
                arguments += "-DVulkan_GLSLC_EXECUTABLE=$glslcPath"
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
