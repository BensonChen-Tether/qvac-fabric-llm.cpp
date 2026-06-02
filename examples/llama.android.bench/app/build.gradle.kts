plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

android {
    namespace = "com.example.llama.bench"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.llama.bench"

        minSdk = 33
        targetSdk = 36

        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        testInstrumentationRunnerArguments["model_path"] = "qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf"
        testInstrumentationRunnerArguments["repetitions"] = "5"
        testInstrumentationRunnerArguments["skip_download"] = "true"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        debug {
            // Debug APK is used for Firebase Test Lab / instrumentation — keep unobfuscated.
            isMinifyEnabled = false
            isShrinkResources = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        jvmToolchain(17)
    }

    androidResources {
        noCompress += "llama-bench"
    }
}

dependencies {
    implementation(libs.bundles.androidx)
    implementation(libs.material)
    implementation(libs.kotlinx.coroutines.android)

    implementation(project(":lib"))

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.kotlinx.coroutines.android)
}
