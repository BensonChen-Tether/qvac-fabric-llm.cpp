package com.example.llama.bench

import android.Manifest
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BenchmarkInstrumentedTest {

    @get:Rule
    val grantPermissionRule: GrantPermissionRule = GrantPermissionRule.grant(
        Manifest.permission.INTERNET,
    )

    @Test
    fun runSelectedModelBenchmark() {
        runBlocking(Dispatchers.IO) {
            val args = InstrumentationRegistry.getArguments()
            val modelPath = args.getString("model_path")
                ?: DEFAULT_MODEL_PATH
            val repetitions = args.getString("repetitions")?.toIntOrNull()
                ?: BenchConfig.AUTOMATION_REPETITIONS
            val nGpuLayers = args.getString("n_gpu_layers")?.toIntOrNull()
                ?: BenchConfig.DEFAULT_N_GPU_LAYERS
            val skipDownload = args.getString("skip_download")?.toBoolean() ?: true

            val result = BenchmarkAutomation.run(
                context = InstrumentationRegistry.getInstrumentation().targetContext,
                modelPathInRepo = modelPath,
                repetitions = repetitions,
                nGpuLayers = nGpuLayers,
                skipDownloadIfCached = skipDownload,
            )

            assertTrue(
                "Expected llama-bench JSON output",
                result.benchOutput.contains("{") || result.benchOutput.contains("["),
            )
            assertTrue("Expected result file", result.resultFile.exists())
        }
    }

    companion object {
        private const val DEFAULT_MODEL_PATH = "qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf"
    }
}
