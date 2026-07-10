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
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val bundled = DeviceFarmBenchConfig.loadFromAssets(context)

            val repetitions = bundled?.repetitions
                ?: args.getString("repetitions")?.toIntOrNull()
                ?: BenchConfig.AUTOMATION_REPETITIONS
            val skipDownload = bundled?.skipDownload
                ?: args.getString("skip_download")?.toBoolean()
                ?: true
            val runCpuAndGpu = args.getString("run_cpu_and_gpu")?.toBoolean() ?: false
            val downloadURL = bundled?.modelDownloadUrl

            val modelPaths = bundled?.modelPath?.let { listOf(it) } ?: parseModelPaths(args)
            val nGpuLayersList = when {
                bundled?.nGpuLayers != null -> listOf(bundled.nGpuLayers)
                runCpuAndGpu -> listOf(BenchConfig.DEFAULT_N_GPU_LAYERS, 0)
                else -> listOf(
                    args.getString("n_gpu_layers")?.toIntOrNull()
                        ?: BenchConfig.DEFAULT_N_GPU_LAYERS,
                )
            }

            val results = if (modelPaths.size > 1 || runCpuAndGpu) {
                BenchmarkAutomation.runMatrix(
                    context = context,
                    modelPathsInRepo = modelPaths,
                    nGpuLayersList = nGpuLayersList,
                    repetitions = repetitions,
                    skipDownloadIfCached = skipDownload,
                )
            } else {
                listOf(
                    BenchmarkAutomation.run(
                        context = context,
                        modelPathInRepo = modelPaths.first(),
                        repetitions = repetitions,
                        nGpuLayers = nGpuLayersList.first(),
                        skipDownloadIfCached = skipDownload,
                        downloadURL = downloadURL,
                    ),
                )
            }

            val failures = results.filter { result ->
                !(result.benchOutput.contains("{") || result.benchOutput.contains("["))
            }
            assertTrue(
                "Benchmark failed for: ${failures.map { it.modelPathInRepo }.joinToString()}",
                failures.isEmpty(),
            )
        }
    }

    private fun parseModelPaths(args: android.os.Bundle): List<String> {
        args.getString("model_paths")
            ?.split(';')
            ?.map { it.trim() }
            ?.filter { it.isNotEmpty() }
            ?.takeIf { it.isNotEmpty() }
            ?.let { return it }

        val single = args.getString("model_path") ?: DEFAULT_MODEL_PATH
        return listOf(single)
    }

    companion object {
        private const val DEFAULT_MODEL_PATH = "qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf"
    }
}
