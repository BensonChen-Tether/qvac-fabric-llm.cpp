package com.example.llama.bench

import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

object BenchmarkAutomation {
    const val RESULT_LOG_TAG = "LLAMA_BENCH_RESULT"
    const val META_LOG_TAG = "LLAMA_BENCH_META"
    private val TAG = BenchmarkAutomation::class.java.simpleName
    private const val MODELS_DIR = "models"
    private const val RESULT_FILE = "benchmark_result.json"
    private const val META_FILE = "benchmark_meta.json"

    data class RunResult(
        val modelPathInRepo: String,
        val modelFile: File,
        val benchOutput: String,
        val resultFile: File,
        val metaFile: File,
    )

    suspend fun run(
        context: Context,
        modelPathInRepo: String,
        repetitions: Int = BenchConfig.AUTOMATION_REPETITIONS,
        nGpuLayers: Int = BenchConfig.DEFAULT_N_GPU_LAYERS,
        skipDownloadIfCached: Boolean = true,
    ): RunResult = withContext(Dispatchers.IO) {
        val modelFile = ensureModel(context, modelPathInRepo, skipDownloadIfCached)
        val config = BenchConfig(
            repetitions = repetitions,
            nGpuLayers = nGpuLayers,
            jsonOutput = true,
        )

        val benchOutput = LlamaBenchRunner.run(context, modelFile.absolutePath, config)
        val meta = buildMeta(context, modelPathInRepo, modelFile, repetitions, nGpuLayers)
        val metaFile = writeText(context, META_FILE, meta.toString())
        val resultFile = writeText(context, RESULT_FILE, benchOutput.trim())

        Log.i(META_LOG_TAG, meta.toString())
        Log.i(RESULT_LOG_TAG, benchOutput.trim())

        RunResult(
            modelPathInRepo = modelPathInRepo,
            modelFile = modelFile,
            benchOutput = benchOutput,
            resultFile = resultFile,
            metaFile = metaFile,
        )
    }

    private suspend fun ensureModel(
        context: Context,
        modelPathInRepo: String,
        skipDownloadIfCached: Boolean,
    ): File {
        val modelsDir = File(context.filesDir, MODELS_DIR).also { it.mkdirs() }
        val modelFile = File(modelsDir, HuggingFaceModels.localFileName(modelPathInRepo))
        if (skipDownloadIfCached && modelFile.exists() && modelFile.length() > 0) {
            Log.i(TAG, "Using cached model: ${modelFile.name}")
            return modelFile
        }

        ModelDownloader.download(
            url = HuggingFaceModels.downloadUrl(modelPathInRepo),
            destination = modelFile,
            onProgress = { _, _ -> },
        )
        return modelFile
    }

    private fun buildMeta(
        context: Context,
        modelPathInRepo: String,
        modelFile: File,
        repetitions: Int,
        nGpuLayers: Int,
    ): JSONObject {
        return JSONObject().apply {
            put("model_path", modelPathInRepo)
            put("model_file", modelFile.name)
            put("model_bytes", modelFile.length())
            put("repetitions", repetitions)
            put("prompt_tokens", BenchConfig.DEFAULT_PROMPT_TOKENS)
            put("gen_tokens", BenchConfig.DEFAULT_GEN_TOKENS)
            put("n_gpu_layers", nGpuLayers)
            put("repo_id", HuggingFaceModels.REPO_ID)
            put("device_model", Build.MODEL)
            put("device", Build.DEVICE)
            put("manufacturer", Build.MANUFACTURER)
            put("android_release", Build.VERSION.RELEASE)
            put("sdk_int", Build.VERSION.SDK_INT)
            put("supported_abis", Build.SUPPORTED_ABIS.joinToString(","))
            put("package", context.packageName)
        }
    }

    private fun writeText(context: Context, fileName: String, content: String): File {
        val file = File(context.filesDir, fileName)
        file.writeText(content)
        return file
    }
}
