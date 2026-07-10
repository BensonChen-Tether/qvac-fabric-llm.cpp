package com.example.llama.bench

import android.content.Context
import org.json.JSONObject

data class DeviceFarmBenchConfig(
    val automation: Boolean = false,
    val modelPath: String? = null,
    val modelDownloadUrl: String? = null,
    val nGpuLayers: Int? = null,
    val repetitions: Int? = null,
    val skipDownload: Boolean? = null,
) {
    companion object {
        fun loadFromAssets(context: Context): DeviceFarmBenchConfig? {
            return try {
                context.assets.open("devicefarm_bench.json").use { stream ->
                    val json = JSONObject(stream.bufferedReader().readText())
                    DeviceFarmBenchConfig(
                        automation = json.optBoolean("automation", false),
                        modelPath = json.optString("model_path", null),
                        modelDownloadUrl = json.optString("model_download_url", null).takeIf { it.isNotEmpty() },
                        nGpuLayers = if (json.has("n_gpu_layers")) json.getInt("n_gpu_layers") else null,
                        repetitions = if (json.has("repetitions")) json.getInt("repetitions") else null,
                        skipDownload = if (json.has("skip_download")) json.getBoolean("skip_download") else null,
                    )
                }
            } catch (_: Exception) {
                null
            }
        }
    }
}
