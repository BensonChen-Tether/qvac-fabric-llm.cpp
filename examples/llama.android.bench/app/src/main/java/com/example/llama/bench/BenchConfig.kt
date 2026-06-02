package com.example.llama.bench

data class BenchConfig(
    val promptTokens: Int = DEFAULT_PROMPT_TOKENS,
    val genTokens: Int = DEFAULT_GEN_TOKENS,
    val repetitions: Int = DEFAULT_REPETITIONS,
    val nGpuLayers: Int = DEFAULT_N_GPU_LAYERS,
    val jsonOutput: Boolean = false,
) {
    companion object {
        const val DEFAULT_PROMPT_TOKENS = 64
        const val DEFAULT_GEN_TOKENS = 32
        const val DEFAULT_REPETITIONS = 20
        const val DEFAULT_N_GPU_LAYERS = 999
        const val AUTOMATION_REPETITIONS = 5
    }
}
