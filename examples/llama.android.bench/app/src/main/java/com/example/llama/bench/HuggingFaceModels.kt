package com.example.llama.bench

object HuggingFaceModels {
    const val REPO_ID = "Benson-Chen/tether-gguf-models"
    const val REVISION = "main"

    data class Entry(
        val pathInRepo: String,
        val label: String,
    ) {
        val family: String = pathInRepo.substringBefore('/')
    }

    val models = listOf(
        Entry("qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf", "Qwen3 0.6B Q4_K_M"),
        Entry("qwen3-0.6B/Qwen3-0.6B-TQ2_0_Tether.gguf", "Qwen3 0.6B TQ2_0"),
        Entry("qwen3-0.6B/Qwen3-0.6B-TQ2_0.gguf", "Qwen3 0.6B TQ2_0 (alt)"),
        Entry("qwen3-1.7B/Qwen3-1.7B-Q2_K.gguf", "Qwen3 1.7B Q2_K"),
        Entry("qwen3-1.7B/Qwen3-1.7B-Q3_K_M.gguf", "Qwen3 1.7B Q3_K_M"),
        Entry("qwen3-1.7B/Qwen3-1.7B-Q4_0.gguf", "Qwen3 1.7B Q4_0"),
        Entry("qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf", "Qwen3 1.7B Q4_K_M"),
        Entry("qwen3-1.7B/Qwen3-1.7B-TQ1_0.gguf", "Qwen3 1.7B TQ1_0"),
        Entry("qwen3-1.7B/Qwen3-1.7B-TQ2_0.gguf", "Qwen3 1.7B TQ2_0"),
        Entry("bonsai-1.7B/Bonsai-1.7B-Q2_0.gguf", "Bonsai 1.7B Q2_0"),
    )

    val families: List<String> = models.map { it.family }.distinct()

    fun modelsForFamily(family: String): List<Entry> =
        models.filter { it.family == family }

    fun downloadUrl(pathInRepo: String): String =
        "https://huggingface.co/$REPO_ID/resolve/$REVISION/$pathInRepo"

    fun localFileName(pathInRepo: String): String =
        pathInRepo.substringAfterLast('/')
}
