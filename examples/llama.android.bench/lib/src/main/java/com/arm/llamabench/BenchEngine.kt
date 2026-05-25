package com.arm.llamabench

import kotlinx.coroutines.flow.StateFlow

/**
 * Runs llama-bench on a loaded GGUF model path.
 */
interface BenchEngine {
    val state: StateFlow<State>

    suspend fun runBench(modelPath: String): String

    fun destroy()

    sealed class State {
        object Uninitialized : State()
        object Initializing : State()
        object Ready : State()
        object Running : State()
        data class Error(val exception: Exception) : State()
    }
}
