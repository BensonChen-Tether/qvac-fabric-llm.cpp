package com.arm.llamabench.internal

import android.content.Context
import android.util.Log
import com.arm.llamabench.BenchEngine
import dalvik.annotation.optimization.FastNative
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

internal class BenchEngineImpl private constructor(
    private val nativeLibDir: String,
) : BenchEngine {

    companion object {
        private val TAG = BenchEngineImpl::class.java.simpleName

        @Volatile
        private var instance: BenchEngine? = null

        fun getInstance(context: Context): BenchEngine =
            instance ?: synchronized(this) {
                val nativeLibDir = context.applicationInfo.nativeLibraryDir
                require(nativeLibDir.isNotBlank()) { "Expected a valid native library path!" }
                BenchEngineImpl(nativeLibDir).also { instance = it }
            }
    }

    @FastNative
    private external fun init(nativeLibDir: String)

    @FastNative
    private external fun systemInfo(): String

    @FastNative
    private external fun runBenchNative(modelPath: String): String

    @FastNative
    private external fun shutdown()

    private val _state = MutableStateFlow<BenchEngine.State>(BenchEngine.State.Uninitialized)
    override val state: StateFlow<BenchEngine.State> = _state.asStateFlow()

    @OptIn(ExperimentalCoroutinesApi::class)
    private val benchDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val benchScope = CoroutineScope(benchDispatcher + SupervisorJob())

    init {
        benchScope.launch {
            try {
                _state.value = BenchEngine.State.Initializing
                System.loadLibrary("llama-bench")
                init(nativeLibDir)
                _state.value = BenchEngine.State.Ready
                Log.i(TAG, "Native library loaded\n${systemInfo()}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load native library", e)
                _state.value = BenchEngine.State.Error(e)
                throw e
            }
        }
    }

    override suspend fun runBench(modelPath: String): String = withContext(benchDispatcher) {
        check(_state.value is BenchEngine.State.Ready) {
            "Cannot run bench in ${_state.value.javaClass.simpleName}"
        }

        File(modelPath).let {
            require(it.exists()) { "File not found" }
            require(it.isFile) { "Not a valid file" }
            require(it.canRead()) { "Cannot read file" }
        }

        try {
            _state.value = BenchEngine.State.Running
            Log.i(TAG, "Running llama-bench on $modelPath")
            runBenchNative(modelPath).also {
                _state.value = BenchEngine.State.Ready
            }
        } catch (e: Exception) {
            Log.e(TAG, "Benchmark failed", e)
            _state.value = BenchEngine.State.Error(e)
            throw e
        }
    }

    override fun destroy() {
        runBlocking(benchDispatcher) {
            shutdown()
        }
        benchScope.cancel()
    }
}
