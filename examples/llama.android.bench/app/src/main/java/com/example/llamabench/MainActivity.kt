package com.example.llamabench

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arm.llamabench.BenchEngine
import com.arm.llamabench.LlamaBench
import com.arm.llamabench.gguf.GgufMetadata
import com.arm.llamabench.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var ggufTv: TextView
    private lateinit var benchResultTv: TextView
    private lateinit var statusTv: TextView
    private lateinit var userActionFab: FloatingActionButton

    private lateinit var engine: BenchEngine
    private var benchJobRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        ggufTv = findViewById(R.id.gguf)
        benchResultTv = findViewById(R.id.bench_result)
        statusTv = findViewById(R.id.status)
        userActionFab = findViewById(R.id.fab)

        lifecycleScope.launch(Dispatchers.Default) {
            engine = LlamaBench.getBenchEngine(applicationContext)
        }

        userActionFab.setOnClickListener {
            if (benchJobRunning) {
                Toast.makeText(this, "Benchmark already running", Toast.LENGTH_SHORT).show()
            } else {
                getContent.launch(arrayOf("*/*"))
            }
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        Log.i(TAG, "Selected file uri:\n $uri")
        uri?.let { handleSelectedModel(it) }
    }

    private fun handleSelectedModel(uri: Uri) {
        userActionFab.isEnabled = false
        statusTv.text = getString(R.string.parsing_gguf)
        ggufTv.text = "Parsing metadata from selected file\n$uri"
        benchResultTv.text = ""

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val metadata = contentResolver.openInputStream(uri)?.use {
                    GgufMetadataReader.create().readStructuredMetadata(it)
                } ?: throw IllegalStateException("Could not read GGUF file")

                withContext(Dispatchers.Main) {
                    ggufTv.text = metadata.toString()
                }

                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                val modelFile = contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                } ?: throw IllegalStateException("Could not copy model file")

                runBenchmark(modelName, modelFile)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to run benchmark", e)
                withContext(Dispatchers.Main) {
                    benchResultTv.text = "Error: ${e.message}"
                    statusTv.text = getString(R.string.bench_failed)
                    userActionFab.isEnabled = true
                }
            }
        }
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                if (!file.exists()) {
                    Log.i(TAG, "Copying file to $modelName")
                    withContext(Dispatchers.Main) {
                        statusTv.text = getString(R.string.copying_file)
                    }
                    FileOutputStream(file).use { input.copyTo(it) }
                }
            }
        }

    private suspend fun runBenchmark(modelName: String, modelFile: File) {
        benchJobRunning = true
        withContext(Dispatchers.Main) {
            statusTv.text = getString(R.string.running_bench, modelName)
            benchResultTv.text = getString(R.string.bench_running)
        }

        withContext(Dispatchers.Default) {
            engine.state.filter { it is BenchEngine.State.Ready || it is BenchEngine.State.Error }.first()
        }
        check(engine.state.value is BenchEngine.State.Ready) {
            "Engine not ready: ${engine.state.value}"
        }

        val result = withContext(Dispatchers.Default) {
            engine.runBench(modelFile.absolutePath)
        }

        withContext(Dispatchers.Main) {
            benchResultTv.text = result
            statusTv.text = getString(R.string.bench_done)
            userActionFab.isEnabled = true
            benchJobRunning = false
        }
    }

    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) {
                it.delete()
            }
            if (!it.exists()) {
                it.mkdir()
            }
        }

    override fun onDestroy() {
        if (::engine.isInitialized) {
            engine.destroy()
        }
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"
    }
}

private fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size -> "$name-$size" } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid -> "$arch-$uuid" } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> "model-${System.currentTimeMillis().toString(16)}"
}
