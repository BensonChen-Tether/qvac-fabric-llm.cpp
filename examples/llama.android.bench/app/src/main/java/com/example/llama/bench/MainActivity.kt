package com.example.llama.bench

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var statusTv: TextView
    private lateinit var outputTv: TextView
    private lateinit var outputScroll: ScrollView
    private lateinit var actionFab: FloatingActionButton
    private lateinit var downloadFab: FloatingActionButton

    private var modelFile: File? = null
    private var benchJob: Job? = null
    private var downloadJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        statusTv = findViewById(R.id.status)
        outputTv = findViewById(R.id.output)
        outputScroll = findViewById(R.id.output_scroll)
        actionFab = findViewById(R.id.fab)
        downloadFab = findViewById(R.id.fab_download)

        actionFab.setOnClickListener {
            if (modelFile != null) {
                runBenchmark()
            } else {
                getContent.launch(arrayOf("*/*"))
            }
        }

        downloadFab.setOnClickListener {
            showDownloadDialog()
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        Log.i(TAG, "Selected file uri:\n $uri")
        uri?.let { handleSelectedModel(it) }
    }

    private fun showDownloadDialog() {
        if (downloadJob?.isActive == true) {
            Toast.makeText(this, "Download already in progress", Toast.LENGTH_SHORT).show()
            return
        }

        val families = HuggingFaceModels.families.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle(R.string.choose_model_family)
            .setItems(families) { _, which ->
                showVariantPicker(HuggingFaceModels.families[which])
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun showVariantPicker(family: String) {
        val variants = HuggingFaceModels.modelsForFamily(family)
        if (variants.isEmpty()) {
            Toast.makeText(this, "No models found for $family", Toast.LENGTH_SHORT).show()
            return
        }

        var selectedIndex = 0
        val labels = variants.map { entry -> formatModelOptionLabel(entry) }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle(getString(R.string.choose_model_variant) + " ($family)")
            .setSingleChoiceItems(labels, selectedIndex) { _, which ->
                selectedIndex = which
            }
            .setPositiveButton(R.string.download) { _, _ ->
                downloadModel(variants[selectedIndex])
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun formatModelOptionLabel(entry: HuggingFaceModels.Entry): String {
        val fileName = HuggingFaceModels.localFileName(entry.pathInRepo)
        val cached = isModelCached(entry)
        return if (cached) {
            "${entry.label}\n$fileName (already on device)"
        } else {
            "${entry.label}\n$fileName"
        }
    }

    private fun isModelCached(entry: HuggingFaceModels.Entry): Boolean {
        val file = File(ensureModelsDirectory(), HuggingFaceModels.localFileName(entry.pathInRepo))
        return file.exists() && file.length() > 0
    }

    private fun downloadModel(entry: HuggingFaceModels.Entry) {
        val modelName = HuggingFaceModels.localFileName(entry.pathInRepo)
        val destination = File(ensureModelsDirectory(), modelName)
        val url = HuggingFaceModels.downloadUrl(entry.pathInRepo)

        setUiBusy(true)
        statusTv.text = "Preparing download...\n${entry.label}\n$url"
        outputTv.text = ""

        downloadJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                if (destination.exists() && destination.length() > 0) {
                    Log.i(TAG, "Model already downloaded: ${destination.name}")
                    withContext(Dispatchers.Main) {
                        statusTv.text = "Using cached model: ${destination.name}"
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        statusTv.text = "Downloading ${entry.label}...\n$url"
                    }
                    var lastProgressUpdateMs = 0L
                    ModelDownloader.download(url, destination) { bytesRead, totalBytes ->
                        val now = System.currentTimeMillis()
                        if (now - lastProgressUpdateMs < 250 && totalBytes > 0 && bytesRead < totalBytes) {
                            return@download
                        }
                        lastProgressUpdateMs = now
                        runOnUiThread {
                            statusTv.text = buildDownloadStatus(entry.label, bytesRead, totalBytes)
                        }
                    }
                }

                loadModelFromLocalFile(destination)
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                withContext(Dispatchers.Main) {
                    statusTv.append("\n\nDownload failed:\n${e.message}")
                    Toast.makeText(this@MainActivity, "Download failed", Toast.LENGTH_SHORT).show()
                    setUiBusy(false)
                }
            }
        }
    }

    private fun buildDownloadStatus(label: String, bytesRead: Long, totalBytes: Long): String {
        val progress = if (totalBytes > 0) {
            val percent = (bytesRead * 100 / totalBytes).toInt()
            "$percent% (${formatBytes(bytesRead)} / ${formatBytes(totalBytes)})"
        } else {
            formatBytes(bytesRead)
        }
        return "Downloading $label...\n$progress"
    }

    private fun handleSelectedModel(uri: Uri) {
        setUiBusy(true)
        statusTv.text = "Parsing GGUF metadata...\n$uri"
        outputTv.text = ""

        lifecycleScope.launch(Dispatchers.IO) {
            Log.i(TAG, "Parsing GGUF metadata...")
            contentResolver.openInputStream(uri)?.use {
                GgufMetadataReader.create().readStructuredMetadata(it)
            }?.let { metadata ->
                Log.i(TAG, "GGUF parsed: \n$metadata")
                withContext(Dispatchers.Main) {
                    statusTv.text = metadata.toString()
                }

                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                }?.let { file ->
                    loadModelFromLocalFile(file, metadata)
                }
            } ?: withContext(Dispatchers.Main) {
                Toast.makeText(this@MainActivity, "Failed to parse GGUF file", Toast.LENGTH_SHORT).show()
                setUiBusy(false)
            }
        }
    }

    private suspend fun loadModelFromLocalFile(file: File, metadata: GgufMetadata? = null) {
        try {
            val parsedMetadata = metadata ?: withContext(Dispatchers.IO) {
                file.inputStream().use {
                    GgufMetadataReader.create().readStructuredMetadata(it)
                }
            }

            modelFile = file
            withContext(Dispatchers.Main) {
                statusTv.text = parsedMetadata.toString()
                statusTv.append("\n\nModel ready: ${file.name}")
                actionFab.setImageResource(R.drawable.outline_play_arrow_24)
                setUiBusy(false)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model file", e)
            withContext(Dispatchers.Main) {
                statusTv.append("\n\nFailed to load model:\n${e.message}")
                Toast.makeText(this@MainActivity, "Failed to load model", Toast.LENGTH_SHORT).show()
                setUiBusy(false)
            }
        }
    }

    private fun runBenchmark() {
        val file = modelFile ?: return
        setUiBusy(true)
        statusTv.append("\n\nRunning llama-bench -p 64 -n 32 -r 20 -ngl 999...")
        outputTv.text = "Benchmark running...\n"

        benchJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = LlamaBenchRunner.run(applicationContext, file.absolutePath)
                withContext(Dispatchers.Main) {
                    outputTv.text = result
                    outputScroll.post { outputScroll.fullScroll(ScrollView.FOCUS_DOWN) }
                    setUiBusy(false)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark failed", e)
                withContext(Dispatchers.Main) {
                    outputTv.text = "Benchmark failed:\n${e.message}"
                    Toast.makeText(this@MainActivity, "Benchmark failed", Toast.LENGTH_SHORT).show()
                    setUiBusy(false)
                }
            }
        }
    }

    private fun setUiBusy(busy: Boolean) {
        actionFab.isEnabled = !busy
        downloadFab.isEnabled = !busy
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                if (!file.exists()) {
                    Log.i(TAG, "Start copying file to $modelName")
                    withContext(Dispatchers.Main) {
                        statusTv.append("\nCopying model file...")
                    }
                    FileOutputStream(file).use { input.copyTo(it) }
                    Log.i(TAG, "Finished copying file to $modelName")
                } else {
                    Log.i(TAG, "File already exists $modelName")
                }
            }
        }

    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) { it.delete() }
            if (!it.exists()) { it.mkdir() }
        }

    override fun onStop() {
        benchJob?.cancel()
        downloadJob?.cancel()
        super.onStop()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"

        private fun formatBytes(bytes: Long): String = when {
            bytes >= 1_073_741_824 -> String.format("%.1f GB", bytes / 1_073_741_824.0)
            bytes >= 1_048_576 -> String.format("%.1f MB", bytes / 1_048_576.0)
            else -> "$bytes B"
        }
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size ->
                "$name-$size"
            } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid ->
                "$arch-$uuid"
            } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> {
        "model-${System.currentTimeMillis().toHexString()}"
    }
}
