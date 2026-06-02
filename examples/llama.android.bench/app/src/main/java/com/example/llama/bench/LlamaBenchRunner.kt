package com.example.llama.bench

import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader

object LlamaBenchRunner {
    private val TAG = LlamaBenchRunner::class.java.simpleName

    private const val ASSET_BIN = "bin"
    private const val EXEC_NAME = "llama-bench"

    suspend fun run(
        context: Context,
        modelPath: String,
        config: BenchConfig = BenchConfig(),
    ): String = withContext(Dispatchers.IO) {
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        val benchBin = ensureExecutable(context)

        // Android cannot exec binaries from app storage directly; run via the dynamic linker.
        val command = buildList {
            add(linkerPath())
            add(benchBin.absolutePath)
            addAll(listOf(
                "-m", modelPath,
                "-p", config.promptTokens.toString(),
                "-n", config.genTokens.toString(),
                "-r", config.repetitions.toString(),
                "-ngl", config.nGpuLayers.toString(),
            ))
            if (config.jsonOutput) {
                add("-o")
                add("json")
            }
        }
        Log.i(TAG, "Running: ${command.joinToString(" ")}")

        val process = ProcessBuilder(command)
            .directory(File(nativeLibDir))
            .redirectErrorStream(true)
            .apply {
                environment()["LD_LIBRARY_PATH"] = nativeLibDir
            }
            .start()

        val output = StringBuilder()
        BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
            var line = reader.readLine()
            while (line != null) {
                output.appendLine(line)
                line = reader.readLine()
            }
        }

        val exitCode = process.waitFor()
        if (exitCode != 0) {
            output.appendLine()
            output.appendLine("llama-bench exited with code $exitCode")
        }
        output.toString()
    }

    private fun linkerPath(): String {
        listOf(
            "/apex/com.android.runtime/bin/linker64",
            "/system/bin/linker64",
        ).forEach { path ->
            if (File(path).canExecute()) {
                return path
            }
        }
        return "/system/bin/linker64"
    }

    private fun ensureExecutable(context: Context): File {
        val abi = Build.SUPPORTED_ABIS.firstOrNull()
            ?: throw IllegalStateException("No supported ABI found on this device")
        val assetPath = "$ASSET_BIN/$abi/$EXEC_NAME"
        val outDir = File(context.filesDir, "bin").also { it.mkdirs() }
        val outFile = File(outDir, EXEC_NAME)

        if (outFile.exists() && outFile.canExecute()) {
            return outFile
        }

        context.assets.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        Os.chmod(outFile.absolutePath, 493) // 0755
        return outFile
    }
}
