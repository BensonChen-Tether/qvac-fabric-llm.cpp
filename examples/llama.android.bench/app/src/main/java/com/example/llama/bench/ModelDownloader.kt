package com.example.llama.bench

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelDownloader {
    private val TAG = ModelDownloader::class.java.simpleName

    fun download(
        url: String,
        destination: File,
        onProgress: (bytesRead: Long, totalBytes: Long) -> Unit,
    ) {
        destination.parentFile?.let { parent ->
            if (!parent.exists()) {
                parent.mkdirs()
            }
        }

        val tempFile = File(destination.parentFile, "${destination.name}.part")
        if (tempFile.exists()) {
            tempFile.delete()
        }

        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 30_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        }

        try {
            connection.connect()
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                throw IllegalStateException("Download failed with HTTP $responseCode")
            }

            val totalBytes = connection.contentLengthLong
            Log.i(TAG, "Downloading $url (${formatBytes(totalBytes)})")

            connection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var bytesRead = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) {
                            break
                        }
                        output.write(buffer, 0, read)
                        bytesRead += read
                        onProgress(bytesRead, totalBytes)
                    }
                    output.fd.sync()
                }
            }

            if (destination.exists()) {
                destination.delete()
            }
            if (!tempFile.renameTo(destination)) {
                tempFile.copyTo(destination, overwrite = true)
                tempFile.delete()
            }
        } finally {
            connection.disconnect()
            if (tempFile.exists() && !destination.exists()) {
                tempFile.delete()
            }
        }
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes < 0 -> "unknown size"
        bytes >= 1_073_741_824 -> String.format("%.1f GB", bytes / 1_073_741_824.0)
        bytes >= 1_048_576 -> String.format("%.1f MB", bytes / 1_048_576.0)
        else -> "$bytes B"
    }
}
