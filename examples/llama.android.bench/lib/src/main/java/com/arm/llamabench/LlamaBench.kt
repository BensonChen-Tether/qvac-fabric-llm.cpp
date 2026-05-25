package com.arm.llamabench

import android.content.Context
import com.arm.llamabench.internal.BenchEngineImpl

object LlamaBench {
    fun getBenchEngine(context: Context): BenchEngine = BenchEngineImpl.getInstance(context)
}
