import XCTest

final class BenchmarkUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testRunModelBenchmark() throws {
        let app = XCUIApplication()
        let env = ProcessInfo.processInfo.environment

        app.launchEnvironment["BENCHMARK_AUTOMATION"] = "1"
        app.launchEnvironment["model_path"] = env["model_path"]
            ?? "qwen3-0.6B/Qwen3-0.6B-TQ2_0_Tether.gguf"
        app.launchEnvironment["n_gpu_layers"] = env["n_gpu_layers"] ?? "99"
        app.launchEnvironment["repetitions"] = env["repetitions"] ?? "5"
        app.launchEnvironment["skip_download"] = env["skip_download"] ?? "true"

        app.launch()

        let complete = app.staticTexts["BENCH_COMPLETE"]
        XCTAssertTrue(
            complete.waitForExistence(timeout: 3_600),
            "Benchmark did not complete within 60 minutes"
        )
    }
}
