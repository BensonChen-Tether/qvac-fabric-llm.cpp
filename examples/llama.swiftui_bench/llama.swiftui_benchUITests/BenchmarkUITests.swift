import XCTest

final class BenchmarkUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testRunModelBenchmark() throws {
        let app = XCUIApplication()

        app.launchEnvironment["BENCHMARK_AUTOMATION"] = "1"
        app.launch()

        let complete = app.staticTexts["BENCH_COMPLETE"]
        let failed = app.staticTexts["BENCH_FAILED"]
        let deadline = Date().addingTimeInterval(3_600)

        while Date() < deadline {
            if complete.exists {
                return
            }
            if failed.exists {
                XCTFail("Benchmark automation failed on device (see LLAMA_BENCH_ERROR in logs)")
            }
            RunLoop.current.run(until: Date().addingTimeInterval(1))
        }

        XCTFail("Benchmark did not complete within 60 minutes")
    }
}
