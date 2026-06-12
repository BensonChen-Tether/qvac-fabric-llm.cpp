import Foundation

enum BenchmarkConfig {
    static let repoId = "Benson-Chen/tether-gguf-models"
    static let promptTokens = 64
    static let genTokens = 32
    static let automationRepetitions = 5
    static let defaultNGpuLayers = 99
    static let cpuNGpuLayers = 0

    static func downloadURL(pathInRepo: String) -> URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(pathInRepo)")!
    }

    static func localFileName(pathInRepo: String) -> String {
        (pathInRepo as NSString).lastPathComponent
    }
}
