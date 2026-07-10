import Foundation

enum BenchmarkConfig {
    static let repoId = "Benson-Chen/tether-gguf-models"
    static let s3Bucket = "tether-ai-dev"
    static let s3ModelPrefix = "models/qwen3-checkpoints/gguf/"
    static let promptTokens = 64
    static let genTokens = 32
    static let automationRepetitions = 5
    static let defaultNGpuLayers = 99
    static let cpuNGpuLayers = 0

    static func downloadURL(pathInRepo: String, presignedURL: String? = nil) -> URL {
        if let presignedURL,
           let url = URL(string: presignedURL),
           !presignedURL.isEmpty {
            return url
        }
        return URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(pathInRepo)")!
    }

    static func downloadSource(presignedURL: String?) -> String {
        if let presignedURL, !presignedURL.isEmpty {
            return "s3_presigned"
        }
        return "huggingface"
    }

    static func localFileName(pathInRepo: String) -> String {
        (pathInRepo as NSString).lastPathComponent
    }
}
