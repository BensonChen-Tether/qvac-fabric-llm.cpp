import Foundation

struct DeviceFarmBenchConfig: Decodable {
    let automation: Bool?
    let modelPath: String?
    let modelDownloadURL: String?
    let nGpuLayers: Int?
    let repetitions: Int?
    let skipDownload: Bool?

    enum CodingKeys: String, CodingKey {
        case automation
        case modelPath = "model_path"
        case modelDownloadURL = "model_download_url"
        case nGpuLayers = "n_gpu_layers"
        case repetitions
        case skipDownload = "skip_download"
    }

    static func loadFromBundle() -> DeviceFarmBenchConfig? {
        guard let url = Bundle.main.url(forResource: "devicefarm_bench", withExtension: "json"),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? JSONDecoder().decode(DeviceFarmBenchConfig.self, from: data)
    }
}
