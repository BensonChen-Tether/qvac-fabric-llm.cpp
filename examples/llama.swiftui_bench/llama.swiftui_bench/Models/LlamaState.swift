import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

struct Dataset: Identifiable, Equatable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    @Published var selectedModelFilename: String?
    @Published var isFinetuning = false
    @Published var downloadedDatasets: [Dataset] = []
    @Published var availableDatasets: [Dataset] = []
    @Published var selectedDatasetFilename: String?
    @Published var optionNGpuLayers: Int = -1 {
        didSet {
            let clamped = max(-1, min(optionNGpuLayers, 256))
            if optionNGpuLayers != clamped {
                optionNGpuLayers = clamped
                return
            }
            UserDefaults.standard.set(optionNGpuLayers, forKey: runtimeNglKey)
            if optionNGpuLayers != oldValue {
                reloadModelWithCurrentOptionsIfPossible()
            }
        }
    }

    @Published var optionContextLength: Int = 2048 {
        didSet {
            let clamped = max(32, min(optionContextLength, 8192))
            if optionContextLength != clamped {
                optionContextLength = clamped
                return
            }
            UserDefaults.standard.set(optionContextLength, forKey: runtimeContextKey)
            if optionContextLength != oldValue {
                reloadModelWithCurrentOptionsIfPossible()
            }
        }
    }

    @Published var optionSeed: Int = 42 {
        didSet {
            let clamped = max(0, min(optionSeed, 1_000_000))
            if optionSeed != clamped {
                optionSeed = clamped
                return
            }
            UserDefaults.standard.set(optionSeed, forKey: runtimeSeedKey)
            if optionSeed != oldValue {
                applySamplerUpdate()
            }
        }
    }

    @Published var optionTemperature: Double = 0.0 {
        didSet {
            let clamped = min(max(optionTemperature, 0.0), 0.0)
            if abs(optionTemperature - clamped) > 0.0001 {
                optionTemperature = clamped
                return
            }
            UserDefaults.standard.set(optionTemperature, forKey: runtimeTempKey)
            if abs(optionTemperature - oldValue) > 0.0001 {
                applySamplerUpdate()
            }
        }
    }

    @Published var optionTopP: Double = 0.95 {
        didSet {
            let clamped = min(max(optionTopP, 0.01), 1.0)
            if abs(optionTopP - clamped) > 0.0001 {
                optionTopP = clamped
                return
            }
            UserDefaults.standard.set(optionTopP, forKey: runtimeTopPKey)
            if abs(optionTopP - oldValue) > 0.0001 {
                applySamplerUpdate()
            }
        }
    }

    @Published var optionTopK: Int = 40 {
        didSet {
            let clamped = max(1, min(optionTopK, 500))
            if optionTopK != clamped {
                optionTopK = clamped
                return
            }
            UserDefaults.standard.set(optionTopK, forKey: runtimeTopKKey)
            if optionTopK != oldValue {
                applySamplerUpdate()
            }
        }
    }

    @Published var optionFlashAttention: Bool = false {
        didSet {
            UserDefaults.standard.set(optionFlashAttention, forKey: runtimeFlashAttnKey)
            if optionFlashAttention != oldValue {
                reloadModelWithCurrentOptionsIfPossible()
            }
        }
    }

    @Published var optionSingleTurn: Bool = true {
        didSet {
            UserDefaults.standard.set(optionSingleTurn, forKey: runtimeSingleTurnKey)
        }
    }

    let NS_PER_S = 1_000_000_000.0

    private var llamaContext: LlamaContext?
    private var currentModelURL: URL?
    private let modelSelectionKey = "llama_swiftui_selected_model"
    private let datasetSelectionKey = "llama_swiftui_selected_dataset"
    private let allowedModelExtensions: Set<String> = ["gguf", "ggml", "bin"]
    private let runtimeNglKey = "llama_swiftui_runtime_ngl"
    private let runtimeContextKey = "llama_swiftui_runtime_ctx"
    private let runtimeSeedKey = "llama_swiftui_runtime_seed"
    private let runtimeTempKey = "llama_swiftui_runtime_temp"
    private let runtimeTopPKey = "llama_swiftui_runtime_top_p"
    private let runtimeTopKKey = "llama_swiftui_runtime_top_k"
    private let runtimeFlashAttnKey = "llama_swiftui_runtime_flash_attn"
    private let runtimeSingleTurnKey = "llama_swiftui_runtime_single_turn"

    private typealias FinetuneLogCallback = @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void

    private static let finetuneLogCallback: FinetuneLogCallback = { messagePtr, userData in
        guard let userData else { return }
        let state = Unmanaged<LlamaState>.fromOpaque(userData).takeUnretainedValue()
        guard let messagePtr else { return }
        let text = String(cString: messagePtr)
        Task { @MainActor in
            state.messageLog += text
        }
    }
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    private let finetuneDatasetFilename = "trump.txt"
    private let defaultFinetuneDatasetURL = URL(string: "https://github.com/user-attachments/files/21859494/trump.txt")
    private let rockTweetsDatasetFilename = "the-rock-tweets.txt"
    private let rockTweetsDatasetURL = URL(string: "https://gist.githubusercontent.com/zoq/774ab751bb3599835362bd6b5b604044/raw/a0d58f5d4bcd46621f9a3112c6fee2c3142fc8a0/the-rock-tweets.txt")

    private lazy var defaultDatasets: [Dataset] = {
        var defaults: [Dataset] = []
        if let url = defaultFinetuneDatasetURL {
            defaults.append(Dataset(name: "Sample Trump Interview", url: url.absoluteString, filename: finetuneDatasetFilename, status: "download"))
        }
        if let rockTweetsURL = rockTweetsDatasetURL {
            defaults.append(Dataset(name: "Sample The Rock Tweets", url: rockTweetsURL.absoluteString, filename: rockTweetsDatasetFilename, status: "download"))
        }
        return defaults
    }()

    private let disallowedDatasetExtensions: Set<String> = ["gguf", "ggml", "bin"]

    init() {
        let defaults = UserDefaults.standard
        if let stored = defaults.object(forKey: runtimeNglKey) as? Int {
            optionNGpuLayers = max(-1, min(stored, 256))
        }
        if let stored = defaults.object(forKey: runtimeContextKey) as? Int, stored >= 32 {
            optionContextLength = min(stored, 8192)
        }
        if let stored = defaults.object(forKey: runtimeSeedKey) as? Int, stored >= 0 {
            optionSeed = min(stored, 1_000_000)
        }
        if let stored = defaults.object(forKey: runtimeTempKey) as? Double {
            optionTemperature = min(max(stored, 0.0), 0.0)
        }
        if let stored = defaults.object(forKey: runtimeTopPKey) as? Double {
            optionTopP = min(max(stored, 0.01), 1.0)
        }
        if let stored = defaults.object(forKey: runtimeTopKKey) as? Int, stored >= 1 {
            optionTopK = min(stored, 500)
        }
        if let stored = defaults.object(forKey: runtimeFlashAttnKey) as? Bool {
            optionFlashAttention = stored
        }
        if let stored = defaults.object(forKey: runtimeSingleTurnKey) as? Bool {
            optionSingleTurn = stored
        }

        loadModelsFromDisk()
        loadDefaultModels()
        selectedModelFilename = UserDefaults.standard.string(forKey: modelSelectionKey)
        ensureSelectedModelIsValid()
        loadDatasetsFromDisk()
        loadDefaultDatasets()
        selectedDatasetFilename = UserDefaults.standard.string(forKey: datasetSelectionKey)
        ensureSelectedDatasetIsValid()
    }

    private func loadModelsFromDisk() {
        downloadedModels.removeAll { model in
            let fileURL = modelFileURL(for: model.filename)
            return !FileManager.default.fileExists(atPath: fileURL.path)
        }

        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )

            for modelURL in modelURLs where !modelURL.hasDirectoryPath {
                let ext = modelURL.pathExtension.lowercased()
                guard allowedModelExtensions.contains(ext) else { continue }
                let filename = modelURL.lastPathComponent
                guard !downloadedModels.contains(where: { $0.filename == filename }) else { continue }

                let displayName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: displayName, url: "", filename: filename, status: "downloaded"))
            }
            downloadedModels.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = modelFileURL(for: model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {
                continue
            }
            guard !undownloadedModels.contains(where: { $0.filename == model.filename }) else { continue }

            var undownloadedModel = model
            undownloadedModel.status = "download"
            undownloadedModels.append(undownloadedModel)
        }

        undownloadedModels.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private func ensureSelectedModelIsValid() {
        guard let selectedFilename = selectedModelFilename else { return }
        let fileURL = modelFileURL(for: selectedFilename)
        let bundleMatch = defaultModelUrl?.lastPathComponent == selectedFilename
        if FileManager.default.fileExists(atPath: fileURL.path) || bundleMatch {
            return
        }
        updateSelectedModel(filename: nil, name: nil, announce: false)
    }

    private func updateSelectedModel(filename: String?, name: String?, announce: Bool) {
        if let filename {
            selectedModelFilename = filename
            UserDefaults.standard.set(filename, forKey: modelSelectionKey)
            if announce {
                let displayName = name ?? filename
                messageLog += "Ready to use model \(displayName).\n"
            }
        } else {
            selectedModelFilename = nil
            UserDefaults.standard.removeObject(forKey: modelSelectionKey)
        }
    }

    private func runtimeOptions() -> LlamaRuntimeOptions {
        let context = Int32(optionContextLength)
        let ngl = optionNGpuLayers >= 0 ? Int32(optionNGpuLayers) : -1
        let seedValue = optionSeed < 0 ? 0 : UInt32(optionSeed)
    let temperature = Float(min(max(optionTemperature, 0.0), 0.0))
        let topP = Float(min(max(optionTopP, 0.01), 1.0))
        let topK = Int32(max(1, optionTopK))

        return LlamaRuntimeOptions(
            contextLength: context,
            nGpuLayers: ngl,
            seed: seedValue,
            temperature: temperature,
            topP: topP,
            topK: topK,
            flashAttention: optionFlashAttention
        )
    }

    private func reloadModelWithCurrentOptionsIfPossible() {
        guard !isFinetuning else {
            messageLog += "Finetuning in progress; runtime option will apply after the current run completes.\n"
            return
        }
        guard let currentModelURL else { return }
        do {
            try loadModel(modelUrl: currentModelURL)
        } catch {
            messageLog += "Failed to reload model with new options: \(error.localizedDescription).\n"
        }
    }

    private func applySamplerUpdate() {
        guard let context = llamaContext else { return }
        let options = runtimeOptions()
        Task {
            await context.updateSampler(options: options)
        }
    }

    func selectDownloadedModel(_ model: Model) {
        let fileURL = modelFileURL(for: model.filename)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            messageLog += "Model \(model.filename) no longer exists on disk.\n"
            return
        }

        do {
            try loadModel(modelUrl: fileURL)
            updateSelectedModel(filename: model.filename, name: model.name, announce: true)
        } catch {
            messageLog += "Failed to load model \(model.filename): \(error.localizedDescription).\n"
        }
    }

    func registerDownloadedModel(_ model: Model) {
        var downloaded = model
        downloaded.status = "downloaded"

        if let index = downloadedModels.firstIndex(where: { $0.filename == downloaded.filename }) {
            downloadedModels[index] = downloaded
        } else {
            downloadedModels.append(downloaded)
        }

        undownloadedModels.removeAll { $0.filename == downloaded.filename }
        downloadedModels.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        messageLog += "Downloaded model \(downloaded.name) -> \(downloaded.filename).\n"
    }

    func deleteModel(_ model: Model) {
        let fileURL = modelFileURL(for: model.filename)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                try FileManager.default.removeItem(at: fileURL)
                messageLog += "Deleted model \(model.filename).\n"
            } catch {
                messageLog += "Failed to delete model \(model.filename): \(error.localizedDescription).\n"
            }
        }

        downloadedModels.removeAll { $0.filename == model.filename }

        if !model.url.isEmpty && !undownloadedModels.contains(where: { $0.filename == model.filename }) {
            var restorable = model
            restorable.status = "download"
            undownloadedModels.append(restorable)
            undownloadedModels.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        }

        if selectedModelFilename == model.filename {
            updateSelectedModel(filename: nil, name: nil, announce: false)
        }
    }

    func deleteDownloadedModels(at offsets: IndexSet) {
        let targets = offsets.compactMap { index in
            downloadedModels.indices.contains(index) ? downloadedModels[index] : nil
        }

        for model in targets {
            deleteModel(model)
        }
    }

    private func loadDatasetsFromDisk() {
        downloadedDatasets.removeAll { !isDatasetFilename($0.filename) || isModelFilename($0.filename) }

        do {
            let documentsURL = getDocumentsDirectory()
            let datasetURLs = try FileManager.default.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )

            for datasetURL in datasetURLs where !datasetURL.hasDirectoryPath {
                if !isDatasetFile(datasetURL) || isModelFilename(datasetURL.lastPathComponent) {
                    continue
                }
                let rawName = datasetURL.deletingPathExtension().lastPathComponent
                let displayName = (rawName.removingPercentEncoding ?? rawName)
                    .replacingOccurrences(of: "_", with: " ")
                    .replacingOccurrences(of: "-", with: " ")
                let dataset = Dataset(
                    name: displayName.isEmpty ? datasetURL.lastPathComponent : displayName,
                    url: "",
                    filename: datasetURL.lastPathComponent,
                    status: "downloaded"
                )

                if !downloadedDatasets.contains(where: { $0.filename == dataset.filename }) {
                    downloadedDatasets.append(dataset)
                }
            }
            downloadedDatasets.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        } catch {
            print("Error loading datasets from disk: \(error)")
        }
    }

    private func loadDefaultDatasets() {
        for dataset in defaultDatasets {
            let fileURL = datasetFileURL(for: dataset.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {
                if !downloadedDatasets.contains(where: { $0.filename == dataset.filename }) {
                    var downloaded = dataset
                    downloaded.status = "downloaded"
                    downloadedDatasets.append(downloaded)
                }
            } else if !availableDatasets.contains(where: { $0.filename == dataset.filename }) {
                availableDatasets.append(dataset)
            }
        }

        downloadedDatasets.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        availableDatasets.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private func ensureSelectedDatasetIsValid() {
        if let selectedFilename = selectedDatasetFilename {
            let selectedURL = datasetFileURL(for: selectedFilename)
            if isDatasetFilename(selectedFilename)
                && !isModelFilename(selectedFilename)
                && FileManager.default.fileExists(atPath: selectedURL.path) {
                return
            }
        }

        if let firstDataset = downloadedDatasets.first {
            updateSelectedDataset(filename: firstDataset.filename, name: firstDataset.name, announce: false)
        } else {
            updateSelectedDataset(filename: nil, name: nil, announce: false)
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }

    func datasetFileURL(for filename: String) -> URL {
        getDocumentsDirectory().appendingPathComponent(filename)
    }

    func modelFileURL(for filename: String) -> URL {
        getDocumentsDirectory().appendingPathComponent(filename)
    }

    private func isDatasetFilename(_ filename: String) -> Bool {
        let ext = URL(fileURLWithPath: filename).pathExtension.lowercased()
        if ext.isEmpty {
            return true
        }
        return !disallowedDatasetExtensions.contains(ext)
    }

    private func isDatasetFile(_ url: URL) -> Bool {
        isDatasetFilename(url.lastPathComponent)
    }

    private func updateSelectedDataset(filename: String?, name: String?, announce: Bool) {
        if let filename {
            selectedDatasetFilename = filename
            UserDefaults.standard.set(filename, forKey: datasetSelectionKey)
            if announce {
                let displayName = name ?? filename
                messageLog += "Using dataset \(displayName) for finetuning.\n"
            }
        } else {
            selectedDatasetFilename = nil
            UserDefaults.standard.removeObject(forKey: datasetSelectionKey)
        }
    }

    func selectDataset(_ dataset: Dataset) {
        updateSelectedDataset(filename: dataset.filename, name: dataset.name, announce: true)
    }

    func registerDownloadedDataset(_ dataset: Dataset) {
        guard isDatasetFilename(dataset.filename), !isModelFilename(dataset.filename) else {
            let fileURL = datasetFileURL(for: dataset.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {
                do {
                    try FileManager.default.removeItem(at: fileURL)
                    messageLog += "Removed non-dataset file \(dataset.filename) from datasets folder.\n"
                } catch {
                    messageLog += "Failed to remove non-dataset file \(dataset.filename): \(error.localizedDescription).\n"
                }
            }
            messageLog += "Ignoring \(dataset.filename) because it is not a dataset file.\n"
            return
        }
        var downloaded = dataset
        downloaded.status = "downloaded"

        if let index = downloadedDatasets.firstIndex(where: { $0.filename == downloaded.filename }) {
            downloadedDatasets[index] = downloaded
        } else {
            downloadedDatasets.append(downloaded)
        }

        availableDatasets.removeAll { $0.filename == downloaded.filename }
        downloadedDatasets.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        messageLog += "Downloaded dataset \(downloaded.name) -> \(downloaded.filename).\n"
        updateSelectedDataset(filename: downloaded.filename, name: downloaded.name, announce: true)
    }

    private func isModelFilename(_ filename: String) -> Bool {
        let ext = URL(fileURLWithPath: filename).pathExtension.lowercased()
        if allowedModelExtensions.contains(ext) {
            return true
        }
        return downloadedModels.contains { $0.filename == filename }
    }

    func deleteDataset(_ dataset: Dataset) {
        let fileURL = datasetFileURL(for: dataset.filename)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                try FileManager.default.removeItem(at: fileURL)
                messageLog += "Deleted dataset \(dataset.filename).\n"
            } catch {
                messageLog += "Failed to delete dataset \(dataset.filename): \(error.localizedDescription).\n"
            }
        }

        downloadedDatasets.removeAll { $0.filename == dataset.filename }

        if !dataset.url.isEmpty && !availableDatasets.contains(where: { $0.filename == dataset.filename }) {
            var restorable = dataset
            restorable.status = "download"
            availableDatasets.append(restorable)
            availableDatasets.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        }

        if selectedDatasetFilename == dataset.filename {
            updateSelectedDataset(filename: nil, name: nil, announce: false)
            ensureSelectedDatasetIsValid()
        }
    }

    func deleteDownloadedDatasets(at offsets: IndexSet) {
        let targets = offsets.compactMap { index in
            downloadedDatasets.indices.contains(index) ? downloadedDatasets[index] : nil
        }

        for dataset in targets {
            deleteDataset(dataset)
        }
    }
    private static let tetherGgufModelsRepo = "Benson-Chen/tether-gguf-models"

    private static func tetherGgufModelURL(path: String) -> String {
        "https://huggingface.co/\(tetherGgufModelsRepo)/resolve/main/\(path)"
    }

    private static func tetherModel(name: String, path: String, bytes: Int) -> Model {
        let gib = String(format: "%.2f", Double(bytes) / 1024.0 / 1024.0 / 1024.0)
        let filename = (path as NSString).lastPathComponent
        return Model(
            name: "\(name) (\(gib) GiB)",
            url: tetherGgufModelURL(path: path),
            filename: filename,
            status: "download"
        )
    }

    /// GGUF models from https://huggingface.co/Benson-Chen/tether-gguf-models
    private let defaultModels: [Model] = [
        // qwen3-0.6B
        LlamaState.tetherModel(name: "Qwen3-0.6B Q4_K_M", path: "qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf", bytes: 396_705_472),
        LlamaState.tetherModel(name: "Qwen3-0.6B TQ2_0 Tether", path: "qwen3-0.6B/Qwen3-0.6B-TQ2_0_Tether.gguf", bytes: 247_376_480),
        // qwen3-1.7B
        LlamaState.tetherModel(name: "Qwen3-1.7B TQ1_0", path: "qwen3-1.7B/Qwen3-1.7B-TQ1_0.gguf", bytes: 558_971_200),
        LlamaState.tetherModel(name: "Qwen3-1.7B TQ2_0", path: "qwen3-1.7B/Qwen3-1.7B-TQ2_0.gguf", bytes: 625_031_488),
        LlamaState.tetherModel(name: "Qwen3-1.7B Q2_K", path: "qwen3-1.7B/Qwen3-1.7B-Q2_K.gguf", bytes: 777_795_904),
        LlamaState.tetherModel(name: "Qwen3-1.7B Q3_K_M", path: "qwen3-1.7B/Qwen3-1.7B-Q3_K_M.gguf", bytes: 939_538_752),
        LlamaState.tetherModel(name: "Qwen3-1.7B Q4_0", path: "qwen3-1.7B/Qwen3-1.7B-Q4_0.gguf", bytes: 1_054_423_360),
        LlamaState.tetherModel(name: "Qwen3-1.7B Q4_K_M", path: "qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf", bytes: 1_107_409_216),
        LlamaState.tetherModel(name: "Qwen3-1.7B BF16", path: "qwen3-1.7B/Qwen3-1.7B-BF16.gguf", bytes: 3_447_349_568),
        // bonsai-1.7B
        LlamaState.tetherModel(name: "Bonsai-1.7B Q1_0", path: "bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf", bytes: 248_302_272),
        LlamaState.tetherModel(name: "Bonsai-1.7B Q2_0", path: "bonsai-1.7B/Bonsai-1.7B-Q2_0.gguf", bytes: 463_290_464),
        // bonsai-4B
        LlamaState.tetherModel(name: "Bonsai-4B Q1_0", path: "bonsai-4B/Bonsai-4B-Q1_0.gguf", bytes: 572_270_624),
        LlamaState.tetherModel(name: "Bonsai-4B Q2_0", path: "bonsai-4B/Bonsai-4B-Q2_0.gguf", bytes: 1_074_969_344),
        // bonsai-8B
        LlamaState.tetherModel(name: "Bonsai-8B Q1_0", path: "bonsai-8B/Bonsai-8B-Q1_0.gguf", bytes: 1_158_654_496),
        LlamaState.tetherModel(name: "Bonsai-8B Q2_0", path: "bonsai-8B/Bonsai-8B-Q2_0.gguf", bytes: 2_182_184_672),
    ]
    func loadModel(modelUrl: URL?) throws {
        guard let modelUrl else {
            messageLog += "Load a model from the list below\n"
            return
        }

        messageLog += "Loading model...\n"
        llamaContext = try LlamaContext.create_context(path: modelUrl.path(), options: runtimeOptions())
        messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
        currentModelURL = modelUrl

        let filename = modelUrl.lastPathComponent
        updateDownloadedModels(filename: filename)
        let displayName = modelUrl.deletingPathExtension().lastPathComponent
        updateSelectedModel(filename: filename, name: displayName, announce: false)
    }


    private func updateDownloadedModels(filename: String) {
        undownloadedModels.removeAll { $0.filename == filename }
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        await llamaContext.updateSampler(options: runtimeOptions())

        if optionSingleTurn {
            messageLog = ""
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: text)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        messageLog += "\(text)"

        Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            let tokens_per_second = Double(await llamaContext.n_len) / t_generation

            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(tokens_per_second) t/s\n
                    """
            }
        }
    }

    func bench(cpuOnly: Bool = false) async {
        guard let modelURL = currentModelURL else {
            messageLog += "Load a model before running benchmark.\n"
            return
        }
        if isFinetuning {
            messageLog += "Finetuning in progress; wait for it to finish.\n"
            return
        }

        let savedOptions = runtimeOptions()
        let benchNgl: Int32 = cpuOnly ? 0 : (optionNGpuLayers >= 0 ? Int32(optionNGpuLayers) : 99)
        var benchOptions = savedOptions
        benchOptions.nGpuLayers = benchNgl

        let backendLabel = benchNgl == 0 ? "CPU (ngl=0)" : "Metal (ngl=\(benchNgl))"
        messageLog += "\nRunning \(backendLabel) benchmark...\n"

        if savedOptions.nGpuLayers != benchNgl || llamaContext == nil {
            messageLog += "Loading model for benchmark (\(backendLabel))...\n"
            do {
                llamaContext = try LlamaContext.create_context(path: modelURL.path(), options: benchOptions)
            } catch {
                messageLog += "Failed to load model for benchmark: \(error.localizedDescription).\n"
                return
            }
        }

        guard let llamaContext else { return }

        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            if savedOptions.nGpuLayers != benchNgl {
                restoreModelContext(modelURL: modelURL, options: savedOptions)
            }
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"

        if savedOptions.nGpuLayers != benchNgl {
            restoreModelContext(modelURL: modelURL, options: savedOptions)
        }
    }

    private func restoreModelContext(modelURL: URL, options: LlamaRuntimeOptions) {
        do {
            llamaContext = try LlamaContext.create_context(path: modelURL.path(), options: options)
            messageLog += "Restored model with ngl=\(options.nGpuLayers >= 0 ? options.nGpuLayers : -1).\n"
        } catch {
            messageLog += "Failed to restore model after benchmark: \(error.localizedDescription).\n"
        }
    }

    func finetune() async {
        if isFinetuning {
            messageLog += "A finetuning run is already in progress.\n"
            return
        }

        guard let modelURL = currentModelURL else {
            messageLog += "Load a model before starting finetuning.\n"
            return
        }

        guard let datasetURL = locateFinetuneDataset() else {
            messageLog += "Unable to locate a finetuning dataset. Download one from Settings first.\n"
            return
        }

        let outputURL = getDocumentsDirectory().appendingPathComponent("finetuned-\(modelURL.deletingPathExtension().lastPathComponent).gguf")
        let outputFilename = outputURL.lastPathComponent

        let ngl = optionNGpuLayers >= 0 ? Int32(optionNGpuLayers) : -1
        let options = llama_swift_finetune_options(
            n_ctx: Int32(optionContextLength),
            n_threads: Int32(max(1, ProcessInfo.processInfo.processorCount - 1)),
            n_batch: 256,
            n_ubatch: 256,
            epochs: 2,
            lora_rank: 8,
            lora_alpha: 16,
            learning_rate: 1e-5,
            val_split: 0.05,
            target_modules: 0,
            seed: Int32(optionSeed),
            flash_attn: optionFlashAttention,
            n_gpu_layers: ngl,
            checkpoint_save_steps: 0,
            checkpoint_save_dir: nil,
            resume_from_checkpoint: nil,
            auto_resume: false
        )

        isFinetuning = true
        messageLog += "Starting on-device LoRA finetuning for \(modelURL.lastPathComponent).\n"

        let modelPath = modelURL.path
        let datasetPath = datasetURL.path
        let outputPath = outputURL.path
        let statePointer = Unmanaged.passUnretained(self).toOpaque()
        let callback = LlamaState.finetuneLogCallback

        Task.detached(priority: .userInitiated) {
            var opts = options
            let result: llama_swift_finetune_error = modelPath.withCString { modelC in
                datasetPath.withCString { datasetC in
                    outputPath.withCString { outputC in
                        llama_swift_run_lora_finetune(modelC, datasetC, outputC, &opts, callback, statePointer)
                    }
                }
            }

            Task { @MainActor in
                let state = Unmanaged<LlamaState>.fromOpaque(statePointer).takeUnretainedValue()
                state.isFinetuning = false
                if result == LLAMA_SWIFT_FINETUNE_OK {
                    state.messageLog += "Finetuning finished successfully. Adapter saved as \(outputFilename).\n"
                } else {
                    state.messageLog += "Finetuning failed: \(state.describeFinetuneError(result)).\n"
                }
            }
        }
    }

    private func locateFinetuneDataset() -> URL? {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment

        if let overridePath = environment["LLAMA_FINETUNE_DATASET"], !overridePath.isEmpty {
            let overrideURL = URL(fileURLWithPath: overridePath)
            if fileManager.fileExists(atPath: overrideURL.path) {
                return overrideURL
            }
        }

        if let selectedFilename = selectedDatasetFilename {
            let selectedURL = datasetFileURL(for: selectedFilename)
            if fileManager.fileExists(atPath: selectedURL.path) {
                return selectedURL
            }
        }

        for dataset in downloadedDatasets {
            let candidate = datasetFileURL(for: dataset.filename)
            if fileManager.fileExists(atPath: candidate.path) {
                updateSelectedDataset(filename: dataset.filename, name: dataset.name, announce: false)
                return candidate
            }
        }

        let workingDirectoryCandidate = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(finetuneDatasetFilename)
        if fileManager.fileExists(atPath: workingDirectoryCandidate.path) {
            return workingDirectoryCandidate
        }

        return nil
    }

    private func describeFinetuneError(_ code: llama_swift_finetune_error) -> String {
        switch code {
        case LLAMA_SWIFT_FINETUNE_ERROR_INVALID_ARGUMENT:
            return "invalid configuration"
        case LLAMA_SWIFT_FINETUNE_ERROR_MODEL_LOAD:
            return "failed to load model"
        case LLAMA_SWIFT_FINETUNE_ERROR_CONTEXT_CREATE:
            return "failed to create training context"
        case LLAMA_SWIFT_FINETUNE_ERROR_DATASET:
            return "dataset preparation failed"
        case LLAMA_SWIFT_FINETUNE_ERROR_TRAINING_INIT:
            return "training initialization failed"
        case LLAMA_SWIFT_FINETUNE_ERROR_SAVE:
            return "failed to save adapter"
        default:
            return "unexpected error (\(code.rawValue))"
        }
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }

}
