import Foundation

enum LlamaModelBackend {
    case qvac
    case prism

    static func forModel(path: String) -> LlamaModelBackend {
        let filename = (path as NSString).lastPathComponent.lowercased()
        if filename.contains("bonsai") {
            return .prism
        }
        return .qvac
    }

    static func forModel(url: URL) -> LlamaModelBackend {
        forModel(path: url.path)
    }
}
