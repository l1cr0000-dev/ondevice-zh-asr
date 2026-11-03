import Foundation

/// 找模型根目錄（含 silero-vad-v6.2.3/ 與 sherpa-onnx-sense-voice-.../）。
/// 順序：明確指定 → 環境變數 OZASR_MODELS_DIR → app bundle 的 Resources/models（打包版）→
/// 從 app 所在位置往上找 repo 的 models/（開發版，不複製模型）。
enum ModelLocator {
    static let marker = "silero-vad-v6.2.3/silero_vad.onnx"

    static func valid(_ dir: String) -> Bool {
        FileManager.default.fileExists(atPath: dir + "/" + marker)
    }

    static func locate(explicit: String? = nil) -> String? {
        if let e = explicit, !e.isEmpty { return valid(e) ? e : nil }
        if let env = ProcessInfo.processInfo.environment["OZASR_MODELS_DIR"], valid(env) { return env }
        if let res = Bundle.main.resourceURL?.appendingPathComponent("models").path, valid(res) { return res }
        var url = URL(fileURLWithPath: Bundle.main.bundlePath).deletingLastPathComponent()
        for _ in 0..<8 {
            let cand = url.appendingPathComponent("models").path
            if valid(cand) { return cand }
            url.deleteLastPathComponent()
        }
        return nil
    }
}
