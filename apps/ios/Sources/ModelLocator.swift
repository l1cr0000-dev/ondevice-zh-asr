import Foundation

/// app bundle 內的資料：`oz/models`（VAD、SenseVoice INT8、tokens）與 `oz/wavs`（測試 wav）。
/// 由 scripts/build_ios_app.sh 在建置前複製進 build/ios/stage/oz，Xcode 以資料夾參照打包；不進 git。
enum Bundled {
    static let marker = "silero-vad-v6.2.3/silero_vad.onnx"
    static var root: String { (Bundle.main.resourcePath ?? "") + "/oz" }
    static var modelsDir: String? {
        let env = ProcessInfo.processInfo.environment["OZASR_MODELS_DIR"]
        for c in [env, root + "/models"].compactMap({ $0 }) where FileManager.default.fileExists(atPath: c + "/" + marker) {
            return c
        }
        return nil
    }
    /// 內附 wav（檔名排序；fixture 排第一）
    static var wavs: [String] {
        let dir = root + "/wavs"
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.hasSuffix(".wav") }.sorted()
        return names.map { dir + "/" + $0 }
    }
}
