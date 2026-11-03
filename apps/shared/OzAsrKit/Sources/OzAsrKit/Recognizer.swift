import COzAsr
import Foundation

/// 單調牆鐘（秒）。事件時間戳與延遲計算都用它，不受系統時間調整影響。
public enum OzClock {
    public static func now() -> TimeInterval {
        Double(DispatchTime.now().uptimeNanoseconds) / 1e9
    }
}

public enum OzAsrError: Error, CustomStringConvertible {
    case model(String)
    case runtime(String)
    case invalidArgument(String)
    case io(String)

    public var description: String {
        switch self {
        case .model(let m): return "模型錯誤：\(m)"
        case .runtime(let m): return "執行錯誤：\(m)"
        case .invalidArgument(let m): return "參數錯誤：\(m)"
        case .io(let m): return "檔案錯誤：\(m)"
        }
    }
}

public enum OzAsrEventKind: Int32, Sendable {
    case start = 0, partial = 1, provisional = 2, final = 3
}

/// 辨識事件。音訊時間軸欄位（秒）以 `reset()` / 建立後第一個樣本為 0；
/// `wallTime` 是平台層在 C 呼叫返回後立刻標記的單調牆鐘（`OzClock`）。
public struct OzAsrEvent: Sendable {
    public var kind: OzAsrEventKind
    public var utterance: Int
    public var text: String
    public var audioPos: Double
    public var speechStart: Double
    public var audioStart: Double
    public var speechEnd: Double?
    public var audioEnd: Double?
    /// C++ 核心的「模擬即時」發出時刻（虛擬時鐘），只用於和離線評測對照。
    public var simulatedEmitted: Double
    /// 牆鐘發出時刻（`OzClock.now()`）。
    public var wallTime: TimeInterval
}

public struct OzAsrConfig: Sendable {
    public var vadModelPath: String
    public var asrModelPath: String
    public var tokensPath: String
    public var language = "zh"
    public var asrThreads = 2
    public var vadThreads = 2

    public init(vadModelPath: String, asrModelPath: String, tokensPath: String) {
        self.vadModelPath = vadModelPath
        self.asrModelPath = asrModelPath
        self.tokensPath = tokensPath
    }

    /// 模型根目錄需含 `silero-vad-v6.2.3/silero_vad.onnx` 與
    /// `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/{model.int8.onnx,tokens.txt}`。
    public init(modelsDirectory: String) {
        let sv = modelsDirectory + "/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        self.init(
            vadModelPath: modelsDirectory + "/silero-vad-v6.2.3/silero_vad.onnx",
            asrModelPath: sv + "/model.int8.onnx", tokensPath: sv + "/tokens.txt")
    }

    public var missingFiles: [String] {
        [vadModelPath, asrModelPath, tokensPath].filter { !FileManager.default.fileExists(atPath: $0) }
    }
}

public struct OzAsrStats: Sendable {
    public var numDecodes: Int
    public var decodedAudioSeconds: Double
}

/// C API 的 Swift 包裝。非執行緒安全：同一個實例請固定在同一個序列佇列 / 執行緒呼叫。
public final class OzAsrRecognizer {
    private var handle: OpaquePointer?
    /// 模型載入（`ozasr_create`）的牆鐘時間。
    public let loadSeconds: TimeInterval
    /// acceptWaveform / flush 內部（C++ 推論）累計花掉的牆鐘時間；除以音訊長度即 RTF。
    public private(set) var computeSeconds: TimeInterval = 0
    public static var version: String { String(cString: ozasr_version()) }

    public init(config: OzAsrConfig) throws {
        let missing = config.missingFiles
        if !missing.isEmpty { throw OzAsrError.model("找不到模型檔：" + missing.joined(separator: "、")) }
        var c = ozasr_config()
        ozasr_config_default(&c)
        c.asr_num_threads = Int32(config.asrThreads)
        c.vad_num_threads = Int32(config.vadThreads)
        var h: OpaquePointer?
        let t0 = OzClock.now()
        // C 字串只需在 ozasr_create 期間有效
        let status: ozasr_status = config.vadModelPath.withCString { vad in
            config.asrModelPath.withCString { asr in
                config.tokensPath.withCString { tok in
                    config.language.withCString { lang in
                        c.vad_model_path = vad
                        c.asr_model_path = asr
                        c.tokens_path = tok
                        c.language = lang
                        return ozasr_create(&c, &h)
                    }
                }
            }
        }
        loadSeconds = OzClock.now() - t0
        guard status == OZASR_OK, let h else {
            let msg = ozasr_last_error(nil).map { String(cString: $0) } ?? "未知錯誤"
            throw status == OZASR_ERROR_MODEL ? OzAsrError.model(msg) : OzAsrError.runtime(msg)
        }
        handle = h
    }

    deinit { if let handle { ozasr_destroy(handle) } }

    /// 16 kHz、單聲道、float32 樣本（建議每次 160 個 = 10 ms）。回傳這次產生的事件，已蓋牆鐘。
    public func acceptWaveform(_ samples: UnsafeBufferPointer<Float>) throws -> [OzAsrEvent] {
        let t0 = OzClock.now()
        let st = ozasr_accept_waveform(handle, samples.baseAddress, samples.count)
        let wall = OzClock.now()
        computeSeconds += wall - t0
        try check(st)
        return try drain(wall: wall)
    }

    public func acceptWaveform(_ samples: [Float]) throws -> [OzAsrEvent] {
        try samples.withUnsafeBufferPointer { try acceptWaveform($0) }
    }

    /// 輸入結束，結束進行中的句子。
    public func flush() throws -> [OzAsrEvent] {
        let t0 = OzClock.now()
        let st = ozasr_flush(handle)
        let wall = OzClock.now()
        computeSeconds += wall - t0
        try check(st)
        return try drain(wall: wall)
    }

    public func reset() throws {
        try check(ozasr_reset(handle))
        computeSeconds = 0
    }

    public func stats() throws -> OzAsrStats {
        var s = ozasr_stats()
        try check(ozasr_get_stats(handle, &s))
        return OzAsrStats(numDecodes: Int(s.num_decodes), decodedAudioSeconds: s.decoded_audio_s)
    }

    private func check(_ st: ozasr_status) throws {
        if st == OZASR_OK { return }
        let msg = ozasr_last_error(handle).map { String(cString: $0) } ?? "未知錯誤"
        throw st == OZASR_ERROR_INVALID_ARGUMENT ? OzAsrError.invalidArgument(msg) : OzAsrError.runtime(msg)
    }

    private func drain(wall: TimeInterval) throws -> [OzAsrEvent] {
        var out: [OzAsrEvent] = []
        var ev = ozasr_event()
        while true {
            let r = ozasr_next_event(handle, &ev)
            if r < 0 { throw OzAsrError.runtime(ozasr_last_error(handle).map { String(cString: $0) } ?? "next_event") }
            if r == 0 { break }
            out.append(
                OzAsrEvent(
                    kind: OzAsrEventKind(rawValue: ev.kind) ?? .final, utterance: Int(ev.utterance),
                    text: ev.text.map { String(cString: $0) } ?? "", audioPos: ev.audio_pos_s,
                    speechStart: ev.speech_start_s, audioStart: ev.audio_start_s,
                    speechEnd: ev.speech_end_s >= 0 ? ev.speech_end_s : nil,
                    audioEnd: ev.audio_end_s >= 0 ? ev.audio_end_s : nil,
                    simulatedEmitted: ev.emitted_s, wallTime: wall))
        }
        return out
    }
}
