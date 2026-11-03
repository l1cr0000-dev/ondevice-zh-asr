import AVFoundation
import Foundation
import OzAsrKit
import SwiftUI

enum InputMode: String, CaseIterable, Identifiable {
    case pushToTalk = "按住說話"
    case continuous = "連續收音"
    case wav = "wav 檔"
    var id: String { rawValue }
}

final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}

/// 辨識工作執行緒：recognizer、session、tracker 都只在 `queue` 上存取（C 核心非執行緒安全）。
/// 與 Mac 版的 Worker 同一套邏輯；另外多了 `runAutoRun`（自動驗證，輸出 JSON）。
final class Worker: @unchecked Sendable {
    let queue = DispatchQueue(label: "ozasr.engine", qos: .userInitiated)
    var recognizer: OzAsrRecognizer?
    private var session: OzPushToTalkSession?
    private var tracker: OzUtteranceTracker?
    private var mic: OzMicrophone?
    private var micStarted = false
    private var stopFlag = StopFlag()

    var onUtterances: (([OzUtterance]) -> Void)?
    var onLoaded: ((Double) -> Void)?
    var onStatus: ((String) -> Void)?
    var onHolding: ((Bool) -> Void)?
    var onFinished: ((String?, Double?) -> Void)?  // 錯誤訊息、RTF

    private func main(_ f: @escaping () -> Void) { DispatchQueue.main.async(execute: f) }

    private func ensureRecognizer(dir: String) throws -> OzAsrRecognizer {
        if let r = recognizer { try r.reset(); return r }
        let r = try OzAsrRecognizer(config: OzAsrConfig(modelsDirectory: dir))
        recognizer = r
        let load = r.loadSeconds
        main { self.onLoaded?(load) }
        return r
    }

    private func publish() {
        guard let u = session?.tracker.utterances ?? tracker?.utterances else { return }
        main { self.onUtterances?(u) }
    }

    // MARK: wav（依實際時長以即時節奏送入，牆鐘計時）

    func runWav(dir: String, paths: [String], pushToTalk: Bool) {
        stopFlag = StopFlag()
        let flag = stopFlag
        queue.async {
            do {
                self.main { self.onStatus?("載入模型…") }
                let rec = try self.ensureRecognizer(dir: dir)
                var total = 0.0
                if pushToTalk {
                    let s = OzPushToTalkSession(recognizer: rec)
                    self.session = s
                    self.tracker = nil
                    for p in paths {
                        if flag.isSet { break }
                        let audio = try OzWav.load(path: p)
                        total += Double(audio.count) / Double(OzWav.sampleRate)
                        self.main { self.onHolding?(true); self.onStatus?("按住中（wav 模擬）") }
                        try OzPacedFeeder.feedPushToTalk(session: s, samples: audio, shouldStop: { flag.isSet }) { _ in
                            self.publish()
                        }
                        self.main { self.onHolding?(false) }
                        self.publish()
                    }
                } else {
                    var audio: [Float] = []
                    for p in paths {
                        audio += try OzWav.load(path: p)
                        audio += [Float](repeating: 0, count: OzWav.sampleRate)
                    }
                    total = Double(audio.count) / Double(OzWav.sampleRate)
                    self.session = nil
                    var t = OzUtteranceTracker(streamOrigin: OzClock.now())
                    self.tracker = t
                    self.main { self.onStatus?("辨識中（即時節奏送入 \(String(format: "%.1f", total)) s）") }
                    try OzPacedFeeder.feed(
                        recognizer: rec, samples: audio, realtime: true, shouldStop: { flag.isSet },
                        onOrigin: { t.streamOrigin = $0 },
                        onEvents: { evs in
                            for e in evs { t.apply(e) }
                            self.tracker = t
                            self.publish()
                        })
                }
                let rtf: Double? = total > 0 ? rec.computeSeconds / total : nil
                self.main { self.onFinished?(nil, rtf) }
            } catch {
                self.main { self.onHolding?(false); self.onFinished?("\(error)", nil) }
            }
        }
    }

    // MARK: 自動驗證（wav、牆鐘即時節奏；輸出格式與 Mac 的 --headless 相同，一檔一行 JSON，最後一行 summary）

    func runAutoRun(dir: String, paths: [String], pushToTalk: Bool, outPath: String, warmup: Int, done: @escaping () -> Void) {
        queue.async {
            var lines: [String] = []
            func line(_ o: [String: Any]) {
                if let d = try? JSONSerialization.data(withJSONObject: o, options: [.withoutEscapingSlashes]),
                    let s = String(data: d, encoding: .utf8) { lines.append(s) }
            }
            func flush() { try? (lines.joined(separator: "\n") + "\n").write(toFile: outPath, atomically: true, encoding: .utf8) }
            do {
                self.main { self.onStatus?("載入模型…") }
                let rec = try self.ensureRecognizer(dir: dir)
                let rssLoaded = OzMemory.currentResidentMB()
                var totalAudio = 0.0, totalCompute = 0.0, counted = 0
                var sumFirst = 0.0, nFirst = 0, sumFinal = 0.0, nFinal = 0, sumRel = 0.0, nRel = 0
                for (f, path) in paths.enumerated() {
                    var audio = try OzWav.load(path: path)
                    if !pushToTalk { audio += [Float](repeating: 0, count: OzWav.sampleRate) }
                    try rec.reset()
                    let counting = f >= warmup
                    let name = (path as NSString).lastPathComponent
                    self.main { self.onStatus?("自動驗證 \(f + 1)/\(paths.count)：\(name)") }
                    var tracker = OzUtteranceTracker(streamOrigin: 0)
                    var events: [[String: Any]] = []
                    let session = OzPushToTalkSession(recognizer: rec)
                    self.session = pushToTalk ? session : nil
                    let wall0 = OzClock.now()
                    let computeBefore = rec.computeSeconds
                    let record: ([OzAsrEvent]) -> Void = { evs in
                        for e in evs {
                            if !pushToTalk { tracker.apply(e) }
                            let origin = pushToTalk ? session.tracker.streamOrigin : tracker.streamOrigin
                            events.append([
                                "type": "\(e.kind)", "utterance": e.utterance, "text": e.text,
                                "wall_s": e.wallTime - origin, "simulated_s": e.simulatedEmitted,
                                "audio_pos_s": e.audioPos, "speech_start_s": e.speechStart,
                                "speech_end_s": e.speechEnd ?? NSNull(),
                            ])
                        }
                        let u = pushToTalk ? session.tracker.utterances : tracker.utterances
                        self.main { self.onUtterances?(u) }
                    }
                    if pushToTalk {
                        self.main { self.onHolding?(true) }
                        try OzPacedFeeder.feedPushToTalk(session: session, samples: audio, realtime: counting, onEvents: record)
                        self.main { self.onHolding?(false) }
                        tracker = session.tracker
                    } else {
                        try OzPacedFeeder.feed(
                            recognizer: rec, samples: audio, realtime: counting,
                            onOrigin: { tracker.streamOrigin = $0 }, onEvents: record)
                    }
                    let wall = OzClock.now() - wall0
                    guard counting else { continue }
                    let dur = Double(audio.count) / Double(OzWav.sampleRate)
                    totalAudio += dur
                    totalCompute += rec.computeSeconds - computeBefore
                    counted += 1
                    let utts: [[String: Any]] = tracker.utterances.map { u in
                        if let v = u.firstTokenLatency { sumFirst += v; nFirst += 1 }
                        if let v = u.finalLatency { sumFinal += v; nFinal += 1 }
                        if let v = u.releaseToFinalLatency { sumRel += v; nRel += 1 }
                        return [
                            "index": u.id, "text": u.text, "final": u.status == .final,
                            "speech_start_s": u.speechStart, "speech_end_s": u.speechEnd ?? NSNull(),
                            "first_token_latency_s": u.firstTokenLatency ?? NSNull(),
                            "final_latency_s": u.finalLatency ?? NSNull(),
                            "release_to_final_latency_s": u.releaseToFinalLatency ?? NSNull(),
                            "sim_first_token_latency_s": u.simulatedFirstTokenLatency ?? NSNull(),
                            "sim_final_latency_s": u.simulatedFinalLatency ?? NSNull(),
                        ]
                    }
                    line([
                        "type": "file", "wav": name, "duration_s": dur, "wall_s": wall,
                        "compute_s": rec.computeSeconds - computeBefore, "events": events, "utterances": utts,
                    ])
                    flush()
                }
                line([
                    "type": "summary", "files": counted, "audio_s": totalAudio, "compute_s": totalCompute,
                    "rtf": totalAudio > 0 ? totalCompute / totalAudio : NSNull(), "load_s": rec.loadSeconds,
                    "rss_after_load_mb": rssLoaded, "peak_rss_mb": OzMemory.peakResidentMB(),
                    "mean_first_token_latency_s": nFirst > 0 ? sumFirst / Double(nFirst) : NSNull(),
                    "mean_final_latency_s": nFinal > 0 ? sumFinal / Double(nFinal) : NSNull(),
                    "mean_release_to_final_latency_s": nRel > 0 ? sumRel / Double(nRel) : NSNull(),
                    "ptt": pushToTalk, "realtime": true, "tail_silence_ms": pushToTalk ? 0 : 1000,
                    "version": OzAsrRecognizer.version, "platform": ProcessInfo.processInfo.operatingSystemVersionString,
                    "device": Worker.deviceModel(), "ok": true,
                ])
                flush()
                self.main { self.onFinished?(nil, totalAudio > 0 ? totalCompute / totalAudio : nil); done() }
            } catch {
                line(["type": "error", "message": "\(error)", "ok": false])
                flush()
                self.main { self.onFinished?("\(error)", nil); done() }
            }
        }
    }

    static func deviceModel() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &buf, &size, nil, 0)
        let m = String(cString: buf)
        // 模擬器的 hw.machine 是 arm64；真實機型在環境變數
        return ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"].map { "Simulator(\($0))" } ?? m
    }

    // MARK: 麥克風（只有使用者按「開啟麥克風」才會開啟輸入裝置）

    func startMic(dir: String, pushToTalk: Bool) {
        stopFlag = StopFlag()
        queue.async {
            do {
                self.main { self.onStatus?("載入模型…") }
                let rec = try self.ensureRecognizer(dir: dir)
                self.micStarted = false
                if pushToTalk {
                    self.session = OzPushToTalkSession(recognizer: rec)
                    self.tracker = nil
                } else {
                    self.session = nil
                    self.tracker = OzUtteranceTracker(streamOrigin: OzClock.now())
                }
                let m = OzMicrophone()
                try m.start { chunk in
                    self.queue.async { self.handleChunk(chunk, pushToTalk: pushToTalk) }
                }
                self.mic = m
                self.main { self.onStatus?(pushToTalk ? "麥克風已開啟：按住按鈕說話" : "聆聽中…") }
            } catch {
                self.main { self.onFinished?("\(error)", nil) }
            }
        }
    }

    private func handleChunk(_ chunk: [Float], pushToTalk: Bool) {
        guard mic != nil else { return }
        do {
            if pushToTalk {
                if let s = session, !s.isPressed { return }
                if try !(session?.accept(chunk) ?? []).isEmpty { publish() }
            } else {
                guard var t = tracker else { return }
                if !micStarted { t.streamOrigin = OzClock.now() - 0.01; micStarted = true }
                let evs = try recognizer?.acceptWaveform(chunk) ?? []
                for e in evs { t.apply(e) }
                tracker = t
                if !evs.isEmpty { publish() }
            }
        } catch {
            main { self.onStatus?("錯誤：\(error)") }
        }
    }

    func press() { queue.async { self.session?.press() } }

    func release() {
        queue.async {
            do { _ = try self.session?.release() } catch { self.main { self.onStatus?("錯誤：\(error)") } }
            self.publish()
        }
    }

    func stop() {
        stopFlag.set()
        queue.async {
            self.mic?.stop()
            let hadMic = self.mic != nil
            self.mic = nil
            if hadMic {
                do {
                    if let s = self.session { _ = try s.release() } else if var t = self.tracker, let r = self.recognizer {
                        for e in try r.flush() { t.apply(e) }
                        self.tracker = t
                    }
                } catch {}
                self.publish()
                self.main { self.onFinished?(nil, nil) }
            }
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var mode = InputMode.wav
    @Published var wavPushToTalk = true
    @Published var modelsDir: String = Bundled.modelsDir ?? ""
    @Published var wavs: [String] = Bundled.wavs
    @Published var selectedWav: String?  // nil = 全部內附 wav
    @Published var running = false
    @Published var holding = false
    @Published var utterances: [OzUtterance] = []
    @Published var statusText = "尚未載入模型"
    @Published var loadSeconds: Double?
    @Published var peakRSS = OzMemory.peakResidentMB()
    @Published var currentRSS = OzMemory.currentResidentMB()
    @Published var rtf: Double?
    @Published var errorText: String?

    private let worker = Worker()
    private var memTimer: Timer?

    init() {
        worker.onUtterances = { [weak self] in self?.utterances = $0 }
        worker.onLoaded = { [weak self] in self?.loadSeconds = $0 }
        worker.onStatus = { [weak self] in self?.statusText = $0 }
        worker.onHolding = { [weak self] in self?.holding = $0 }
        worker.onFinished = { [weak self] err, rtf in
            guard let self else { return }
            if let rtf { self.rtf = rtf }
            self.finish(error: err)
        }
        // 自動驗證：--autorun [ptt|stream] [--autorun-out 路徑] [--autorun-limit N] [--autorun-warmup N（預設 1）]
        // 結果預設寫到 Documents/autorun_<模式>.json；只跑內附 wav，絕不開麥克風。
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--autorun") {
            let m = i + 1 < args.count && !args[i + 1].hasPrefix("--") ? args[i + 1] : "ptt"
            let docs = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
            var out = docs + "/autorun_\(m).json"
            if let k = args.firstIndex(of: "--autorun-out"), k + 1 < args.count { out = args[k + 1] }
            var paths = wavs
            if let k = args.firstIndex(of: "--autorun-limit"), k + 1 < args.count, let n = Int(args[k + 1]) { paths = Array(paths.prefix(n)) }
            var warmup = 1
            if let k = args.firstIndex(of: "--autorun-warmup"), k + 1 < args.count, let n = Int(args[k + 1]) { warmup = n }
            mode = .wav
            wavPushToTalk = (m == "ptt")
            startAutoRun(paths: paths, pushToTalk: m == "ptt", out: out, warmup: warmup)
        }
    }

    private func startAutoRun(paths: [String], pushToTalk: Bool, out: String, warmup: Int) {
        guard Bundled.modelsDir != nil, !paths.isEmpty else {
            errorText = "找不到內附模型或 wav"
            try? "{\"type\":\"error\",\"message\":\"no bundled models/wavs\",\"ok\":false}\n".write(toFile: out, atomically: true, encoding: .utf8)
            return
        }
        running = true
        startMemTimer()
        worker.runAutoRun(dir: modelsDir, paths: paths, pushToTalk: pushToTalk, outPath: out, warmup: warmup) {}
    }

    var showsTalkButton: Bool { mode == .pushToTalk || (mode == .wav && wavPushToTalk) }

    var liveUtterance: OzUtterance? {
        guard let u = utterances.last, u.status != .final else { return nil }
        return u
    }

    var selectedWavPaths: [String] {
        if let s = selectedWav { return wavs.filter { ($0 as NSString).lastPathComponent == s } }
        return wavs
    }

    private func startMemTimer() {
        memTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refreshMemory() }
        }
    }

    func start() {
        guard !running else { return }
        errorText = nil
        guard Bundled.modelsDir != nil else { errorText = "app 內沒有模型（請用 scripts/build_ios_app.sh 建置）"; return }
        if mode == .wav && selectedWavPaths.isEmpty { errorText = "app 內沒有 wav"; return }
        if mode == .wav { return begin() }
        // 麥克風模式：先要權限，通過才開輸入裝置
        OzMicrophone.requestPermission { ok in
            DispatchQueue.main.async {
                if ok { self.begin() } else { self.errorText = "沒有麥克風權限（設定 → 隱私權與安全性 → 麥克風）" }
            }
        }
    }

    private func begin() {
        running = true
        utterances = []
        rtf = nil
        startMemTimer()
        switch mode {
        case .wav: worker.runWav(dir: modelsDir, paths: selectedWavPaths, pushToTalk: wavPushToTalk)
        case .pushToTalk: worker.startMic(dir: modelsDir, pushToTalk: true)
        case .continuous: worker.startMic(dir: modelsDir, pushToTalk: false)
        }
    }

    func stop() {
        holding = false
        worker.stop()
    }

    /// 按住說話按鈕：按下。麥克風模式需已開啟麥克風；wav 模式的按鈕只是顯示（由「開始」驅動）。
    func pressStart() {
        guard mode == .pushToTalk, running, !holding else { return }
        holding = true
        statusText = "收音中…放開即定稿"
        worker.press()
    }

    func pressEnd() {
        guard mode == .pushToTalk, holding else { return }
        holding = false
        statusText = "麥克風已開啟：按住按鈕說話"
        worker.release()
    }

    private func refreshMemory() {
        peakRSS = OzMemory.peakResidentMB()
        currentRSS = OzMemory.currentResidentMB()
    }

    private func finish(error: String?) {
        running = false
        holding = false
        memTimer?.invalidate()
        refreshMemory()
        if let error { errorText = error }
        statusText = error == nil ? "完成" : "已停止"
    }
}
