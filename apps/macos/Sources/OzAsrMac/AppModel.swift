import AVFoundation
import AppKit
import SwiftUI
import Foundation
import OzAsrKit

enum InputMode: String, CaseIterable, Identifiable {
    case wav = "wav 檔"
    case microphone = "麥克風"
    var id: String { rawValue }
}

/// 跨執行緒的停止旗標
final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var mode = InputMode.wav
    @Published var modelsDir: String = ModelLocator.locate() ?? ""
    @Published var wavPaths: [String] = []
    @Published var running = false
    @Published var utterances: [OzUtterance] = []
    @Published var statusText = "尚未載入模型"
    @Published var loadSeconds: Double?
    @Published var peakRSS = OzMemory.peakResidentMB()
    @Published var currentRSS = OzMemory.currentResidentMB()
    @Published var rtf: Double?
    @Published var errorText: String?

    private var recognizer: OzAsrRecognizer?
    private let queue = DispatchQueue(label: "ozasr.engine", qos: .userInitiated)
    private var stopFlag = StopFlag()
    private var mic: OzMicrophone?
    private var memTimer: Timer?
    private var audioSeconds = 0.0

    init() {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--models"), i + 1 < args.count { modelsDir = args[i + 1] }
        var j = 0
        while j < args.count {
            if args[j] == "--autorun-wav", j + 1 < args.count { wavPaths.append(args[j + 1]); j += 1 }
            j += 1
        }
        // 文件截圖用：--snapshot-after 秒數 --snapshot-to 路徑；把自己的視窗內容存成 PNG 後結束（不需要螢幕錄製權限）
        if let i = args.firstIndex(of: "--snapshot-to"), i + 1 < args.count {
            let path = args[i + 1]
            var after = 10.0
            if let k = args.firstIndex(of: "--snapshot-after"), k + 1 < args.count { after = Double(args[k + 1]) ?? 10 }
            DispatchQueue.main.asyncAfter(deadline: .now() + after) { [self] in
                let model = self
                let r = ImageRenderer(content: ContentView(model: model).frame(width: 760, height: 580)
                    .background(Color(nsColor: .windowBackgroundColor)).environment(\.colorScheme, .light))
                r.scale = 1.5
                if let img = r.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) {
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                }
                NSApp.terminate(nil)
            }
        }
        if !wavPaths.isEmpty {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in self?.start() }
        }
    }

    var liveUtterance: OzUtterance? {
        guard let u = utterances.last, u.status != .final else { return nil }
        return u
    }

    func chooseModels() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.message = "選擇模型根目錄（含 silero-vad-v6.2.3 與 sherpa-onnx-sense-voice-…）"
        if p.runModal() == .OK, let u = p.url { modelsDir = u.path }
    }

    func chooseWavs() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.wav]
        p.allowsMultipleSelection = true
        if p.runModal() == .OK { wavPaths = p.urls.map(\.path) }
    }

    func start() {
        guard !running else { return }
        errorText = nil
        guard ModelLocator.valid(modelsDir) else {
            errorText = "找不到模型：\(modelsDir.isEmpty ? "未設定" : modelsDir)"
            return
        }
        if mode == .wav && wavPaths.isEmpty { errorText = "請先選擇 wav 檔"; return }
        running = true
        utterances = []
        rtf = nil
        stopFlag = StopFlag()
        memTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refreshMemory() }
        }
        mode == .wav ? startWav() : startMic()
    }

    func stop() {
        stopFlag.set()
        if mode == .microphone { stopMic() }
    }

    private func refreshMemory() {
        peakRSS = OzMemory.peakResidentMB()
        currentRSS = OzMemory.currentResidentMB()
    }

    private func finish(error: String? = nil) {
        running = false
        memTimer?.invalidate()
        refreshMemory()
        if let error { errorText = error }
        statusText = "已停止"
    }

    /// 在 engine 佇列載入模型（只載一次）
    private func ensureRecognizer() throws -> OzAsrRecognizer {
        if let r = recognizer { try r.reset(); return r }
        let t = OzClock.now()
        let r = try OzAsrRecognizer(config: OzAsrConfig(modelsDirectory: modelsDir))
        _ = t
        recognizer = r
        return r
    }

    // MARK: wav 模式（依實際時長以即時節奏送入，牆鐘計時）

    private func startWav() {
        let paths = wavPaths
        let flag = stopFlag
        let dir = modelsDir
        statusText = "載入模型…"
        queue.async { [weak self] in
            do {
                let hadModel = DispatchQueue.main.sync { self?.recognizer != nil }
                let rec: OzAsrRecognizer
                if hadModel {
                    rec = try DispatchQueue.main.sync { try self!.ensureRecognizer() }
                } else {
                    rec = try OzAsrRecognizer(config: OzAsrConfig(modelsDirectory: dir))
                    DispatchQueue.main.async {
                        self?.recognizer = rec
                        self?.loadSeconds = rec.loadSeconds
                    }
                }
                // 多個檔案接成一條連續的串流，每個檔案後補 1 s 靜音（和麥克風持續收音一樣，讓定稿門檻在牆鐘下觸發）
                var audio: [Float] = []
                for p in paths {
                    audio += try OzWav.load(path: p)
                    audio += [Float](repeating: 0, count: OzWav.sampleRate)
                }
                let total = Double(audio.count) / Double(OzWav.sampleRate)
                var tracker = OzUtteranceTracker(streamOrigin: OzClock.now())
                DispatchQueue.main.async { self?.statusText = "辨識中（即時節奏送入 \(String(format: "%.1f", total)) s）" }
                let origin = try OzPacedFeeder.feed(
                    recognizer: rec, samples: audio, realtime: true, shouldStop: { flag.isSet },
                    onOrigin: { tracker.streamOrigin = $0 },
                    onEvents: { evs in
                        for e in evs { tracker.apply(e) }
                        let snap = tracker.utterances
                        DispatchQueue.main.async { self?.utterances = snap }
                    })
                _ = origin
                let compute = rec.computeSeconds
                DispatchQueue.main.async {
                    self?.rtf = compute / total
                    self?.finish()
                    self?.statusText = flag.isSet ? "已停止" : "完成"
                }
            } catch {
                DispatchQueue.main.async { self?.finish(error: "\(error)") }
            }
        }
    }

    // MARK: 麥克風模式（只有使用者按「開始」才會開啟輸入裝置）

    private func startMic() {
        let flag = stopFlag
        let dir = modelsDir
        statusText = "載入模型…"
        queue.async { [weak self] in
            do {
                let rec: OzAsrRecognizer
                if let existing = DispatchQueue.main.sync(execute: { self?.recognizer }) {
                    try existing.reset()
                    rec = existing
                } else {
                    rec = try OzAsrRecognizer(config: OzAsrConfig(modelsDirectory: dir))
                    DispatchQueue.main.async {
                        self?.recognizer = rec
                        self?.loadSeconds = rec.loadSeconds
                    }
                }
                var tracker = OzUtteranceTracker(streamOrigin: OzClock.now())
                var started = false
                let m = OzMicrophone()
                try m.start { chunk in
                    self?.queue.async {
                        if flag.isSet { return }
                        if !started { tracker.streamOrigin = OzClock.now() - 0.01; started = true }
                        guard let evs = try? rec.acceptWaveform(chunk) else { return }
                        if evs.isEmpty { return }
                        for e in evs { tracker.apply(e) }
                        let snap = tracker.utterances
                        DispatchQueue.main.async { self?.utterances = snap }
                    }
                }
                DispatchQueue.main.async {
                    self?.mic = m
                    self?.statusText = "聆聽中…"
                }
            } catch {
                DispatchQueue.main.async { self?.finish(error: "\(error)") }
            }
        }
    }

    private func stopMic() {
        mic?.stop()
        mic = nil
        queue.async { [weak self] in
            // 讓排在佇列前面的 chunk 先處理完，再 flush 結束進行中的句子
            DispatchQueue.main.async {
                self?.finish()
            }
        }
    }
}
