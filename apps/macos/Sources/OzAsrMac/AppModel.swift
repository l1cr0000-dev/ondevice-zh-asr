import AVFoundation
import AppKit
import Foundation
import OzAsrKit
import SwiftUI

enum InputMode: String, CaseIterable, Identifiable {
    case pushToTalk = "按住說話"
    case continuous = "連續收音"
    case wav = "wav 檔"
    var id: String { rawValue }
}

/// 跨執行緒的停止旗標
final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}

/// 辨識工作執行緒：recognizer、session、tracker 都只在 `queue` 上存取（C 核心非執行緒安全）。
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
                var rtf: Double?
                if pushToTalk {
                    // 每個檔案當作一次「按住」，音訊結束時「放開」並強制定稿（不補靜音）
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
                    // 多個檔案接成一條連續串流，每個檔案後補 1 s 靜音（讓 700 ms 定稿門檻在牆鐘下觸發）
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
                if total > 0 { rtf = rec.computeSeconds / total }
                self.main { self.onFinished?(nil, rtf) }
            } catch {
                self.main { self.onHolding?(false); self.onFinished?("\(error)", nil) }
            }
        }
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
                    // 音訊執行緒只 dispatch；按下 / 放開 / chunk 都在同一佇列，順序才正確
                    self.queue.async { self.handleChunk(chunk, pushToTalk: pushToTalk) }
                }
                self.mic = m
                self.main {
                    self.onStatus?(pushToTalk ? "麥克風已開啟：按住按鈕或空白鍵說話" : "聆聽中…")
                }
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

    func press() {
        queue.async { self.session?.press() }
    }

    func release() {
        queue.async {
            do { _ = try self.session?.release() } catch { self.main { self.onStatus?("錯誤：\(error)") } }
            self.publish()
        }
    }

    /// 停止：wav 中止；麥克風關閉。連續收音會 flush 進行中的句子；按住說話若還按著則先放開定稿。
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
    @Published var mode = InputMode.pushToTalk
    @Published var wavPushToTalk = true
    @Published var modelsDir: String = ModelLocator.locate() ?? ""
    @Published var wavPaths: [String] = []
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
    private var keyMonitor: Any?

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
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--models"), i + 1 < args.count { modelsDir = args[i + 1] }
        var j = 0
        while j < args.count {
            if args[j] == "--autorun-wav", j + 1 < args.count { wavPaths.append(args[j + 1]); j += 1 }
            j += 1
        }
        if args.contains("--autorun-continuous") { wavPushToTalk = false }
        if !wavPaths.isEmpty { mode = .wav }
        installSpaceKey()
        // 文件截圖用：--snapshot-after 秒數 --snapshot-to 路徑；把介面渲染成 PNG 後結束（不需要螢幕錄製權限）
        if let i = args.firstIndex(of: "--snapshot-to"), i + 1 < args.count {
            let path = args[i + 1]
            var after = 10.0
            if let k = args.firstIndex(of: "--snapshot-after"), k + 1 < args.count { after = Double(args[k + 1]) ?? 10 }
            DispatchQueue.main.asyncAfter(deadline: .now() + after) { [self] in
                let model = self
                let r = ImageRenderer(content: ContentView(model: model).frame(width: 760, height: 640)
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

    /// 空白鍵按住說話（只在按住說話模式且麥克風已開啟時作用；放開即定稿）
    private func installSpaceKey() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] e in
            guard e.keyCode == 49, let self, self.mode == .pushToTalk, self.running else { return e }
            if e.type == .keyDown { if !e.isARepeat { self.pressStart() } } else { self.pressEnd() }
            return nil
        }
    }

    var showsTalkButton: Bool { mode == .pushToTalk || (mode == .wav && wavPushToTalk) }

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
        memTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refreshMemory() }
        }
        switch mode {
        case .wav: worker.runWav(dir: modelsDir, paths: wavPaths, pushToTalk: wavPushToTalk)
        case .pushToTalk: worker.startMic(dir: modelsDir, pushToTalk: true)
        case .continuous: worker.startMic(dir: modelsDir, pushToTalk: false)
        }
    }

    func stop() {
        holding = false
        worker.stop()
    }

    /// 按下（按鈕或空白鍵）。只有按住說話模式且麥克風已開啟才有作用。
    func pressStart() {
        guard mode == .pushToTalk, running, !holding else { return }
        holding = true
        statusText = "收音中…放開即定稿"
        worker.press()
    }

    func pressEnd() {
        guard mode == .pushToTalk, holding else { return }
        holding = false
        statusText = "麥克風已開啟：按住按鈕或空白鍵說話"
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
