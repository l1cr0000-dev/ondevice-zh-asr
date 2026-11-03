import Foundation
import OzAsrKit

private func jsonLine(_ obj: [String: Any]) {
    guard let d = try? JSONSerialization.data(withJSONObject: obj, options: [.withoutEscapingSlashes]),
        let s = String(data: d, encoding: .utf8)
    else { return }
    print(s)
}

private let headlessUsage = """
    用法：OzAsrMac --headless [--models DIR] [--list FILE] [--tail-silence-ms N] [--no-realtime] [--warmup N] <wav>...
      依 wav 實際時長、以即時節奏（牆鐘）送入 10 ms chunk，每個檔案輸出一行 JSON，最後一行 summary。
      --tail-silence-ms  每個檔案後補的靜音（預設 1000，模擬麥克風持續收音，讓 700 ms 定稿門檻在牆鐘下觸發）
      --ptt              按住說話模擬：每個 wav 當作一次「按住」，音訊結束的瞬間「放開」並強制定稿（不補靜音）；
                         額外輸出放開到定稿的牆鐘延遲 release_to_final_latency_s
      --ptt-release-after-speech-ms N
                         搭配 --ptt：先以不等待的方式跑一遍取得最後一句的語音終點，再把音訊截到「語音終點 + N ms」才放開
                         （N 小於 150 時放開瞬間尚未出暫定結果，量到的是放開時才解碼整句的最壞延遲）
      --no-realtime      不等待、盡快送入（只能比對文字，牆鐘延遲沒有意義）
      --warmup N         前 N 個檔案以 --no-realtime 暖機（不輸出、不計入 summary）

    """

func runHeadless(_ args: [String]) -> Int32 {
    var models: String?
    var wavs: [String] = []
    var tailMs = 1000, warmup = 0
    var realtime = true
    var ptt = false
    var releaseAfterMs: Int?
    var i = 0
    func val() -> String { i += 1; return i < args.count ? args[i] : "" }
    while i < args.count {
        switch args[i] {
        case "--headless": break
        case "--models": models = val()
        case "--list":
            if let t = try? String(contentsOfFile: val(), encoding: .utf8) {
                wavs += t.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
            }
        case "--tail-silence-ms": tailMs = Int(val()) ?? 1000
        case "--warmup": warmup = Int(val()) ?? 0
        case "--no-realtime": realtime = false
        case "--ptt": ptt = true
        case "--ptt-release-after-speech-ms": releaseAfterMs = Int(val())
        case "-h", "--help": print(headlessUsage); return 0
        default: wavs.append(args[i])
        }
        i += 1
    }
    guard !wavs.isEmpty else { FileHandle.standardError.write(Data(headlessUsage.utf8)); return 2 }
    guard let dir = ModelLocator.locate(explicit: models) else {
        FileHandle.standardError.write(Data("找不到模型目錄（用 --models 或 OZASR_MODELS_DIR 指定）\n".utf8))
        return 1
    }
    do {
        let rec = try OzAsrRecognizer(config: OzAsrConfig(modelsDirectory: dir))
        let rssLoaded = OzMemory.currentResidentMB()
        var totalAudio = 0.0, totalCompute = 0.0, counted = 0
        var sumRelease = 0.0, nRelease = 0
        var sumFirst = 0.0, nFirst = 0, sumFinal = 0.0, nFinal = 0
        for (f, path) in wavs.enumerated() {
            var audio = try OzWav.load(path: path)
            if !ptt { audio += [Float](repeating: 0, count: tailMs * OzWav.sampleRate / 1000) }
            if ptt, let after = releaseAfterMs {
                // 先跑一遍（不等待）找最後一句的語音終點，再截斷音訊
                try rec.reset()
                var lastEnd = 0.0
                try OzPacedFeeder.feed(recognizer: rec, samples: audio, realtime: false) { evs in
                    for e in evs { if let end = e.speechEnd { lastEnd = end } }
                }
                let keep = Int((lastEnd + Double(after) / 1000) * Double(OzWav.sampleRate))
                if keep < audio.count { audio = Array(audio[0..<keep]) }
            }
            try rec.reset()
            let counting = f >= warmup
            var tracker = OzUtteranceTracker(streamOrigin: 0)
            var events: [[String: Any]] = []
            let wall0 = OzClock.now()
            let session = OzPushToTalkSession(recognizer: rec)
            let record: ([OzAsrEvent]) -> Void = { evs in
                    for e in evs {
                        if !ptt { tracker.apply(e) }
                        let origin = ptt ? session.tracker.streamOrigin : tracker.streamOrigin
                        events.append([
                            "type": "\(e.kind)", "utterance": e.utterance, "text": e.text,
                            "wall_s": e.wallTime - origin, "simulated_s": e.simulatedEmitted,
                            "audio_pos_s": e.audioPos, "speech_start_s": e.speechStart,
                            "speech_end_s": e.speechEnd ?? NSNull(),
                        ])
                    }
            }
            if ptt {
                try OzPacedFeeder.feedPushToTalk(
                    session: session, samples: audio, realtime: realtime && counting, onEvents: record)
                tracker = session.tracker
            } else {
                try OzPacedFeeder.feed(
                    recognizer: rec, samples: audio, realtime: realtime && counting,
                    onOrigin: { tracker.streamOrigin = $0 }, onEvents: record)
            }
            let wall = OzClock.now() - wall0
            guard counting else { continue }
            let dur = Double(audio.count) / Double(OzWav.sampleRate)
            totalAudio += dur
            totalCompute += rec.computeSeconds
            counted += 1
            let utts: [[String: Any]] = tracker.utterances.map { u in
                if let v = u.firstTokenLatency { sumFirst += v; nFirst += 1 }
                if let v = u.finalLatency { sumFinal += v; nFinal += 1 }
                if let v = u.releaseToFinalLatency { sumRelease += v; nRelease += 1 }
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
            jsonLine([
                "type": "file", "wav": path, "duration_s": dur, "wall_s": wall,
                "compute_s": rec.computeSeconds, "events": events, "utterances": utts,
            ])
        }
        jsonLine([
            "type": "summary", "files": counted, "audio_s": totalAudio, "compute_s": totalCompute,
            "rtf": totalAudio > 0 ? totalCompute / totalAudio : NSNull(), "load_s": rec.loadSeconds,
            "rss_after_load_mb": rssLoaded, "peak_rss_mb": OzMemory.peakResidentMB(),
            "mean_first_token_latency_s": nFirst > 0 ? sumFirst / Double(nFirst) : NSNull(),
            "mean_final_latency_s": nFinal > 0 ? sumFinal / Double(nFinal) : NSNull(),
            "mean_release_to_final_latency_s": nRelease > 0 ? sumRelease / Double(nRelease) : NSNull(),
            "ptt": ptt, "ptt_release_after_speech_ms": releaseAfterMs ?? NSNull(), "realtime": realtime, "tail_silence_ms": tailMs, "version": OzAsrRecognizer.version,
        ])
        return 0
    } catch {
        FileHandle.standardError.write(Data("錯誤：\(error)\n".utf8))
        return 1
    }
}
