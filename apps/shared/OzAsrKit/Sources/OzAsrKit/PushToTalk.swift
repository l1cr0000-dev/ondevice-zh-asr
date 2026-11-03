import Foundation

/// 按住說話：按住期間收音並即時出 partial（VAD 與 A2 照常，句中停頓仍出暫定結果）；
/// 放開立刻定稿（`finalizeUtterance`），不等靜音門檻。放開期間的音訊不送入，音訊時間軸不前進。
/// 非執行緒安全：press / accept / release 請在同一個序列佇列呼叫（麥克風 chunk 也 dispatch 到同一佇列，順序才正確）。
public final class OzPushToTalkSession {
    public let recognizer: OzAsrRecognizer
    public private(set) var tracker: OzUtteranceTracker
    public private(set) var isPressed = false
    private var samplesFed = 0

    public init(recognizer: OzAsrRecognizer) {
        self.recognizer = recognizer
        tracker = OzUtteranceTracker(streamOrigin: OzClock.now())
    }

    public func press(wall: TimeInterval = OzClock.now()) {
        guard !isPressed else { return }
        isPressed = true
        tracker.beginSegment(audio: Double(samplesFed) / Double(OzWav.sampleRate), wall: wall)
    }

    /// 按住期間的 chunk；沒按住時直接丟棄。回傳這次的事件（已套用到 tracker）。
    @discardableResult
    public func accept(_ chunk: [Float]) throws -> [OzAsrEvent] {
        guard isPressed else { return [] }
        samplesFed += chunk.count
        let evs = try recognizer.acceptWaveform(chunk)
        for e in evs { tracker.apply(e) }
        return evs
    }

    /// 放開：立刻定稿。`wall` 是放開的牆鐘（預設現在）。
    @discardableResult
    public func release(wall: TimeInterval = OzClock.now()) throws -> [OzAsrEvent] {
        guard isPressed else { return [] }
        isPressed = false
        tracker.markRelease(wall: wall)
        let evs = try recognizer.finalizeUtterance()
        for e in evs { tracker.apply(e) }
        return evs
    }
}

extension OzPacedFeeder {
    /// 用 wav 模擬按住說話：開始即按下，依即時節奏送入全部音訊，音訊結束的瞬間放開（不補靜音）。
    public static func feedPushToTalk(
        session: OzPushToTalkSession, samples: [Float], realtime: Bool = true,
        shouldStop: () -> Bool = { false }, onEvents: ([OzAsrEvent]) -> Void
    ) throws {
        let origin = OzClock.now()
        session.press(wall: origin)
        var i = 0, k = 0
        while i < samples.count {
            if shouldStop() { break }
            let n = min(chunkSamples, samples.count - i)
            if realtime {
                let due = origin + Double(k + 1) * Double(chunkSamples) / Double(OzWav.sampleRate)
                let wait = due - OzClock.now()
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
            let evs = try session.accept(Array(samples[i..<i + n]))
            if !evs.isEmpty { onEvents(evs) }
            i += n
            k += 1
        }
        let evs = try session.release()
        if !evs.isEmpty { onEvents(evs) }
    }
}
