import Foundation

/// 一個句子的顯示狀態與牆鐘延遲。
public struct OzUtterance: Identifiable, Sendable {
    public enum Status: Sendable { case listening, partial, provisional, final }
    public var id: Int
    public var text = ""
    public var status = Status.listening
    /// VAD 起點（音訊時間軸，秒）與語音終點（定稿時才有）。
    public var speechStart = 0.0
    public var speechEnd: Double?
    /// 首字延遲：第一個非空文字的牆鐘時刻 − 語音起點的牆鐘時刻（秒）。
    public var firstTokenLatency: Double?
    /// 尾字延遲：定稿的牆鐘時刻 − 語音終點的牆鐘時刻（秒）。
    public var finalLatency: Double?
    /// 定稿（或最後一次更新）的牆鐘時刻 − 語音終點的音訊時間；模擬值對照用。
    public var simulatedFirstTokenLatency: Double?
    public var simulatedFinalLatency: Double?
    /// 按住說話：放開（`markRelease`）到定稿事件的牆鐘（秒）。連續收音模式為 nil。
    public var releaseToFinalLatency: Double?
    var simulatedStart = 0.0
}

/// 把事件流整理成句子清單並計算牆鐘延遲。
/// `streamOrigin` = 第一個樣本送入時的牆鐘；音訊時間 t 對應牆鐘 `streamOrigin + t`。
/// wav 模式是精確的（依即時節奏送入）；麥克風模式忽略系統收音緩衝（約數十 ms），為近似值。
public struct OzUtteranceTracker: Sendable {
    public private(set) var utterances: [OzUtterance] = []
    /// 第一段 = (音訊時間 0, 串流起點牆鐘)；按住說話每次按下會新增一段（音訊時間軸在放開期間不前進）。
    private var segments: [(audio: Double, wall: TimeInterval)]
    private var releaseWall: TimeInterval?

    public var streamOrigin: TimeInterval {
        get { segments[0].wall }
        set { segments[0].wall = newValue }
    }

    public init(streamOrigin: TimeInterval) { segments = [(0, streamOrigin)] }

    /// 音訊時間 t（秒）對應的牆鐘。
    public func wallTime(ofAudio t: Double) -> TimeInterval {
        let seg = segments.last(where: { $0.audio <= t }) ?? segments[0]
        return seg.wall + (t - seg.audio)
    }

    /// 按住說話：按下。`audio` = 此刻已送入辨識器的音訊長度（秒）。
    public mutating func beginSegment(audio: Double, wall: TimeInterval) {
        if let last = segments.last, last.audio == audio { segments[segments.count - 1].wall = wall; return }
        segments.append((audio, wall))
        releaseWall = nil
    }

    /// 按住說話：放開的牆鐘（在呼叫 finalize 之前標記）。
    public mutating func markRelease(wall: TimeInterval) { releaseWall = wall }

    public mutating func apply(_ e: OzAsrEvent) {
        if utterances.last?.id != e.utterance {
            utterances.append(OzUtterance(id: e.utterance))
        }
        let i = utterances.count - 1
        utterances[i].speechStart = e.speechStart
        utterances[i].simulatedStart = e.speechStart
        switch e.kind {
        case .start:
            utterances[i].status = .listening
            return
        case .partial: utterances[i].status = .partial
        case .provisional: utterances[i].status = .provisional
        case .final:
            utterances[i].status = .final
            if let r = releaseWall { utterances[i].releaseToFinalLatency = e.wallTime - r }
            releaseWall = nil
            utterances[i].speechEnd = e.speechEnd
            if let end = e.speechEnd {
                utterances[i].finalLatency = e.wallTime - wallTime(ofAudio: end)
                utterances[i].simulatedFinalLatency = e.simulatedEmitted - end
            }
        }
        utterances[i].text = e.text
        if utterances[i].firstTokenLatency == nil, !e.text.isEmpty {
            utterances[i].firstTokenLatency = e.wallTime - wallTime(ofAudio: e.speechStart)
            utterances[i].simulatedFirstTokenLatency = e.simulatedEmitted - e.speechStart
        }
    }
}
