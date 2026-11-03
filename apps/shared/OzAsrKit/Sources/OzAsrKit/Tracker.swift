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
    var simulatedStart = 0.0
}

/// 把事件流整理成句子清單並計算牆鐘延遲。
/// `streamOrigin` = 第一個樣本送入時的牆鐘；音訊時間 t 對應牆鐘 `streamOrigin + t`。
/// wav 模式是精確的（依即時節奏送入）；麥克風模式忽略系統收音緩衝（約數十 ms），為近似值。
public struct OzUtteranceTracker: Sendable {
    public private(set) var utterances: [OzUtterance] = []
    public var streamOrigin: TimeInterval

    public init(streamOrigin: TimeInterval) { self.streamOrigin = streamOrigin }

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
            utterances[i].speechEnd = e.speechEnd
            if let end = e.speechEnd {
                utterances[i].finalLatency = e.wallTime - (streamOrigin + end)
                utterances[i].simulatedFinalLatency = e.simulatedEmitted - end
            }
        }
        utterances[i].text = e.text
        if utterances[i].firstTokenLatency == nil, !e.text.isEmpty {
            utterances[i].firstTokenLatency = e.wallTime - (streamOrigin + e.speechStart)
            utterances[i].simulatedFirstTokenLatency = e.simulatedEmitted - e.speechStart
        }
    }
}
