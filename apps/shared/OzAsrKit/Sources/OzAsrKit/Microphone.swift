#if canImport(AVFoundation)
import AVFoundation
import Foundation

/// AVAudioEngine 麥克風收音：重採樣成 16 kHz mono float32，以 10 ms（160 樣本）chunk 回呼。
/// 回呼在 AVAudioEngine 的音訊執行緒上；請在回呼裡只做 dispatch，不要在這裡做辨識。
/// iOS 使用時需先設定 AVAudioSession（.playAndRecord 或 .record）；macOS 需 NSMicrophoneUsageDescription。
public final class OzMicrophone {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var pending: [Float] = []
    private let target = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    public private(set) var isRunning = false

    public init() {}

    public func start(onChunk: @escaping ([Float]) -> Void) throws {
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, let conv = AVAudioConverter(from: inFormat, to: target) else {
            throw OzAsrError.runtime("無法建立輸入格式轉換（沒有可用的輸入裝置？）")
        }
        converter = conv
        pending.removeAll()
        input.installTap(onBus: 0, bufferSize: 1024, format: inFormat) { [weak self] buf, _ in
            guard let self, let conv = self.converter else { return }
            let cap = AVAudioFrameCount(Double(buf.frameLength) * 16000 / inFormat.sampleRate) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: self.target, frameCapacity: cap) else { return }
            var supplied = false
            var err: NSError?
            conv.convert(to: out, error: &err) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return buf
            }
            guard err == nil, let ch = out.floatChannelData?[0] else { return }
            self.pending.append(contentsOf: UnsafeBufferPointer(start: ch, count: Int(out.frameLength)))
            let n = OzPacedFeeder.chunkSamples
            var off = 0
            while self.pending.count - off >= n {
                onChunk(Array(self.pending[off..<off + n]))
                off += n
            }
            if off > 0 { self.pending.removeFirst(off) }
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    public func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
    }
}
#endif
