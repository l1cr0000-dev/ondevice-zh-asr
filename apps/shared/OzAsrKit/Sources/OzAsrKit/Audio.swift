import Foundation

/// 最小的 wav 讀取：PCM 16/24/32 與 float32，多聲道取平均；取樣率非 16 kHz 時用線性內插重採樣。
public enum OzWav {
    public static let sampleRate = 16000

    public static func load(path: String) throws -> [Float] {
        guard let d = FileManager.default.contents(atPath: path) else { throw OzAsrError.io("找不到音訊檔：\(path)") }
        let b = [UInt8](d)
        func u16(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        guard b.count >= 12, Array(b[0..<4]) == Array("RIFF".utf8), Array(b[8..<12]) == Array("WAVE".utf8) else {
            throw OzAsrError.io("不是 RIFF/WAVE：\(path)")
        }
        var fmt = 0, ch = 0, rate = 0, bits = 0, off = 0, len = 0
        var o = 12
        while o + 8 <= b.count {
            let l = u32(o + 4)
            let tag = String(decoding: b[o..<o + 4], as: UTF8.self)
            if tag == "fmt " {
                fmt = u16(o + 8); ch = u16(o + 10); rate = u32(o + 12); bits = u16(o + 22)
                if fmt == 0xFFFE && l >= 26 { fmt = u16(o + 8 + 24) }
            } else if tag == "data" {
                off = o + 8; len = min(l, b.count - off); break
            }
            o += 8 + l + (l & 1)
        }
        guard off > 0, ch > 0, bits > 0 else { throw OzAsrError.io("wav 缺少 fmt / data：\(path)") }
        let bytes = bits / 8
        let frames = len / (bytes * ch)
        var mono = [Float](repeating: 0, count: frames)
        for f in 0..<frames {
            var acc: Float = 0
            for c in 0..<ch {
                let p = off + (f * ch + c) * bytes
                switch (fmt, bits) {
                case (1, 16): acc += Float(Int16(bitPattern: UInt16(u16(p)))) / 32768
                case (1, 24):
                    var v = Int32(b[p]) | Int32(b[p + 1]) << 8 | Int32(b[p + 2]) << 16
                    if v & 0x800000 != 0 { v -= 0x1000000 }
                    acc += Float(v) / 8388608
                case (1, 32): acc += Float(Int32(bitPattern: UInt32(u32(p)))) / 2147483648
                case (3, 32): acc += Float(bitPattern: UInt32(u32(p)))
                default: throw OzAsrError.io("不支援的 wav 格式 fmt=\(fmt) bits=\(bits)：\(path)")
                }
            }
            mono[f] = acc / Float(ch)
        }
        if rate == sampleRate { return mono }
        // 線性內插重採樣（測試素材通常已是 16 kHz；這只是保底）
        let n = Int(Double(frames) * Double(sampleRate) / Double(rate))
        return (0..<n).map { i in
            let x = Double(i) * Double(rate) / Double(sampleRate)
            let i0 = min(Int(x), frames - 1), i1 = min(i0 + 1, frames - 1)
            let fr = Float(x - Double(i0))
            return mono[i0] * (1 - fr) + mono[i1] * fr
        }
    }
}

/// 把音訊依實際時長、以即時節奏（牆鐘）逐 chunk 送進辨識器，不是一次灌入。
/// 用絕對 deadline（起點 + k × chunk），某次推論較慢時後面的 chunk 會立刻補上，與真實收音的行為一致。
public enum OzPacedFeeder {
    public static let chunkSamples = 160  // 10 ms

    /// - Parameters:
    ///   - realtime: false = 不等待（盡快送入，只用於不需要牆鐘延遲的比對）
    ///   - shouldStop: 回傳 true 時中止（不 flush）
    /// - Returns: 第一個樣本送入前的牆鐘（`OzUtteranceTracker.streamOrigin`）
    @discardableResult
    public static func feed(
        recognizer: OzAsrRecognizer, samples: [Float], realtime: Bool = true,
        shouldStop: () -> Bool = { false }, onOrigin: (TimeInterval) -> Void = { _ in },
        onEvents: ([OzAsrEvent]) -> Void
    ) throws -> TimeInterval {
        let origin = OzClock.now()
        onOrigin(origin)
        var i = 0
        var k = 0
        while i < samples.count {
            if shouldStop() { return origin }
            let n = min(chunkSamples, samples.count - i)
            if realtime {
                // chunk k 在 origin + (k+1) × 10 ms 才「收齊」
                let due = origin + Double(k + 1) * Double(chunkSamples) / Double(OzWav.sampleRate)
                let wait = due - OzClock.now()
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
            let evs = try samples.withUnsafeBufferPointer {
                try recognizer.acceptWaveform(UnsafeBufferPointer(rebasing: $0[i..<i + n]))
            }
            if !evs.isEmpty { onEvents(evs) }
            i += n
            k += 1
        }
        if shouldStop() { return origin }
        let evs = try recognizer.flush()
        if !evs.isEmpty { onEvents(evs) }
        return origin
    }
}
