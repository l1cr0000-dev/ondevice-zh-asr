import OzAsrKit
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel
    init(model: AppModel) { self.model = model }

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider()
            liveCard
            if model.showsTalkButton { talkButton }
            Divider()
            history
            Divider()
            footer
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("模式", selection: $model.mode) {
                ForEach(InputMode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(model.running)
            HStack {
                if model.mode == .wav {
                    Toggle("模擬按住說話", isOn: $model.wavPushToTalk).disabled(model.running).fixedSize()
                    Menu(model.selectedWav ?? "全部 wav（\(model.wavs.count)）") {
                        Button("全部 wav（\(model.wavs.count)）") { model.selectedWav = nil }
                        ForEach(model.wavs.map { ($0 as NSString).lastPathComponent }, id: \.self) { n in
                            Button(n) { model.selectedWav = n }
                        }
                    }
                    .disabled(model.running)
                }
                Spacer()
                if model.running {
                    Button(model.mode == .wav ? "停止" : "關閉麥克風") { model.stop() }.buttonStyle(.bordered)
                } else {
                    Button(model.mode == .wav ? "開始" : "開啟麥克風") { model.start() }.buttonStyle(.borderedProminent)
                }
            }
            if let e = model.errorText { Text(e).foregroundStyle(.red).font(.callout) }
        }
        .padding(12)
    }

    private var liveCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("即時").font(.headline)
                Text(model.statusText).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Group {
                if let u = model.liveUtterance {
                    switch u.status {
                    case .provisional: Text(u.text + "（暫定）").foregroundStyle(.orange)
                    case .partial: Text(u.text).foregroundStyle(.secondary)
                    default: Text("…").foregroundStyle(.tertiary)
                    }
                } else if let last = model.utterances.last {
                    Text(last.text).foregroundStyle(.primary)
                } else if model.mode == .pushToTalk && model.running {
                    Text("按住說話").foregroundStyle(.tertiary)
                } else {
                    Text(" ")
                }
            }
            .font(.system(size: 24, weight: .medium))
            .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
        }
        .padding(12)
    }

    /// 大按鈕：按住收音、放開立刻定稿（觸控按住）
    private var talkButton: some View {
        let on = model.holding
        let usable = model.mode == .pushToTalk && model.running
        return Text(on ? "放開結束" : (model.mode == .wav ? "按住說話（wav 模擬）" : (usable ? "按住說話" : "先開啟麥克風")))
            .font(.system(size: 22, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, minHeight: 84)
            .background(RoundedRectangle(cornerRadius: 16).fill(on ? Color.red : (usable || model.mode == .wav ? Color.accentColor : Color.gray)))
            .padding(.horizontal, 12).padding(.bottom, 12)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in model.pressStart() }
                    .onEnded { _ in model.pressEnd() })
    }

    private var history: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(model.utterances.filter { $0.status != .listening || !$0.text.isEmpty }.reversed()) { u in
                    HStack(alignment: .firstTextBaseline) {
                        Text("#\(u.id + 1)").foregroundStyle(.tertiary).frame(width: 34, alignment: .leading)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(u.status == .provisional ? u.text + "（暫定）" : u.text)
                                .font(.title3)
                                .foregroundStyle(u.status == .final ? Color.primary : (u.status == .provisional ? .orange : .secondary))
                            HStack(spacing: 10) {
                                Text("首字 \(fmt(u.firstTokenLatency))")
                                if let r = u.releaseToFinalLatency { Text("放開→定稿 \(fmt(r))") } else { Text("尾字 \(fmt(u.finalLatency))") }
                            }
                            .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var footer: some View {
        HStack(spacing: 14) {
            stat("載入", model.loadSeconds.map { String(format: "%.2f s", $0) } ?? "—")
            stat("峰值記憶體", String(format: "%.0f MB", model.peakRSS))
            stat("目前", String(format: "%.0f MB", model.currentRSS))
            stat("RTF", model.rtf.map { String(format: "%.3f", $0) } ?? "—")
            Spacer()
        }
        .padding(10)
    }

    private func stat(_ k: String, _ v: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(k).font(.caption).foregroundStyle(.secondary)
            Text(v).font(.system(.callout, design: .monospaced))
        }
    }

    private func fmt(_ v: Double?) -> String { v.map { String(format: "%.0f ms", $0 * 1000) } ?? "—" }
}
