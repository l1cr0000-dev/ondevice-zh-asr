import OzAsrKit
import SwiftUI


struct OzAsrMacApp: App {
    var body: some Scene {
        WindowGroup("端側中文 ASR") { RootView() }
            .windowResizability(.contentMinSize)
    }
}

struct PillButton: View {
    let title: String
    var prominent = false
    var enabled = true
    let action: () -> Void
    var body: some View {
        Text(title)
            .padding(.horizontal, 12).padding(.vertical, 5)
            .foregroundStyle(prominent ? Color.white : Color.primary)
            .background(RoundedRectangle(cornerRadius: 6).fill(prominent ? Color.accentColor : Color.gray.opacity(0.18)))
            .opacity(enabled ? 1 : 0.4)
            .contentShape(Rectangle())
            .onTapGesture { if enabled { action() } }
    }
}

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
        .frame(minWidth: 720, minHeight: 560)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                ForEach(InputMode.allCases) { m in
                    PillButton(title: m.rawValue, prominent: model.mode == m, enabled: !model.running) { model.mode = m }
                }
                if model.mode == .wav {
                    PillButton(title: "選擇 wav…", enabled: !model.running) { model.chooseWavs() }
                    PillButton(title: model.wavPushToTalk ? "✓ 模擬按住說話" : "模擬按住說話", enabled: !model.running) {
                        model.wavPushToTalk.toggle()
                    }
                    Text(model.wavPaths.isEmpty ? "未選擇" : model.wavPaths.map { ($0 as NSString).lastPathComponent }.joined(separator: "、"))
                        .lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                }
                Spacer()
                if model.running {
                    PillButton(title: model.mode == .wav ? "停止" : "關閉麥克風") { model.stop() }
                } else {
                    PillButton(title: model.mode == .wav ? "開始" : "開啟麥克風", prominent: true) { model.start() }
                }
            }
            HStack {
                Text("模型").foregroundStyle(.secondary)
                Text(model.modelsDir.isEmpty ? "未設定" : model.modelsDir)
                    .lineLimit(1).truncationMode(.head).font(.system(.caption, design: .monospaced))
                PillButton(title: "變更…", enabled: !model.running) { model.chooseModels() }
            }
            if let e = model.errorText { Text(e).foregroundStyle(.red).font(.callout) }
        }
        .padding(12)
    }

    private var liveCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("即時").font(.headline)
                Text(model.statusText).font(.caption).foregroundStyle(.secondary)
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
            .font(.system(size: 26, weight: .medium))
            .frame(maxWidth: .infinity, minHeight: 70, alignment: .leading)
        }
        .padding(12)
    }

    /// 大按鈕：按住收音、放開立刻定稿；滑鼠 / 觸控按住（iPhone 同一個手勢）或空白鍵
    private var talkButton: some View {
        let on = model.holding
        let usable = model.mode == .pushToTalk && model.running
        return Text(on ? "放開結束" : (model.mode == .wav ? "按住說話（wav 模擬）" : (usable ? "按住說話　（空白鍵）" : "先開啟麥克風")))
            .font(.system(size: 22, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, minHeight: 78)
            .background(RoundedRectangle(cornerRadius: 14).fill(on ? Color.red : (usable || model.mode == .wav ? Color.accentColor : Color.gray)))
            .padding(.horizontal, 12).padding(.bottom, 12)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in model.pressStart() }
                    .onEnded { _ in model.pressEnd() })
    }

    private var history: some View {
        historyScroll {
          VStack(alignment: .leading, spacing: 10) {
            ForEach(model.utterances.filter { $0.status != .listening || !$0.text.isEmpty }.reversed()) { u in
                HStack(alignment: .firstTextBaseline) {
                    Text("#\(u.id + 1)").foregroundStyle(.tertiary).frame(width: 36, alignment: .leading)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(u.status == .provisional ? u.text + "（暫定）" : u.text)
                            .font(.title3)
                            .foregroundStyle(u.status == .final ? Color.primary : (u.status == .provisional ? .orange : .secondary))
                        HStack(spacing: 14) {
                            Text("首字 \(fmt(u.firstTokenLatency))")
                            Text("尾字 \(fmt(u.finalLatency))")
                            if let r = u.releaseToFinalLatency { Text("放開→定稿 \(fmt(r))") }
                            if let s = u.speechEnd {
                                Text(String(format: "語音 %.2f–%.2f s", u.speechStart, s))
                            }
                        }
                        .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
          }
          .padding(12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// ImageRenderer（文件截圖用）畫不出 ScrollView，截圖模式改用一般容器
    @ViewBuilder private func historyScroll<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        if CommandLine.arguments.contains("--snapshot-to") { content() } else { ScrollView { content() } }
    }

    private var footer: some View {
        HStack(spacing: 18) {
            stat("模型載入", model.loadSeconds.map { String(format: "%.2f s", $0) } ?? "—")
            stat("峰值記憶體", String(format: "%.0f MB", model.peakRSS))
            stat("目前記憶體", String(format: "%.0f MB", model.currentRSS))
            stat("RTF", model.rtf.map { String(format: "%.3f", $0) } ?? "—")
            Spacer()
            Text("核心 \(OzAsrRecognizer.version)").font(.caption).foregroundStyle(.tertiary)
        }
        .padding(10)
    }

    private func stat(_ k: String, _ v: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(k).font(.caption).foregroundStyle(.secondary)
            Text(v).font(.system(.body, design: .monospaced))
        }
    }

    private func fmt(_ v: Double?) -> String { v.map { String(format: "%.0f ms", $0 * 1000) } ?? "—" }
}

struct RootView: View {
    @StateObject private var model = AppModel()
    var body: some View { ContentView(model: model) }
}
