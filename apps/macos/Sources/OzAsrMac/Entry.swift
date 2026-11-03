import Foundation

// 有 --headless 就走無介面驗證模式（不開視窗、不碰麥克風）；否則啟動 SwiftUI app。
@main
enum Entry {
    static func main() {
        if CommandLine.arguments.contains("--headless") {
            exit(runHeadless(Array(CommandLine.arguments.dropFirst())))
        }
        OzAsrMacApp.main()
    }
}
