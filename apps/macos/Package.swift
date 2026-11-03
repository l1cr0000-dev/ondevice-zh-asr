// swift-tools-version:5.9
import PackageDescription

// macOS app：SwiftUI 介面 + 無介面（--headless）驗證模式。libozasr 的連結旗標由 scripts/build_mac_app.sh 以
// -Xlinker 傳入（-L cpp/build -lozasr、rpath），bundle 組裝（嵌入 dylib、Info.plist、簽名）也在該腳本。
let package = Package(
    name: "OzAsrMac",
    platforms: [.macOS(.v13)],
    dependencies: [.package(path: "../shared/OzAsrKit")],
    targets: [
        .executableTarget(
            name: "OzAsrMac", dependencies: [.product(name: "OzAsrKit", package: "OzAsrKit")],
            path: "Sources/OzAsrMac")
    ]
)
