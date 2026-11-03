// swift-tools-version:5.9
import PackageDescription

// 平台無關的 Swift 包裝層：包 C API（cpp/include/ozasr/ozasr_c_api.h），給 macOS / iOS app 共用。
// 連結 libozasr 由使用端負責（macOS：動態庫放進 bundle；iOS：靜態庫或 xcframework）。
let package = Package(
    name: "OzAsrKit",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [.library(name: "OzAsrKit", targets: ["OzAsrKit"])],
    targets: [
        // 只有標頭（include/ozasr 是指向 cpp/include/ozasr 的相對 symlink），讓 Swift 能 import COzAsr。
        .target(name: "COzAsr", path: "Sources/COzAsr"),
        .target(name: "OzAsrKit", dependencies: ["COzAsr"], path: "Sources/OzAsrKit"),
    ]
)
