import SwiftUI

@main
struct OzAsrIOSApp: App {
    var body: some Scene {
        WindowGroup { RootView() }
    }
}

struct RootView: View {
    @StateObject private var model = AppModel()
    var body: some View { ContentView(model: model) }
}
