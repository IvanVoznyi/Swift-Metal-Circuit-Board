import SwiftUI

struct ContentView: View {
    @StateObject private var model = BoardModel()
    @StateObject private var stats = FrameStats()

    var body: some View {
        ZStack(alignment: .bottom) {
            BoardView(params: model.params, options: model.renderOptions, stats: stats)
                .ignoresSafeArea()

            if model.chromeHidden {
                // Leave a way back — hiding the bar must not hide its own switch.
                Button("⛶") { model.chromeHidden = false }
                    .buttonStyle(PCBButton())
                    .padding(.bottom, 20)
            } else {
                controls
                    .padding(.bottom, 16)
            }
        }
        .overlay(alignment: .topTrailing) {
            if model.showStats {
                StatsOverlay(stats: stats).padding(12)
            }
        }
        .background(Color(red: 0.047, green: 0.063, blue: 0.141))
        #if os(macOS)
        .frame(minWidth: 640, minHeight: 520)
        #endif
    }

    private var controls: some View {
        VStack(spacing: 10) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 10)],
                      spacing: 8) {
                Button(fullScreenLabel) { model.toggleFullScreen() }
                    .buttonStyle(PCBButton())
                Button("⟳ New board") { model.newBoard() }
                    .buttonStyle(PCBButton())
                Button(model.teardrops ? "◗ Cone: on" : "◗ Cone: off") {
                    model.teardrops.toggle()
                }
                .buttonStyle(PCBButton())
                Button(model.glow ? "✷ Glow: on" : "✷ Glow: off") {
                    model.glow.toggle()
                }
                .buttonStyle(PCBButton())
                Button("◐ \(model.scheme.label)") {
                    model.scheme = model.scheme.next
                }
                .buttonStyle(PCBButton())
                Button(model.tilted ? "◈ 3D: on" : "◈ 3D: off") {
                    model.tilted.toggle()
                }
                .buttonStyle(PCBButton())
                Button(model.backgroundPads ? "◎ Deep pads: on" : "◎ Deep pads: off") {
                    model.backgroundPads.toggle()
                }
                .buttonStyle(PCBButton())
                Button(model.showStats ? "◱ Hide stats" : "◱ Stats") {
                    model.showStats.toggle()
                    stats.reset()
                }
                .buttonStyle(PCBButton())
            }
            .frame(maxWidth: 620)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 132), spacing: 14)],
                      spacing: 8) {
                slider(Sliders.traces, $model.traces)
                slider(Sliders.groups, $model.groups)
                slider(Sliders.hug, $model.hug)
                slider(Sliders.rails, $model.rails)
                slider(Sliders.parts, $model.parts)
                slider(Sliders.nums, $model.nums)
            }
            .frame(maxWidth: 620)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        // A flat fill, NOT `.ultraThinMaterial`. A material samples and blurs
        // whatever is behind it, and behind it is a Metal view that changes
        // every frame — so the blur was being recomputed at full size sixty
        // times a second, on the same GPU trying to draw the board.
        .background(Color(red: 0.07, green: 0.08, blue: 0.15).opacity(0.88),
                    in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 12)
    }

    /// On iOS there is no window to expand, so the button hides the chrome.
    private var fullScreenLabel: String {
        #if os(macOS)
        return "⛶ Fullscreen"
        #else
        return "⛶ Hide UI"
        #endif
    }

    private func slider(_ spec: Sliders.Spec, _ value: Binding<Float>) -> some View {
        HStack(spacing: 6) {
            Text(spec.label)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.white.opacity(0.75))
            Slider(value: value,
                   in: Float(spec.range.lowerBound)...Float(spec.range.upperBound),
                   step: 1)
                .tint(Color(red: 0.357, green: 0.247, blue: 0.839))
        }
    }
}

private struct PCBButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold, design: .monospaced))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .foregroundStyle(.white)
            .padding(.horizontal, 15)
            .padding(.vertical, 9)
            .background(Color(red: 0.357, green: 0.247, blue: 0.839)
                .opacity(configuration.isPressed ? 0.7 : 1))
            .clipShape(RoundedRectangle(cornerRadius: 7))
    }
}

#Preview {
    ContentView()
}
