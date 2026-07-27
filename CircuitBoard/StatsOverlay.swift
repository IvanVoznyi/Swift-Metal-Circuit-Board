import SwiftUI

/// The frame-time readout. Off by default; one button shows it.
///
/// The number that matters is the worst frame in the last second, not the
/// average — an average of 4 ms hides a 20 ms frame every eight seconds
/// perfectly well, and that one frame is the whole complaint. `frame` and
/// `bake` split the GPU time so a long frame says immediately whether the cost
/// is in drawing the screen or in rasterizing new board.
struct StatsOverlay: View {
    @ObservedObject var stats: FrameStats

    var body: some View {
        let s = stats.snapshot
        let budget = 1000.0 / Double(max(s.refreshHz, 1))
        VStack(alignment: .leading, spacing: 2) {
            row("fps", String(format: "%.0f", s.fps), warn: s.fps < Double(s.refreshHz) * 0.9)
            row("worst", String(format: "%.2f ms", s.worstFrameMs), warn: s.worstFrameMs > budget * 1.5)
            row("budget", String(format: "%.2f ms", budget), warn: false)
            row("gpu frame", String(format: "%.2f ms", s.frameGpuMs), warn: s.frameGpuMs > budget)
            row("gpu bake", String(format: "%.2f ms", s.bakeGpuMs), warn: s.bakeGpuMs > budget)
            row("late", "\(s.lateFrames)", warn: s.lateFrames > 0)
            row("tiles", "\(s.tilesReady)", warn: false)
        }
        .font(.system(size: 11, design: .monospaced))
        .padding(8)
        .background(Color.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
        .allowsHitTesting(false)
    }

    private func row(_ label: String, _ value: String, warn: Bool) -> some View {
        HStack(spacing: 6) {
            Text(label).foregroundStyle(.white.opacity(0.55))
            Spacer(minLength: 8)
            Text(value).foregroundStyle(warn ? .red : .white)
        }
        .frame(width: 130)
    }
}
