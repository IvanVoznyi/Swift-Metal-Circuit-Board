import Combine
import Foundation
import QuartzCore

/// Frame timing, collected on the render thread and republished a few times a
/// second for the overlay.
///
/// This exists because the interesting number is not the average — it is the
/// *worst* frame in the last second, and where that time went. An average of
/// 4 ms hides a 20 ms frame every eight seconds perfectly well, and that one
/// frame is the whole complaint.
///
/// Safe to hand across threads: every mutable field below is guarded by
/// `lock`, and `snapshot` — the only thing SwiftUI reads — is assigned solely
/// on the main actor. The compiler cannot see that invariant, so it is stated
/// here rather than assumed.
final class FrameStats: ObservableObject, @unchecked Sendable {

    struct Snapshot {
        var fps: Double = 0
        /// Worst wall-clock gap between frames in the last second, ms.
        var worstFrameMs: Double = 0
        /// GPU time for the frame's own command buffer, ms (rolling max).
        var frameGpuMs: Double = 0
        /// GPU time for tile bake chunks, ms (rolling max).
        var bakeGpuMs: Double = 0
        /// Frames since launch that took longer than the refresh interval.
        var lateFrames = 0
        var refreshHz = 60
        var tilesReady = 0
    }

    @Published private(set) var snapshot = Snapshot()

    private var lock = NSLock()
    private var lastFrameTime: CFTimeInterval = 0
    private var windowStart: CFTimeInterval = 0
    private var framesInWindow = 0
    private var worstGapInWindow: Double = 0
    private var frameGpuInWindow: Double = 0
    private var bakeGpuInWindow: Double = 0
    private var lateFrames = 0
    private var refreshInterval: Double = 1.0 / 60
    private var pendingRefreshHz = 60
    private var tilesReady = 0

    func setRefreshRate(_ hz: Int) {
        lock.lock()
        refreshInterval = 1.0 / Double(max(hz, 1))
        pendingRefreshHz = hz
        lock.unlock()
    }

    /// Called once per frame on the render thread. Cheap: a few compares.
    func beginFrame(at now: CFTimeInterval, readyTiles: Int) {
        lock.lock()
        tilesReady = readyTiles
        if lastFrameTime > 0 {
            let gap = now - lastFrameTime
            worstGapInWindow = max(worstGapInWindow, gap)
            // A tenth of a frame of slack, so ordinary vsync wobble is not
            // counted as a dropped frame.
            if gap > refreshInterval * 1.5 { lateFrames += 1 }
        }
        lastFrameTime = now
        framesInWindow += 1
        if windowStart == 0 { windowStart = now }
        let elapsed = now - windowStart
        guard elapsed >= 0.25 else { lock.unlock(); return }

        let next = Snapshot(fps: Double(framesInWindow) / elapsed,
                            worstFrameMs: worstGapInWindow * 1000,
                            frameGpuMs: frameGpuInWindow * 1000,
                            bakeGpuMs: bakeGpuInWindow * 1000,
                            lateFrames: lateFrames,
                            refreshHz: pendingRefreshHz,
                            tilesReady: tilesReady)
        windowStart = now
        framesInWindow = 0
        worstGapInWindow = 0
        frameGpuInWindow = 0
        bakeGpuInWindow = 0
        lock.unlock()

        // Published on the main actor, which is where SwiftUI reads it.
        Task { @MainActor [weak self] in self?.snapshot = next }
    }

    /// From a completed command buffer's `gpuEndTime - gpuStartTime`. Called on
    /// a Metal completion thread.
    func recordFrameGpu(_ seconds: Double) {
        lock.lock(); frameGpuInWindow = max(frameGpuInWindow, seconds); lock.unlock()
    }

    func recordBakeGpu(_ seconds: Double) {
        lock.lock(); bakeGpuInWindow = max(bakeGpuInWindow, seconds); lock.unlock()
    }

    func reset() {
        lock.lock()
        lateFrames = 0
        worstGapInWindow = 0
        frameGpuInWindow = 0
        bakeGpuInWindow = 0
        lock.unlock()
    }
}
