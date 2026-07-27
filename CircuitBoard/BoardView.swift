import MetalKit
import SwiftUI

#if os(macOS)
import AppKit
typealias PlatformViewRepresentable = NSViewRepresentable
#else
import UIKit
typealias PlatformViewRepresentable = UIViewRepresentable
#endif

/// The board surface: an `MTKView` and nothing else.
///
/// There is deliberately no gesture handling. The board is a continuous
/// downward drift, and every input path was another way for the position to
/// jump — a wheel event mid-frame, a pan recogniser competing with the display
/// link, a key repeat. Removing them makes the motion one uninterrupted flow,
/// which is the whole point of the piece.
struct BoardView: PlatformViewRepresentable {
    var params: BoardParams
    var options: RenderOptions
    var stats: FrameStats

    final class Coordinator {
        var renderer: Renderer?
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    static var displayRefreshRate: Int {
        #if os(macOS)
        return max(60, NSScreen.main?.maximumFramesPerSecond ?? 60)
        #else
        return max(60, UIScreen.main.maximumFramesPerSecond)
        #endif
    }

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        // Set before the renderer builds its pipelines against it.
        view.colorPixelFormat = .bgra8Unorm
        // Run at the display's own rate. Every frame is a handful of textured
        // quads, so this is sustainable at 120 Hz as easily as at 60.
        view.preferredFramesPerSecond = BoardView.displayRefreshRate
        view.framebufferOnly = true
        // No MSAA and no depth: every edge on the board is antialiased
        // analytically in the fragment shader, which is sharper and cheaper.
        view.sampleCount = 1
        view.depthStencilPixelFormat = .invalid
        let renderer = Renderer(view: view, params: params, stats: stats)
        renderer?.apply(options)
        coordinator.renderer = renderer
        view.delegate = renderer
        return view
    }

    private func update(_ coordinator: Coordinator) {
        coordinator.renderer?.apply(params)
        coordinator.renderer?.apply(options)
    }

    #if os(macOS)
    func makeNSView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateNSView(_ view: MTKView, context: Context) { update(context.coordinator) }
    #else
    func makeUIView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateUIView(_ view: MTKView, context: Context) { update(context.coordinator) }
    #endif
}
