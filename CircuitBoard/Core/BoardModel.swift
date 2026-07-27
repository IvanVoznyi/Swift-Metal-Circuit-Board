import Combine
import Foundation

#if os(macOS)
import AppKit
#endif

/// The seven sliders and the toggles of the HTML control bar, as observable
/// state. Every one of them invalidates the tile cache, so they live in one
/// place and reach the renderer as a single `BoardParams` snapshot rather than
/// as seven separate bindings.
///
/// Kept out of the view file so the mapping from control to generator input is
/// testable on its own.
final class BoardModel: ObservableObject {
    @Published var traces = Float(Sliders.traces.initial)
    @Published var groups = Float(Sliders.groups.initial)
    @Published var hug = Float(Sliders.hug.initial)
    @Published var rails = Float(Sliders.rails.initial)
    @Published var parts = Float(Sliders.parts.initial)
    @Published var nums = Float(Sliders.nums.initial)
    @Published var teardrops = false
    /// Colour scheme. Part of the board's identity, not of how it is drawn, so
    /// changing it throws every cached tile away — trace colours are picked
    /// during generation.
    @Published var scheme: PaletteScheme = .neon
    @Published var world = UInt32.random(in: 0..<1_000_000_000)
    @Published var backgroundPads = true
    @Published var tilted = true
    @Published var glow = true
    @Published var chromeHidden = false
    @Published var showStats = false

    var params: BoardParams {
        var p = BoardParams()
        p.traces = traces
        p.groups = groups
        p.hug = hug
        p.rails = rails
        p.parts = parts
        p.nums = nums
        p.teardrops = teardrops
        p.scheme = scheme
        p.world = world
        return p
    }

    var renderOptions: RenderOptions {
        var o = RenderOptions(backgroundPads: backgroundPads, tilted: tilted)
        o.glow = glow
        if !glow { o.post.bloomIntensity = 0 }
        return o
    }

    func newBoard() { world = UInt32.random(in: 0..<1_000_000_000) }

    /// Main actor because it reaches for the key window; SwiftUI calls it from
    /// a button, which is already there.
    @MainActor
    func toggleFullScreen() {
        #if os(macOS)
        NSApp.keyWindow?.toggleFullScreen(nil)
        #else
        // iOS has no window chrome to shed, so the equivalent gesture is
        // getting the controls out of the way.
        chromeHidden.toggle()
        #endif
    }
}
