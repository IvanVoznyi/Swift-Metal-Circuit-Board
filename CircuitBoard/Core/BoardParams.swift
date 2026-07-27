import Foundation
import simd

/// Board width decides the routing grid, so it is part of a tile's identity:
/// change it and every cached tile is invalid, exactly as the HTML's `resize`
/// clears both caches.
struct BoardGeometry: Equatable {
    let width: Float          // board units (points)
    let cols: Int
    let originX: Float        // OFX

    init(width: Float) {
        let w = max(Board.minimumWidth, width.rounded(.down))
        self.width = w
        cols = Int(w / Board.gridSize)
        originX = (w - Float(cols) * Board.gridSize) / 2 + Board.gridSize / 2
    }

    /// Grid column → board x. (`px`)
    @inline(__always) func px(_ cx: Int) -> Float { originX + Float(cx) * Board.gridSize }
    /// Grid row → board y. (`py`)
    @inline(__always) func py(_ cy: Int) -> Float { Board.originY + Float(cy) * Board.gridSize }

    /// Every slider count scales with width so a wide board is not sparser.
    var widthFactor: Float { width / Board.referenceWidth }
}

/// The seven sliders plus the two toggles, snapshotted so a background
/// generator never reads UI state that is changing underneath it.
struct BoardParams: Equatable {
    var traces = Float(Sliders.traces.initial)
    var groups = Float(Sliders.groups.initial)
    var hug = Float(Sliders.hug.initial)
    var rails = Float(Sliders.rails.initial)
    var parts = Float(Sliders.parts.initial)
    var nums = Float(Sliders.nums.initial)
    /// The cone (teardrop) fillet where copper lands on a pad. Off gives a
    /// plain straight stroke into the pad centre.
    var teardrops = true
    /// Trace colours and pad metals are drawn from this during generation, out
    /// of the same seeded stream as everything else — so it belongs to the
    /// board's identity, not to how the board is drawn.
    var scheme: PaletteScheme = .neon
    var world: UInt32 = 0
    var geometry = BoardGeometry(width: Board.referenceWidth)

    // Counts, scaled by width exactly as `genTile` does.
    private func scaled(_ value: Float, floor: Int) -> Int {
        max(floor, Int((value * geometry.widthFactor).rounded()))
    }

    var traceCount: Int { scaled(traces, floor: 1) }
    var busCount: Int { scaled(groups, floor: 0) }
    var railCount: Int { scaled(rails, floor: 0) }
    var partCount: Int { scaled(parts, floor: 0) }
    var numberCount: Int { scaled(nums, floor: 0) }

    var palette: Palette { scheme.colours }

    /// The same board, generated to whatever detail a layer actually draws.
    ///
    /// The inner layers draw copper only, so the silkscreen and the footprints
    /// are pure cost: the numbers alone mean rasterising a glyph atlas per tile
    /// and routing around every letter. Dropping them and thinning the copper is
    /// invisible at 11% fade behind a blur, and it is a large part of the wait
    /// before the board first appears.
    func detailed(for layer: Parallax.Layer) -> BoardParams {
        guard layer.detail < 1 else { return self }
        var thin = self
        thin.parts = 0
        thin.nums = 0
        thin.traces *= layer.detail
        thin.groups *= layer.detail
        thin.rails *= layer.detail
        return thin
    }

    /// `HUGMUL = 1 - 0.45 * (hug / 100)`.
    var hugMultiplier: Float {
        1 - Routing.hugMultiplierSpan * (hug / 100)
    }
}

/// Settings that change only how the board is drawn.
///
/// Deliberately separate from `BoardParams`: that is the *generation* identity,
/// and changing it throws away every cached tile. These cost nothing but a
/// different draw call, so flipping one must not stall the board while a
/// minute of routing is redone.
struct RenderOptions: Equatable {
    /// Show pads on the parallax layers behind the live board. Off by default:
    /// the depth reads as bare copper, which is quieter.
    var backgroundPads = false
    /// The tilted, receding view. Off gives the flat overhead board.
    var tilted = true
    /// The bloom, and the pad brightness that feeds it. Off gives flat copper
    /// on flat substrate — the board as a drawing rather than as a lit object.
    var glow = true

    /// Exposure, saturation and bloom mix for the composite.
    var post = PostOptions()
}
