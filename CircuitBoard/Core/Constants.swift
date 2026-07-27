import CoreGraphics
import Foundation
import simd

// Values that the GPU also needs are re-exported from PCBBridge.h
// rather than restated, so Swift and Metal cannot disagree.

// MARK: - Board metrics

enum Board {
    /// Routing grid pitch, px. (`GS`)
    static let gridSize = Float(PCB_GRID_SIZE)
    /// Rows of routing grid per tile. (`ROWS`)
    static let rows = Int(PCB_ROWS)
    /// One tile's height in board units. (`TILE_H`)
    static let tileHeight = Float(PCB_TILE_HEIGHT)
    /// Copper→copper clearance, px. (`CLEAR`)
    static let clearance = Float(PCB_CLEARANCE)
    /// Vertical grid origin. (`OFY`)
    static let originY = Float(PCB_ORIGIN_Y)
    /// Closest two legal centre-lines can ever get: two parallel 45° runs one
    /// cell apart are only GS/√2 apart, and that is the binding case.
    static let pitch = Float(PCB_GRID_SIZE / 2.0.squareRoot())
    /// Slider counts scale by `width / referenceWidth`. (`wf`)
    static let referenceWidth = Float(PCB_REFERENCE_WIDTH)
    /// How far a seam trace runs past its tile's edge to meet its twin on the
    /// other side (`TileData.polyline`). It is the only reason a tile just off
    /// the near edge of the view needs drawing at all.
    static let seamOverhang: Float = 5

    /// Narrowest board we will lay out, points.
    static let minimumWidth: Float = 320

    /// Board units laid out across, per point of window width, in the tilted
    /// view.
    ///
    /// The camera scales the board up until it spans `Camera.widthFill` times
    /// the visible width, so pushing that number up to cover the far end of the
    /// strip also magnifies every trace by the same factor. Laying out
    /// proportionally more circuitry cancels it exactly: `widthFill /
    /// tiltedWidthScale` is the apparent zoom, and at 8 / 2.35 that is the 3.4
    /// the board was drawn at before it had to reach the horizon.
    ///
    /// It is not free — routing is linear in width, so this is also the factor
    /// on the ~65 ms a tile costs to generate.
    static let tiltedWidthScale: Float = 2.35
}

// MARK: - Trace classes

enum TraceClass: CaseIterable {
    case main, bus, signal, fine

    var width: Float {
        switch self {
        case .main:   return Float(PCB_W_MAIN)     // power / ground rail
        case .bus:    return Float(PCB_W_BUS)
        case .signal: return Float(PCB_W_SIGNAL)
        case .fine:   return Float(PCB_W_FINE)
        }
    }

    /// Widest thin class, used to size pad keepout.
    static var widestThin: Float { Float(PCB_W_BUS) }
}

// MARK: - Routing

enum Routing {
    static let hugRadius = Int(PCB_HUG_RADIUS)
    static let turn45 = Float(PCB_TURN_45)
    static let turn90 = Float(PCB_TURN_90)
    static let budget = Int(PCB_ASTAR_BUDGET)
    /// `HUGMUL = 1 - 0.45 * hugSlider`, so the slider's floor is this.
    static let hugMultiplierFloor = Float(PCB_HUG_MIN_MUL)
    static let hugMultiplierSpan: Float = 0.45

    /// Clockwise, so |di - dj| is the turn magnitude (1 = 45°, 2 = 90°).
    static let directions: [SIMD2<Int32>] = [
        SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1), SIMD2(-1, 1),
        SIMD2(-1, 0), SIMD2(-1, -1), SIMD2(0, -1), SIMD2(1, -1),
    ]

    /// Octile heuristic's diagonal surcharge (√2 − 1).
    static let diagonalExtra: Float = 0.4142
    static let diagonalStep: Float = 1.4142

    /// Attempts before a placement or routing helper gives up.
    static let freeCellTries = 80
    static let padTraceTries = 10
    static let meanderTries = 10
    static let clusterTries = 8
    static let decorTries = 12
    static let poolProbeTries = 10
    static let combProbeTries = 8
}

// MARK: - Placement rates

enum Rates {
    static let chip = Float(PCB_CHIP_RATE)
    static let dipChip = Float(PCB_DIPCHIP_RATE)
    static let decor = Float(PCB_DECOR_RATE)
    static let meander = Float(PCB_MEANDER_RATE)
    /// A trace picks `fine` this often, `signal` otherwise.
    static let fineOverSignal: Float = 0.35
    /// Loose pads scattered per tile, before the width scale.
    static let loosePads: ClosedRange<Int> = 26...42
    static let decorCount: ClosedRange<Int> = 3...7
    static let busMembers: ClosedRange<Int> = 3...6
}

// MARK: - Seam ports

enum Seam {
    /// Hash salt so seam ports derive from a different stream than tiles.
    static let worldSalt: UInt32 = 0x5bf0_3635
    static let indexSalt: UInt32 = 104_729
    static let tileSalt: UInt32 = 7_919
    /// Ports must be at least this many columns apart, or the first trace's
    /// keepout blocks the second and that column loses its seam crossing.
    static let minColumnGap: Int = 4
    /// Cells reserved inward from each port until that port's turn comes.
    static let reservedDepth = 9
    /// `seamFallback` walks this far in looking for room to drop a via.
    static let fallbackDepth: ClosedRange<Int> = 3...7
    /// Class mix: u < mainCut → main, u < busCut → bus, else signal.
    static let mainCut: Float = 0.14
    static let busCut: Float = 0.45
}

// MARK: - Palette

/// One colour scheme for the whole board.
///
/// A value rather than a namespace of constants, because the scheme is now a
/// choice: it is part of `BoardParams`, so switching it invalidates every
/// cached tile — trace colours and pad metals are picked during *generation*,
/// out of the same seeded stream that decides everything else, and a board
/// recoloured after the fact would draw a different picture from the one the
/// router laid out.
struct Palette: Equatable {
    /// Trace colours. (`PAL`)
    let traces: [SIMD3<Float>]
    /// Pad / package metals. (`METALS`)
    let metals: [SIMD3<Float>]
    let substrate: SIMD3<Float>
    let padBacking: SIMD4<Float>
    let padHole: SIMD3<Float>
    let smdOutline: SIMD4<Float>
    let chipSolidBody: SIMD4<Float>
    let chipOutlineBody: SIMD4<Float>
    /// Silkscreen number ink.
    let numberInk: SIMD3<Float>

    /// The halo was `rgba(10,6,20,.6)` painted over the substrate. With the
    /// substrate a flat colour that composite is a constant, so it is stored
    /// pre-blended and fully opaque — which is what lets a trace be drawn as
    /// plain overlapping capsules: opaque over opaque is idempotent, so a
    /// corner cannot darken however many times it is covered. It follows the
    /// scheme's own substrate, or the halo would be the wrong dark.
    var traceHalo: SIMD4<Float> {
        let ink = Palette.rgb(10, 6, 20)
        let alpha: Float = 0.6
        return SIMD4(ink * alpha + substrate * (1 - alpha), 1)
    }

    /// The light runner's colour, and how much of it to use. `w` at 0 means the
    /// runner takes each trace's own colour, which is the default and what makes
    /// the light look like it belongs to that wire; raise it to tint every
    /// runner the same, whatever it is running along.
    var runner: SIMD4<Float> { SIMD4(1, 1, 1, 0) }

    /// Pad metal is drawn this far above 1. The scene target is HDR, so that is
    /// not clipping — it is what the bloom prefilter reads as a light source.
    /// Scheme-independent: it is an exposure, not a colour.
    static let padGlow = Float(PCB_PAD_GLOW)

    static func rgb(_ r: Int, _ g: Int, _ b: Int) -> SIMD3<Float> {
        SIMD3(Float(r) / 255, Float(g) / 255, Float(b) / 255)
    }

    /// The original spread: eight saturated hues on a dark blue substrate.
    static let neon = Palette(
        traces: [
            rgb(0xff, 0x5b, 0xb0), rgb(0xff, 0xd2, 0x4a),
            rgb(0x4a, 0xd6, 0xff), rgb(0x7b, 0xff, 0x8a),
            rgb(0xff, 0x8a, 0x3d), rgb(0xb0, 0x6b, 0xff),
            rgb(0xff, 0x4d, 0x4d), rgb(0x2a, 0xff, 0xc7),
        ],
        metals: [
            rgb(0xd9, 0xdb, 0xe4), rgb(0xe7, 0xc9, 0x77),
            rgb(0xc4, 0xc8, 0xd2), rgb(0xb8, 0xc9, 0xe0),
        ],
        substrate: rgb(0x0c, 0x10, 0x24),
        padBacking: SIMD4(8 / 255, 5 / 255, 16 / 255, 0.55),
        padHole: rgb(0x14, 0x0a, 0x22),
        smdOutline: SIMD4(20 / 255, 12 / 255, 30 / 255, 0.6),
        chipSolidBody: SIMD4(5 / 255, 3 / 255, 12 / 255, 0.92),
        chipOutlineBody: SIMD4(12 / 255, 16 / 255, 36 / 255, 0.72),
        numberInk: rgb(158, 200, 244))

    /// Amber and cyan on near-black navy, from the reference boards. Two
    /// families rather than a spread: warm copper against cold signal, which is
    /// what those photographs are actually made of. Deliberately fewer, more
    /// closely related hues than `neon` — the reference reads as one board lit
    /// from inside, not as eight colours of wire.
    static let amber = Palette(
        traces: [
            rgb(0xff, 0xa5, 0x2b), rgb(0xff, 0x7a, 0x18),
            rgb(0xff, 0xce, 0x73), rgb(0x2f, 0xc2, 0xff),
            rgb(0x00, 0x9b, 0xff), rgb(0x8f, 0xde, 0xff),
            rgb(0xff, 0x8c, 0x42), rgb(0x1b, 0x7f, 0xe0),
        ],
        metals: [
            rgb(0xe8, 0xc0, 0x74), rgb(0xd6, 0xdd, 0xe8),
            rgb(0xf2, 0xa9, 0x4c), rgb(0xa9, 0xc4, 0xde),
        ],
        substrate: rgb(0x05, 0x0b, 0x18),
        padBacking: SIMD4(5 / 255, 8 / 255, 16 / 255, 0.55),
        padHole: rgb(0x0a, 0x10, 0x1e),
        smdOutline: SIMD4(18 / 255, 26 / 255, 40 / 255, 0.6),
        chipSolidBody: SIMD4(3 / 255, 6 / 255, 14 / 255, 0.92),
        chipOutlineBody: SIMD4(10 / 255, 18 / 255, 34 / 255, 0.72),
        numberInk: rgb(0xf0, 0xc8, 0x8a))
}

/// The schemes on offer, in the order the control cycles them.
enum PaletteScheme: Int, CaseIterable, Equatable {
    case neon, amber

    var colours: Palette {
        switch self {
        case .neon:  return .neon
        case .amber: return .amber
        }
    }

    var label: String {
        switch self {
        case .neon:  return "neon"
        case .amber: return "amber"
        }
    }

    var next: PaletteScheme {
        PaletteScheme(rawValue: rawValue + 1) ?? .neon
    }
}

// MARK: - Drawing

enum Draw {
    /// Halo stroke is `trace width + haloExtra` wide. (`t.w + 3.4`)
    static let haloExtra = Float(PCB_HALO_EXTRA)
    static var haloHalfExtra: Float { haloExtra / 2 }

    /// How far a layer's substrate slab runs past its tile at each end, so
    /// consecutive slabs overlap instead of sharing an antialiased edge.
    ///
    /// A quarter of a tile, not a board unit or two: the far end of the strip
    /// compresses thousands of board units into a few pixels, so a small margin
    /// there is still less than a pixel and the seam survives. This is measured
    /// against the tile precisely so it stays wide enough wherever the tile
    /// lands.
    static let substrateOverlap = Board.tileHeight * 0.25

    static let teardropExtra = Float(PCB_TEARDROP_EXTRA)
    static let teardropRatio = Float(PCB_TEARDROP_RATIO)

    // Pad geometry, straight from `drawPad`.
    enum Pad {
        static let smdBackingGrow: Float = 4
        static let smdBackingRadius: Float = 4
        static let smdRadius: Float = 3
        static let smdOutlineWidth: Float = 1
        static let squareBackingGrow: Float = 5
        static let squareBackingRadius: Float = 2
        static let squareRadius: Float = 1.5
        static let squareHoleRatio: Float = 0.5
        static let ringBackingGrow: Float = 3
        static let ringHoleRatio: Float = 0.62
        static let ringInnerRatio: Float = 0.78
        static let ringInnerWidth: Float = 1.6
        static let roundBackingGrow: Float = 2.5
        static let viaHoleRatio: Float = 0.4
        static let throughHoleRatio: Float = 0.5
    }

    // Package body, straight from `drawChip`.
    enum Chip {
        static let bodyRadius: Float = 7
        static let bodyStroke: Float = 2
        static let innerRadius: Float = 4
        static let solidRingAlpha: Float = 0.6
        static let solidRingWidth: Float = 1.2
        static let dashRingAlpha: Float = 0.42
        static let dashRingWidth: Float = 1
        static let dashOn: Float = 5
        static let dashOff: Float = 4
        static let insetRange: ClosedRange<Float> = 0.05...0.15
    }

    // Silkscreen components, straight from `drawDecor`.
    enum Decor {
        static let alpha: Float = 0.6
        static let scaleRange: ClosedRange<Float> = 1...1.15
        static let soicLegWidth: Float = 1.4
        static let bodyRadius: Float = 1.5
        static let dotRadius: Float = 1.3
        static let frameOuterWidth: Float = 1.6
        static let frameInnerWidth: Float = 1.1
        static let frameInset: Float = 3
        static let dipOutlineWidth: Float = 1.2
        static let dipPinRadius: Float = 1.1
        static let dipPinOffset: Float = 2
        static let dipPadding: Float = 3
        static let dipHalfHeight: Float = 5
    }

    /// Silkscreen numbers keep this much px margin so a trace's halo never
    /// grazes a digit.
    static let numberMargin: Float = 3.5
    static let numberAlpha: ClosedRange<Float> = 0.10...0.26
    static let numberSize: ClosedRange<Int> = 9...13
    static let numberVerticalChance: Float = 0.15
    static let numberTries = 40

    /// Reference designators reserve keepout on this pixel lattice, so copper
    /// routes around the letter shapes.
    static let glyphLattice = 4
    static let glyphAlphaCutoff: UInt8 = 128
    static let labelPrefixes = ["U", "R", "C", "J", "Q", "D", "L", "IC", "T", "SW", "Y"]
    static let labelSize: ClosedRange<Int> = 15...21
    static let topLabel = "T O P"
    static let topLabelSize: Float = 30
}

// MARK: - Animation & residency

enum Anim {
    /// Board units per second the content drifts downward, which is the same
    /// as saying new tiles arrive at the top.
    static let scrollSpeed = Float(PCB_SCROLL_SPEED)

    // FrameClock tuning. The clock advances itself and only leans toward the
    // wall clock, so the timestamps it produces are near-perfectly evenly
    // spaced — which is what the eye reads as smooth.
    /// How fast the measured frame interval follows reality.
    static let intervalTracking: Double = 0.05
    /// How hard each frame is pulled toward the wall clock.
    static let phaseTracking: Double = 0.03
    /// Drift beyond this many frames means a stall; resynchronise hard.
    static let resyncFrames: Double = 4
    static let minFrameInterval: Double = 1.0 / 240
    static let maxFrameInterval: Double = 1.0 / 20

    /// Seconds the board takes to fade up once the whole strip is resident.
    /// The board is never seen half-built: nothing is drawn until every visible
    /// tile on every layer is ready, and then all of it arrives together.
    static let introFade: Double = 0.9

    /// How opaque the board is, given when it was first complete. Nil means it
    /// still is not — draw nothing rather than a board with holes in the
    /// distance.
    static func introAlpha(revealedAt: Double?, now: Double) -> Float {
        guard let revealedAt else { return 0 }
        let t = min(max((now - revealedAt) / introFade, 0), 1)
        return Float(t * t * (3 - 2 * t))     // smoothstep, so it eases in and out
    }
}

/// The light that runs the length of a trace.
///
/// Every trace has its **own** clock: its own moment to set off, its own time to
/// cross, its own wait before going again. One shared clock made the whole board
/// blink in unison, which reads as a light show; independent ones read as
/// traffic on a circuit.
///
/// Arc-length parameterised, so a head crosses its trace in `travel` seconds
/// whether that trace is long or short. Constant board-units-per-second instead
/// would make every short trace a fast blink and every long one a crawl.
enum Pulse {
    /// Seconds a head takes to cross its trace.
    static let travel: ClosedRange<Float> =
        Float(PCB_PULSE_TRAVEL_MIN)...Float(PCB_PULSE_TRAVEL_MAX)
    /// Seconds the trace stays dark before the next head sets off.
    static let delay: ClosedRange<Float> =
        Float(PCB_PULSE_DELAY_MIN)...Float(PCB_PULSE_DELAY_MAX)
    /// How far above the trace's own colour the head burns. It is a multiple of
    /// that colour rather than white, so a green trace pulses green.
    static let gain = Float(PCB_PULSE_GAIN)
    static let width = Float(PCB_PULSE_WIDTH)
    static let tail = Float(PCB_PULSE_TAIL)

    /// One trace's clock, in the form the shader consumes.
    struct Clock {
        var phase: Float      // seconds; where in its cycle it stands at t = 0
        var period: Float     // seconds from one setting-off to the next
        var speed: Float      // fraction of the trace crossed per second
    }

    /// Draws a clock from `rng`. The period is not free: it has to be long
    /// enough for the head to cross *and* for its tail to clear the far end,
    /// or the next one sets off while the last is still lit.
    static func clock(from rng: Rng) -> Clock {
        let crossing = rng.float(travel)
        let clearing = crossing * width * tail
        let period = crossing + clearing + rng.float(delay)
        return Clock(phase: rng.float(0, period),
                     period: period,
                     speed: 1 / crossing)
    }

    /// Where a head is along its trace at `time`. Past 1 it has run off the end
    /// and the trace is dark — that is the wait, expressed as position rather
    /// than as a second timer.
    static func head(_ clock: Clock, at time: Double) -> Float {
        let period = max(clock.period, 1e-3)
        var cycle = (Float(time) + clock.phase).truncatingRemainder(dividingBy: period)
        if cycle < 0 { cycle += period }
        return cycle * clock.speed
    }
}

// MARK: - Parallax

/// Inner layers of the board, stacked below the live one.
///
/// A real board has layers, and under a tilted camera the honest way to show
/// that is to put them where they belong: parallel planes a little further
/// down, seen through the same perspective. They reuse the *same* tile
/// instance buffers — no extra generation, no extra memory — and a sideways
/// shift plus a depth offset stops them reading as a duplicate of the top
/// layer. The perspective supplies the parallax for free; they all scroll at
/// the same rate, because a real inner layer does not slide relative to the
/// board it is part of.
///
/// Deeper layers draw copper only. Pads and silkscreen at 20% fade would be
/// visual noise, and skipping them is most of the cost back.
enum Parallax {
    struct Layer {
        /// World height relative to the live board. Negative sinks it.
        let height: Float
        /// 1 = the live board; below that the colour mixes toward the substrate.
        let fade: Float
        /// Sideways shift, board units, so layers do not line up.
        let offsetX: Float
        /// How far along the strip this layer is shifted, so its tiles do not
        /// share seams with the layer above.
        let offsetZ: Float
        /// Shifts this layer into a different stretch of the infinite board.
        ///
        /// A tile's content comes from `hash(world, index)`, so a layer drawing
        /// indices a few hundred thousand away is drawing genuinely different
        /// circuitry — different routing, different packages — rather than the
        /// same board nudged sideways. Consecutive indices within the offset
        /// still meet at their seams, so each layer stays continuous.
        let indexOffset: Int
        /// The live layer draws pads and silkscreen; deeper ones do not.
        let drawsDetail: Bool
        /// Whether the signal pulse runs along this layer's copper.
        let pulses: Bool
        /// How much of the board's detail this layer's tiles are generated with.
        ///
        /// The inner layers draw copper and nothing else — no silkscreen, no
        /// package bodies, pads only on request — yet they were being generated
        /// with all of it: reference designators routed around, footprints
        /// placed, a full glyph atlas rasterised per tile. At 11% and 24% fade
        /// behind a blur none of that is visible, and it is most of the wait
        /// before the board first appears.
        let detail: Float
        /// How dark this layer's own substrate slab is drawn, against the live
        /// board's. Deeper is darker: the slabs are otherwise all the same
        /// colour, so a stack of them reads as one flat surface with wires at
        /// different heights rather than as boards behind boards. Making each
        /// step down darker is what separates them, and what leaves the live
        /// board as the only thing at full strength.
        let substrateDim: Float
    }

    /// Back to front. The last entry is the live board and must be the
    /// unmodified one.
    static let layers: [Layer] = [
        Layer(height: -46, fade: 0.11, offsetX: 137, offsetZ: 191,
              indexOffset: 611_953, drawsDetail: false, pulses: true,
              detail: 0.5, substrateDim: 0.34),
        Layer(height: -22, fade: 0.24, offsetX: -83, offsetZ: -97,
              indexOffset: 274_177, drawsDetail: false, pulses: true,
              detail: 0.5, substrateDim: 0.62),
        Layer(height: 0, fade: 1.00, offsetX: 0, offsetZ: 0,
              indexOffset: 0, drawsDetail: true, pulses: true,
              detail: 1.0, substrateDim: 1.00),
    ]

    static var foreground: Layer { layers[layers.count - 1] }

    /// Which layer a tile index belongs to.
    ///
    /// By nearest offset rather than by an exact range, so it stays right as the
    /// board drifts: indices march away from their layer's offset forever, and
    /// any fixed window would eventually be wrong. Nothing collides until the
    /// drift covers the gap between two offsets, which at one tile per 8.3
    /// seconds is weeks of running.
    static func layer(owning index: Int) -> Layer {
        var best = layers[0]
        var bestDistance = Int.max
        for candidate in layers {
            let d = abs(index - candidate.indexOffset)
            if d < bestDistance { bestDistance = d; best = candidate }
        }
        return best
    }
}

enum Layout {
    /// Tiles resident in the ring.
    ///
    /// This has to hold everything the renderer *asks* for, not merely what is
    /// on screen — and those differ by a factor that is easy to lose track of.
    /// Three parallax layers each want their own run of the strip, up to ten
    /// tiles at the depth clamp's ceiling, plus the prefetch lead: forty-two.
    ///
    /// Sized to the visible set alone (thirty) it appeared to work and did not.
    /// The prefetch had two spare slots to rotate through, so a lead tile was
    /// almost always evicted before it was wanted; then every time the strip
    /// advanced by one, the newly visible tile had to be generated from
    /// scratch while the frame went out without it. A whole tile of board
    /// missing for as long as routing takes — and once there is a wide blur
    /// over the far end, that reads as the picture blinking.
    ///
    /// A slot is about 1.75 MB, most of it the glyph atlas.
    static let maxCachedTiles = 64
    /// Tiles kept ready beyond the viewport. Heavily biased upward: that is
    /// where fresh board arrives from, and a tile that is ready early costs
    /// nothing but a slice.
    static let prefetchAbove = 3
    static let prefetchBelow = 1
    /// Tiles whose capsules are expanded per frame. Expansion is one small
    /// compute dispatch per tile — a few hundred threads — so this only paces
    /// arrival, it is not a frame cost worth spreading.
    static let expansionsPerFrame = 1

    /// Concurrent generator tasks. Three, at `.utility`: routing a tile is
    /// 100–200 ms of solid CPU at the tilted board's width, and three layers
    /// each want their own, so there is three times as much of it to get
    /// through. In the steady state a tile is still only needed every ~2.8
    /// seconds; this is really about filling the strip at launch. Nothing here
    /// races the render thread.
    static let generatorConcurrency = 3

    /// Workers while the board is still filling in for the first time, on a
    /// `.userInitiated` queue. Nothing is on screen yet, so there is no render
    /// thread to leave room for — the machine may as well be used. Two cores
    /// are left over so the app stays responsive and the OS has somewhere to
    /// put everything else.
    static let burstConcurrency = max(3, ProcessInfo.processInfo.activeProcessorCount - 2)

    /// Tiles expanded per frame while the board is still filling in. Expansion
    /// is a tiny dispatch, and pacing it one-per-frame only makes sense once
    /// there is a frame worth protecting; at launch it just adds a frame of
    /// latency per tile to a wait the user is already sitting through.
    static let burstExpansionsPerFrame = 8

    // ── Per-slot capacities ─────────────────────────────────────────────────
    // Measured worst case over ±12 tiles at every slider maximum and board
    // widths up to 3200 points (a 6K display): 741 path points, 97 traces,
    // 1295 solids, 160 numbers. These carry roughly 2.5× headroom and cost
    // ~1.6 MB per slot, ~25 MB for the whole ring — allocated once, never
    // grown. A tile that would exceed one is clamped rather than allowed to
    // allocate on a worker thread.
    static let maxPathPoints = 2048
    static let maxTraces = 256
    static let maxSolids = 4096
    static let maxGlyphs = 256
    static let glyphAtlasWidth = 1024
    static let glyphAtlasHeight = 1024
}

// MARK: - Controls

enum Sliders {
    struct Spec {
        let id: String
        let label: String
        let range: ClosedRange<Double>
        let initial: Double
    }

    static let traces = Spec(id: "tr", label: "traces", range: 2...70, initial: 34)
    static let groups = Spec(id: "bs", label: "groups", range: 0...12, initial: 6)
    static let hug = Spec(id: "hugS", label: "hug", range: 0...100, initial: 72)
    static let rails = Spec(id: "rails", label: "rails", range: 0...6, initial: 2)
    static let parts = Spec(id: "fp", label: "parts", range: 0...8, initial: 4)
    static let nums = Spec(id: "numS", label: "nums", range: 0...40, initial: 7)

    static let all = [traces, groups, hug, rails, parts, nums]
}
