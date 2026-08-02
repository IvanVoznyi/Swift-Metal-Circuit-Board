import Foundation
import os
import MetalKit
import QuartzCore
import simd

/// Drives the viewport: advances the drift, keeps the right tiles resident, and
/// draws them.
///
/// The board drifts downward forever — position decreases without bound, so
/// freshly generated tiles arrive at the top. Tile indices go negative
/// indefinitely and the seam hash is defined over signed indices, so copper
/// still crosses every boundary.
///
/// Tiles are drawn straight to the drawable as vector instances. There is no
/// intermediate texture, which means no bake to schedule, no slice residency to
/// juggle, and no resample between the board and the screen — the scroll offset
/// is an exact position. Per-frame GPU work is proportional to visible
/// complexity, but it is the *same* every frame, and constancy is what
/// smoothness actually needs.
final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let frameQueue: MTLCommandQueue
    private let setupQueue: MTLCommandQueue
    private let pass: BoardPass
    private let post: PostPass
    private let ring: TileRing
    private let store: TileStore
    let stats: FrameStats

    private var params: BoardParams
    /// Draw-time only, so changing it never invalidates a tile.
    private var options = RenderOptions()
    /// Started only once the whole strip is resident — see `revealedAt`.
    private var clock: ScrollClock
    /// When the *live* board first became complete. Until then nothing moves
    /// and no clock has started: the board arrives whole instead of being
    /// watched to assemble itself out of the distance.
    private var revealedAt: CFTimeInterval?
    /// The same, per parallax layer. The layers behind fade in as they land, so
    /// the picture is not held up by depth nobody is looking at yet.
    private var layerRevealedAt = [CFTimeInterval?](repeating: nil,
                                                    count: Parallax.layers.count)
    private var layerScratch: Set<Int> = []
    private var frameClock: FrameClock
    /// Filled in by the drawable's presented handler; drained once a frame.
    private let presentation = PresentationClock()
    private var scale: Float = 1
    /// Where the board was last frame, so the temporal pass knows how far it
    /// slid. The whole of the scene's motion — nothing else moves.
    private var lastScrollY: Double = 0
    private let refreshInterval: Double

    /// Keeps the CPU from running more than a couple of frames ahead of the
    /// GPU, so a stall shows up as a stall rather than as unbounded latency.
    private let inFlight = DispatchSemaphore(value: 3)
    private var gpuTimer: GPUTimer?
    private var lastTimerReport = 0.0

    /// The app is sandboxed, so it cannot write a report to an arbitrary path,
    /// and under Launch Services it has no stdout to print to. The unified log
    /// is reachable from both. One line per call — a multi-line message gets
    /// truncated.
    private static let timingLog = Logger(subsystem: "com.pcb.render", category: "gputiming")

    private func emitTiming(_ text: String) {
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            Renderer.timingLog.notice("\(line, privacy: .public)")
        }
    }

    /// Reused every frame so the render thread does not allocate.
    private var wantedScratch: [Int] = []
    /// Tiles that are actually on screen this frame, across every layer. The
    /// ring may not evict these; anything else is fair game.
    private var visibleScratch: Set<Int> = []
    private var readyScratch: [(index: Int, slot: Int)] = []

    /// Main actor because it configures the `MTKView`, which is main-actor
    /// isolated. The delegate methods already are, through the protocol; only
    /// the initialiser had to be said out loud. It is built from SwiftUI's
    /// view-making, which is on the main actor anyway, so this costs a
    /// declaration and no hops.
    @MainActor
    init?(view: MTKView, params: BoardParams, stats: FrameStats) {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice(),
              let frameQueue = device.makeCommandQueue(),
              let setupQueue = device.makeCommandQueue(),
              let pass = BoardPass(device: device, colorFormat: PostPass.hdrFormat),
              let post = PostPass(device: device, drawableFormat: view.colorPixelFormat)
        else { return nil }
        self.device = device
        self.frameQueue = frameQueue
        self.setupQueue = setupQueue
        self.pass = pass
        self.post = post
        self.params = params
        self.stats = stats
        ring = TileRing(device: device)
        clock = ScrollClock(now: CACurrentMediaTime())
        frameClock = FrameClock(refreshRate: view.preferredFramesPerSecond)
        refreshInterval = 1.0 / Double(max(view.preferredFramesPerSecond, 1))
        store = TileStore(params: params, ring: ring)
        store.beginBurst()
        if ProcessInfo.processInfo.environment["PCB_GPU_TIMING"] != nil {
            gpuTimer = GPUTimer(device: device, framesInFlight: 3)
            post.timer = gpuTimer
            let state = gpuTimer == nil ? "UNSUPPORTED" : "on"
            Renderer.timingLog.notice("gpu timing: \(state, privacy: .public)")
        }
        wantedScratch.reserveCapacity(64)
        readyScratch.reserveCapacity(Layout.maxCachedTiles)

        view.framebufferOnly = true
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        let substrate = params.palette.substrate
        view.clearColor = MTLClearColor(red: Double(substrate.x),
                                        green: Double(substrate.y),
                                        blue: Double(substrate.z), alpha: 1)
        super.init()
        stats.setRefreshRate(view.preferredFramesPerSecond)
    }

    /// Takes the sliders from the UI. Board geometry is owned by the renderer —
    /// it comes from the drawable, not from SwiftUI — so it is carried over.
    func apply(_ ui: BoardParams) {
        var merged = ui
        merged.geometry = params.geometry
        guard merged != params else { return }
        params = merged
        syncStore()
    }

    /// Draw-time settings. Deliberately not routed through `apply`: these cost
    /// nothing but a different draw call, and pushing them through the
    /// generation identity would throw away every cached tile.
    func apply(_ new: RenderOptions) {
        // The pad brightness is baked into a tile's instances, so this one
        // setting does have to re-encode. It is still not a *rebuild*: nothing
        // is re-routed, the generated boards are untouched.
        let reencode = new.glow != options.glow
        options = new
        if reencode { syncStore() }
    }

    private func syncStore() {
        let filler = TileFiller(geo: params.geometry, scale: scale,
                                teardrops: params.teardrops,
                                palette: params.palette,
                                glow: options.glow)
        store.update(params, filler: filler)
    }

    private var tilePixelHeight: Int {
        max(1, Int((Board.tileHeight * scale).rounded()))
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        inFlight.wait()
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor else {
            inFlight.signal()
            return
        }

        let pixelW = Int(view.drawableSize.width)
        let pixelH = Int(view.drawableSize.height)
        guard pixelW > 0, pixelH > 0 else {
            inFlight.signal()
            return
        }

        // Backing scale, derived from the drawable rather than the screen so a
        // resized window and a retina display are the same code path.
        let newScale = Float(view.drawableSize.width / max(view.bounds.width, 1))
        if newScale != scale { scale = newScale }
        // The tilted board is laid out wider than the window so that magnifying
        // it to cover the far end of the strip does not also magnify the
        // copper. The flat view maps board units to points, so it takes the
        // window width unchanged.
        let widthScale = options.tilted ? Board.tiltedWidthScale : 1
        let newGeometry = BoardGeometry(width: Float(pixelW) / scale * widthScale)
        if newGeometry != params.geometry || store.needsFiller {
            params.geometry = newGeometry
            syncStore()
        }

        // ── Where the board is ──────────────────────────────────────────────
        // What the display did with the last frame, before deciding where the
        // board is for this one.
        frameClock.observePresented(presentation.latest)
        let now = frameClock.tick(now: CACurrentMediaTime())
        // Nothing has moved yet if the strip is still filling in — the drift
        // starts from the moment the board is whole.
        let scrollY = revealedAt == nil ? 0 : clock.position(at: now)
        let viewHeightPoints = Double(pixelH) / Double(scale)

        // Solved before residency, because how far down the strip is visible
        // falls out of the camera: a wider board is scaled up, so each tile
        // covers more ground and fewer of them fill the same depth.
        let aspect = Float(pixelW) / Float(max(pixelH, 1))
        let camera = Camera.setup(aspect: aspect, boardWidth: params.geometry.width)

        // Enough slots for what THIS window needs, rather than one fixed number.
        //
        // A bigger window lays out a wider board, which scales the world down,
        // which means each tile covers less ground and more of them are needed
        // to reach the top of the frame. Fixed at 48 that ran out above about
        // 2045 points of width at 16:10 — "More Space" on a large laptop, or any
        // external 4K — and the symptom was the board *ending on screen*, with
        // the haze finishing partway up the picture and new tiles arriving in
        // plain sight at that edge.
        //
        // Sized from the camera instead, so a phone carries a phone's ring
        // (about 30 slots) and a 6K display gets what it actually needs. At
        // 1.45 MB a slot that is the difference between 43 MB and 93 MB.
        // Growing it re-allocates, which drops every resident tile — but the
        // only thing that changes it is a resize, and a resize already changes
        // the board width and invalidates them all anyway.
        let tileUnits = Double(Board.tileHeight)
        let span = Int((camera.depthBoardUnits / tileUnits).rounded(.up)) + 1
        let needed = min((span + Layout.prefetchAbove + Layout.prefetchBelow)
                         * Parallax.layers.count, Layout.maxCachedTiles)
        if ring.slotCount < needed { _ = ring.allocate(slots: needed) }

        // Which tiles are on screen.
        //
        // The tilted view runs the strip AWAY from the eye, so what is visible
        // extends to *lower* tile indices — the direction new board arrives
        // from — not forward the way the flat view scrolled. Getting this
        // backwards leaves the board a thin sliver at the bottom of the frame
        // with empty sky above it.
        let front: ClosedRange<Int>
        if options.tilted {
            let depthBoardUnits = camera.depthBoardUnits
            let tile = Double(Board.tileHeight)
            let farthest = Int(((scrollY - depthBoardUnits) / tile).rounded(.down))
            // A whole tile of slack here was a whole tile of board that is
            // never visible at any scroll position — generated, and drawn, for
            // nothing. Measured: world z 162…324 against a visible range of
            // −868…0. The only thing that actually reaches back past the near
            // edge is a seam trace overhanging its tile by a few units.
            let nearest = Int(((scrollY + Double(Board.seamOverhang)) / tile)
                              .rounded(.down))
            front = farthest...max(farthest, nearest)
        } else {
            front = TileLayout.visibleRange(scrollY: scrollY,
                                            heightPoints: viewHeightPoints)
        }
        // Flat view: the live board only. Without perspective there is no depth
        // for the inner layers to sit in, so they would just stack three
        // unrelated boards on top of one another.
        let layers = options.tilted ? Parallax.layers : [Parallax.foreground]

        // Each parallax layer draws a different stretch of the infinite board,
        // so every layer needs its own run of tiles resident — three disjoint
        // index ranges rather than one.
        //
        // Until the live board is up, the others are not asked for at all.
        // Every layer, from the first frame: the reveal waits for all of them,
        // so leaving any out of the request would hold the picture back
        // indefinitely. What *is* deferred is the prefetch lead — those tiles
        // are for seconds from now, and starting them alongside the ones the
        // reveal is waiting on only splits the machine between them.
        let launching = revealedAt == nil
        wantedScratch.removeAll(keepingCapacity: true)
        visibleScratch.removeAll(keepingCapacity: true)
        let lowest = front.lowerBound - (launching ? 0 : Layout.prefetchAbove)
        let highest = front.upperBound + (launching ? 0 : Layout.prefetchBelow)

        for layer in layers {
            for i in lowest...highest { wantedScratch.append(i + layer.indexOffset) }
            for i in front { visibleScratch.insert(i + layer.indexOffset) }
        }
        // Nearest to the viewport first: that is what the eye reaches soonest.
        // Distance is measured *within* a layer — after subtracting its offset —
        // so a shifted layer is not treated as infinitely far away.
        //
        // Until the live board is up, its tiles jump the whole queue. That is
        // most of the launch time: waiting for all three layers meant twenty-one
        // tiles of A\* before anything appeared, where the layer the eye is
        // actually on needs seven.
        let centre = (front.lowerBound + front.upperBound) / 2
        wantedScratch.sort { rank($0, centre: centre) < rank($1, centre: centre) }
        store.request(wantedScratch, keeping: visibleScratch)

        expandArrivals(near: centre)

        // ── The reveal ──────────────────────────────────────────────────────
        // Per layer, and the live board goes first. The board still arrives
        // whole — no layer is ever seen half-built — but the depth behind it
        // fades in as it lands rather than holding up the picture. Latched: a
        // gap opening later is a dropped tile for the prefetch to catch, not a
        // reason to hide the board and start over.
        for (i, layer) in layers.enumerated() where layerRevealedAt[i] == nil {
            layerScratch.removeAll(keepingCapacity: true)
            for j in front { layerScratch.insert(j + layer.indexOffset) }
            guard ring.allReady(layerScratch) else { continue }
            layerRevealedAt[i] = now
            guard layer.drawsDetail else { continue }
            // The live board is what the drift and every clock are timed from.
            revealedAt = now
            clock = ScrollClock(now: now)
            // Off the launch lane: there is a frame to protect from here on.
            store.endBurst()
        }

        // Just the clock. Where each trace's head is depends on that trace's own
        // phase, period and speed, which only the fragment shader can resolve.
        let palette = params.palette
        let pulseTime = Float(revealedAt.map { now - $0 } ?? 0)

        // ── Draw ────────────────────────────────────────────────────────────
        var drawn = 0
        post.resize(width: pixelW, height: pixelH)

        guard let command = frameQueue.makeCommandBuffer() else { inFlight.signal(); return }
        gpuTimer?.beginFrame()
        command.addCompletedHandler { [inFlight] _ in inFlight.signal() }
        guard let enc = post.beginScene(in: command) else { return }

        // The board sits on the XZ plane. Under the tilted camera the scroll
        // carries it toward the eye; the flat projection is the same board seen
        // straight down, which is what the tests measure against.
        let viewProj = options.tilted
            ? camera.viewProj
            : Camera.orthographic(pixelWidth: Float(pixelW),
                                  pixelHeight: Float(pixelH), scale: scale)

        for (layerIndex, layer) in layers.enumerated() {
            let shift = layer.indexOffset
            let layerRange = (front.lowerBound + shift)...(front.upperBound + shift)
            ring.collectReady(in: layerRange, into: &readyScratch)

            // Drop tiles the haze has already finished with.
            //
            // The farthest drawn tile sits *entirely* beyond `fogEnd` by design
            // — that is what stops a tile popping into view when it arrives
            // (see `Camera.hazedDepth`). Every one of its fragments is shaded
            // and then multiplied by a fog of zero, once per layer, every frame.
            //
            // A tile's nearest edge is its far board edge carried by the scroll;
            // if even that is past where the haze completes, nothing it draws
            // can reach the screen. Conservative on purpose: the parallax layers
            // sit a little above and below the plane this measures, so the test
            // has to be one that cannot cull something still faintly visible.
            if options.tilted {
                readyScratch.removeAll {
                    Camera.isFullyHazed(place: $0.index - layer.indexOffset,
                                        scrollY: scrollY, setup: camera)
                }
            }
            guard !readyScratch.isEmpty else { continue }
            drawn = max(drawn, readyScratch.count)

            var uniforms = PCBViewUniforms()
            uniforms.viewProj = viewProj
            uniforms.viewHalfW = Float(pixelW) * 0.5
            uniforms.viewHalfH = Float(pixelH) * 0.5
            // The tilted camera looks down the middle of the strip, so the
            // board is centred on it. The flat view maps board x straight to
            // pixels and must not be shifted.
            uniforms.originX = options.tilted
                ? layer.offsetX - params.geometry.width / 2
                : layer.offsetX
            uniforms.height = options.tilted ? layer.height : 0
            uniforms.fade = layer.fade
            uniforms.subR = palette.substrate.x
            uniforms.subG = palette.substrate.y
            uniforms.subB = palette.substrate.z
            uniforms.worldScale = options.tilted ? camera.worldScale : 1
            uniforms.camX = camera.eye.x
            uniforms.camY = camera.eye.y
            uniforms.camZ = camera.eye.z
            // Depth of field only makes sense once there is depth.
            uniforms.focusDist = camera.focusDistance
            uniforms.cocScale = options.tilted ? Camera.cocScale : 0
            uniforms.cocMax = Camera.cocMax
            uniforms.fogStart = options.tilted ? camera.fogStart : 0
            uniforms.fogEnd = options.tilted ? camera.fogEnd : 0
            // Each layer fades up from its own moment, so the live board is not
            // held back by the depth behind it.
            uniforms.alpha = Anim.introAlpha(revealedAt: layerRevealedAt[layerIndex],
                                             now: now)
            uniforms.pulseTime = pulseTime
            uniforms.runR = palette.runner.x
            uniforms.runG = palette.runner.y
            uniforms.runB = palette.runner.z
            uniforms.runMix = palette.runner.w
            uniforms.pulseGain = layer.pulses ? Pulse.gain : 0

            // The *inner* layers lay their substrate down opaque before their
            // copper, so each is a solid board rather than a floating mesh of
            // wires and the stack reads as a stack.
            //
            // The live board does not, and that is the whole point: it is the
            // one you look through. Slabbing it too hid every layer behind it,
            // since it spans the full width and the offsets between layers are
            // small next to that — three boards, only one of them visible.
            //
            // Tilted only. Flat, the layers are coplanar and a slab would
            // simply cover the board.
            if options.tilted && !layer.drawsDetail {
                pass.begin(.solids, halo: palette.traceHalo, enc)
                for (index, _) in readyScratch {
                    let place = index - layer.indexOffset
                    uniforms.originZ = Float(Double(place) * Double(Board.tileHeight)
                                             - scrollY) + layer.offsetZ
                    pass.drawSubstrate(width: params.geometry.width,
                                       colour: palette.substrate * layer.substrateDim,
                                       view: &uniforms, in: enc)
                }
            }

            // Stage-major across tiles: a seam trace overhangs its own tile by
            // a few units, so drawing a whole tile at a time would let one
            // tile's halo darken its neighbour's copper.
            for stage in BoardPass.Stage.allCases {
                // A deep layer draws copper only — silkscreen and package
                // bodies at 20% fade are noise. Its pads are optional, and
                // switching them on is a draw-call decision, not a rebuild.
                let padsOnly = !layer.drawsDetail && stage == .solids
                if !layer.drawsDetail {
                    if stage == .glyphs { continue }
                    if padsOnly && !options.backgroundPads { continue }
                }
                pass.begin(stage, halo: palette.traceHalo, enc)
                for (index, slot) in readyScratch {
                    // Board y = 0 of this tile, in world Z: its place in the
                    // strip, carried by the scroll. The layer's index offset
                    // only selects *which* circuitry to draw, so it is taken
                    // back out here — otherwise the layer would sit hundreds of
                    // thousands of tiles away.
                    let place = index - layer.indexOffset
                    uniforms.originZ = Float(Double(place) * Double(Board.tileHeight)
                                             - scrollY) + layer.offsetZ
                    if padsOnly {
                        pass.drawPads(slot: slot, ring: ring, view: &uniforms, in: enc)
                    } else {
                        pass.draw(stage, slot: slot, ring: ring, view: &uniforms, in: enc)
                    }
                }
            }
        }

        enc.endEncoding()

        // The depth of field is a screen-space pass, and everything it needs to
        // know about depth is the two ends of the strip.
        var postOptions = options.post
        if options.tilted {
            postOptions.dofStrength = 1
            postOptions.invNear = camera.invNear
            postOptions.invFar = camera.invFar
            postOptions.focusDistance = camera.focusDistance
        }
        // ── Temporal accumulation ───────────────────────────────────────────
        // The board is a plane and the camera never moves, so the only thing
        // that changed between frames is how far the strip has slid — which is
        // the whole motion vector field, in one number. See `temporalAccumulate`.
        var temporalFrame: TemporalFrame?
        if options.tilted, revealedAt != nil {
            var view = PCBViewUniforms()
            view.viewProj = camera.viewProj
            view.camX = camera.eye.x
            view.camY = camera.eye.y
            view.camZ = camera.eye.z
            let delta = Float(scrollY - lastScrollY) * camera.worldScale
            temporalFrame = TemporalFrame(
                view: view,
                invViewProj: camera.viewProj.inverse,
                boardDelta: delta)
        }
        lastScrollY = scrollY

        post.finish(into: rpd, options: postOptions, temporal: temporalFrame,
                    in: command)

        stats.beginFrame(at: CACurrentMediaTime(), readyTiles: drawn)
        command.addCompletedHandler { [weak stats] buffer in
            stats?.recordFrameGpu(buffer.gpuEndTime - buffer.gpuStartTime)
        }
        if let gpuTimer {
            gpuTimer.endFrame(command)
            let now = CACurrentMediaTime()
            if lastTimerReport == 0 { lastTimerReport = now }
            // Discard the first stretch: the board is still arriving and the
            // burst lane is loaded, so the passes are not doing steady work.
            else if now - lastTimerReport > 5, gpuTimer.frameCount > 120 {
                emitTiming(gpuTimer.report())
                gpuTimer.reset()
                lastTimerReport = now
            }
        }
        #if os(macOS)
        // When this frame actually reaches the screen, which is what the board's
        // motion is paced from — see `FrameClock`.
        //
        // macOS only, and not by choice: `addPresentedHandler` and
        // `presentedTime` are macOS/Mac Catalyst, with no iOS counterpart on
        // `MTLDrawable`. iOS would want `CADisplayLink.targetTimestamp`, which
        // `MTKView` owns privately and does not hand out. Without a reference,
        // `FrameClock` falls back to pacing off the draw call — the behaviour
        // both platforms had before this — so the phone is no worse than it was,
        // just not yet better.
        drawable.addPresentedHandler { [presentation] presented in
            presentation.record(presented.presentedTime)
        }

        // Pin the presentation cadence so a ProMotion panel holds its rate
        // instead of drifting down when the app is this cheap to draw.
        command.present(drawable, afterMinimumDuration: refreshInterval)
        #else
        command.present(drawable)
        #endif
        command.commit()
    }

    /// How urgent a tile is, in tiles from the viewport.
    ///
    /// Each parallax layer draws a different stretch of the board, shifted by
    /// its `indexOffset`, so raw index distance is meaningless across layers —
    /// a background tile would always score as six hundred thousand away. The
    /// offset is subtracted first, and the best (nearest) layer wins.
    private func rank(_ index: Int, centre: Int) -> Int {
        var best = Int.max
        for layer in Parallax.layers {
            best = min(best, abs(index - layer.indexOffset - centre))
        }
        return best
    }

    /// Expands newly arrived tiles into capsule instances. One small dispatch
    /// per tile, on its own queue and outside the frame's command buffer.
    private func expandArrivals(near centre: Int) {
        // While the board is still filling in there is no frame to protect, so
        // arrivals are not paced.
        let budget = revealedAt == nil
            ? Layout.burstExpansionsPerFrame
            : Layout.expansionsPerFrame
        var done = 0
        while done < budget,
              let next = ring.nextToExpand(rank: { rank($0, centre: centre) }),
              let command = setupQueue.makeCommandBuffer() {
            let slot = next.slot
            let epoch = ring.epoch
            command.label = "expand tile \(next.index)"
            pass.encodeExpansion(slot: slot, ring: ring, in: command)
            // The ring, not `self`. This only ever needed the ring, and
            // capturing the renderer pulled a main-actor object into a
            // `@Sendable` closure that runs on whatever thread Metal finishes on.
            command.addCompletedHandler { [ring, weak stats] buffer in
                stats?.recordBakeGpu(buffer.gpuEndTime - buffer.gpuStartTime)
                ring.markReady(slot, epoch: epoch)
            }
            command.commit()
            done += 1
        }
    }
}
