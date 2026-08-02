import Foundation
import Metal
import os

/// The one place a resident tile lives: a fixed ring of slots, every GPU
/// resource allocated once and reused forever.
///
/// The previous design allocated six `MTLBuffer`s and an `MTLTexture` per tile
/// on a worker thread and freed them on eviction. Metal resource creation takes
/// driver locks and GPU virtual memory; doing it while the render thread is
/// encoding is a reliable way to lose a frame. Here a worker only ever
/// `memcpy`s into memory that already exists.
///
/// The layout is deliberately flat — parallel arrays indexed by slot, no
/// per-tile objects, no dictionary, no reference counting on the hot path. It
/// also collapses what used to be two structures kept in sync by hand (the
/// tile cache and the slice pool) into one.
/// Every slot's GPU storage.
///
/// The ONE place this project asserts something the compiler cannot check, and
/// it is asserting a gap in Metal's annotations rather than an argument of its
/// own. `MTLDevice`, `MTLCommandQueue` and the pipeline states are `Sendable`;
/// `MTLBuffer` and `MTLTexture` are deliberately not, because their *contents*
/// are mutable and unsynchronised. The synchronisation those contents need is
/// the slot state machine in `Book`, which the lock below enforces.
///
/// Boxed so the unsafety has a name and a size. Marking the whole ring
/// `@unchecked Sendable` would have waived checking on the bookkeeping too,
/// which is exactly the part worth checking.
private struct SlotStorage: @unchecked Sendable {
    var pathPoints: [MTLBuffer] = []
    var traceInfos: [MTLBuffer] = []
    var traceVertices: [MTLBuffer] = []
    var solids: [MTLBuffer] = []
    var glyphs: [MTLBuffer] = []
    var atlases: [MTLTexture] = []
}

/// `Sendable` for real, not by assertion: every stored property is an immutable
/// `let` of a `Sendable` type, and all the mutable state lives inside the lock —
/// which owns it, so there is no way to reach it unguarded. The manual
/// `lock()`/`unlock()` pairs this replaces were spread over a dozen methods with
/// early returns in several, which is where an unguarded access gets added by
/// accident.
final class TileRing: Sendable {

    enum State: UInt8 {
        case free        // nothing here
        case generating  // a worker owns it
        case filled      // CPU data in place, capsules not yet expanded
        case ready       // expanded; safe to draw
    }

    /// Fixed capacities, in elements. A tile that would exceed one is clamped:
    /// dropping a few instances off the busiest board in a thousand is a far
    /// better trade than an allocation on the wrong thread.
    struct Capacity {
        var pathPoints = Layout.maxPathPoints
        /// Two ribbon corners per centreline point, plus a duplicate at each end
        /// of every trace — those are what stitch all the traces in a tile into
        /// one strip, so the copper goes out in a single draw call.
        var traceVertices = Layout.maxPathPoints * 2 + Layout.maxTraces * 2
        var traceInfos = Layout.maxTraces
        var solids = Layout.maxSolids
        var glyphs = Layout.maxGlyphs
        var atlasWidth = Layout.glyphAtlasWidth
        var atlasHeight = Layout.glyphAtlasHeight
    }

    /// What a worker fills in and the baker reads back. Plain counts — the
    /// buffers themselves belong to the slot.
    struct Counts {
        var pathPoints = 0
        var traceInfos = 0
        var solids = 0
        /// Where the pads start inside `solids`; everything from here on is a
        /// pad, so a layer can draw pads alone.
        var padStart = 0
        var glyphs = 0
        /// How many ribbon corners the expansion kernel laid out, degenerate
        /// stitching vertices included. Set by the encoder, which is where the
        /// trace boundaries are known.
        var traceVertices = 0
    }

    let device: MTLDevice
    let capacity = Capacity()

    /// Everything mutable, in one place, reachable only through the lock.
    private struct Book {
        var slotCount = 0
        // ── Parallel arrays, one entry per slot ──────────────────────────────
        var state: [State] = []
        var tileIndex: [Int] = []
        var counts: [Counts] = []
        var storage = SlotStorage()
        /// Bumped when the board changes; results from before are dropped on
        /// arrival rather than being written into a slot someone else now owns.
        var epoch: UInt64 = 0
    }

    /// Held only across the bookkeeping — never across generation, encoding or
    /// an allocation, so the render thread never waits on a worker doing real
    /// work. `OSAllocatedUnfairLock` rather than a bare `os_unfair_lock` because
    /// it *owns* the state: there is no way to read a slot's state without
    /// taking the lock, which a separate lock and separate fields could never
    /// guarantee.
    private let book = OSAllocatedUnfairLock(initialState: Book())

    var slotCount: Int { book.withLock { $0.slotCount } }
    var epoch: UInt64 { book.withLock { $0.epoch } }

    init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Allocation, once

    /// Allocates every slot's storage. Called only when the drawable geometry
    /// changes; the steady state never reaches this.
    func allocate(slots: Int) -> Bool {
        guard slots > 0 else { return false }
        var points: [MTLBuffer] = [], infos: [MTLBuffer] = []
        var instances: [MTLBuffer] = [], solids: [MTLBuffer] = []
        var glyphs: [MTLBuffer] = [], atlases: [MTLTexture] = []

        let atlasDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: capacity.atlasWidth, height: capacity.atlasHeight,
            mipmapped: false)
        atlasDescriptor.usage = .shaderRead
        atlasDescriptor.storageMode = .shared

        for _ in 0..<slots {
            guard
                let p = device.makeBuffer(
                    length: capacity.pathPoints * MemoryLayout<PCBPathPoint>.stride,
                    options: .storageModeShared),
                let i = device.makeBuffer(
                    length: capacity.traceInfos * MemoryLayout<PCBTraceInfo>.stride,
                    options: .storageModeShared),
                let x = device.makeBuffer(
                    length: capacity.traceVertices * MemoryLayout<PCBTraceVertex>.stride,
                    options: .storageModePrivate),
                let s = device.makeBuffer(
                    length: capacity.solids * MemoryLayout<PCBShapeInstance>.stride,
                    options: .storageModeShared),
                let g = device.makeBuffer(
                    length: capacity.glyphs * MemoryLayout<PCBGlyphInstance>.stride,
                    options: .storageModeShared),
                let a = device.makeTexture(descriptor: atlasDescriptor)
            else { return false }
            points.append(p); infos.append(i); instances.append(x)
            solids.append(s); glyphs.append(g); atlases.append(a)
        }

        // Boxed before the lock, not inside it: `withLock`'s closure is
        // `@Sendable`, so the bare arrays cannot cross into it — which is the
        // check working. `SlotStorage` is the one declared place that crossing
        // is allowed, so it is the thing that goes through.
        let storage = SlotStorage(pathPoints: points, traceInfos: infos,
                                  traceVertices: instances, solids: solids,
                                  glyphs: glyphs, atlases: atlases)
        book.withLock {
            $0.slotCount = slots
            $0.state = Array(repeating: .free, count: slots)
            $0.tileIndex = Array(repeating: Int.min, count: slots)
            $0.counts = Array(repeating: Counts(), count: slots)
            $0.storage = storage
            $0.epoch &+= 1
        }
        return true
    }

    /// Everything currently held is invalid — new board, new geometry.
    func invalidate() {
        book.withLock {
            for i in $0.state.indices {
                $0.state[i] = .free
                $0.tileIndex[i] = Int.min
                $0.counts[i] = Counts()
            }
            $0.epoch &+= 1
        }
    }

    // MARK: - Buffers (immutable after allocate, no lock needed)

    // Through the lock like everything else. `allocate` can replace these when
    // the window grows and the ring has to hold more of the strip, so "immutable
    // after allocation" was only true until the first resize.
    //
    // The box comes out and is indexed outside: a bare `MTLBuffer` cannot be
    // returned from `withLock`, whose result has to be `Sendable`. Handing back
    // `SlotStorage` — six array references, copy-on-write — is the declared
    // crossing point, and it costs six retains on a path that runs a few hundred
    // times a frame.
    private var storage: SlotStorage { book.withLock { $0.storage } }

    func pathPoints(_ slot: Int) -> MTLBuffer { storage.pathPoints[slot] }
    func traceInfos(_ slot: Int) -> MTLBuffer { storage.traceInfos[slot] }
    func traceVertices(_ slot: Int) -> MTLBuffer { storage.traceVertices[slot] }
    func solids(_ slot: Int) -> MTLBuffer { storage.solids[slot] }
    func glyphs(_ slot: Int) -> MTLBuffer { storage.glyphs[slot] }
    func atlas(_ slot: Int) -> MTLTexture { storage.atlases[slot] }

    // MARK: - Bookkeeping

    func slot(for index: Int) -> Int? {
        book.withLock {
            guard let s = $0.tileIndex.firstIndex(of: index),
                  $0.state[s] != .free else { return nil }
            return s
        }
    }

    func state(of slot: Int) -> State {
        book.withLock { $0.state[slot] }
    }

    func counts(of slot: Int) -> Counts {
        book.withLock { $0.counts[slot] }
    }

    func isResident(_ index: Int) -> Bool { slot(for: index) != nil }

    /// True when every one of `indices` is resident *and* expanded, so the
    /// whole set can be drawn this frame. One lock for the lot: this is asked
    /// per frame until the board is complete.
    func allReady(_ indices: Set<Int>) -> Bool {
        book.withLock { b in
            var found = 0
            for s in 0..<b.slotCount
            where b.state[s] == .ready && indices.contains(b.tileIndex[s]) {
                found += 1
            }
            return found == indices.count
        }
    }

    /// Claims a slot for `index`, evicting a resident tile outside `keep` if
    /// nothing is free. Returns nil when every slot is either in flight or on
    /// screen.
    ///
    /// `keep` is a set rather than a range because the parallax layers each
    /// draw their own stretch of the board: what is on screen is several
    /// disjoint runs of indices, not one.
    /// `wanted` is everything the renderer asked for this frame — the visible
    /// tiles *and* the prefetch ring around them. `keep` is only what is on
    /// screen. Evicting from `wanted` is what to avoid: those tiles are about to
    /// be asked for again, so throwing one out buys a slot and immediately
    /// spends a tile of routing to get it back. Measured before this
    /// distinction existed, the ring was regenerating about three hundred and
    /// thirty tiles a second to stand still — three cores, permanently, for a
    /// board that needs a new tile every eight seconds.
    func claim(_ index: Int, keeping keep: Set<Int>,
               wanted: Set<Int>) -> (slot: Int, epoch: UInt64)? {
        book.withLock { b -> (slot: Int, epoch: UInt64)? in
            guard b.slotCount > 0 else { return nil }
            if let existing = b.tileIndex.firstIndex(of: index),
               b.state[existing] != .free {
                return (existing, b.epoch)
            }
            var chosen: Int?
            if let free = b.state.firstIndex(of: .free) {
                chosen = free
            } else {
                // Two tiers. First choice is a tile nothing wants any more —
                // scrolled past, or left over from a previous viewport. Only if
                // there is no such slot does a wanted-but-offscreen prefetch
                // tile get taken, which at least keeps the board progressing
                // when the window is tall enough that the wanted set rivals the
                // ring.
                var fallback: Int?
                for s in 0..<b.slotCount {
                    guard b.state[s] == .ready || b.state[s] == .filled else { continue }
                    let tile = b.tileIndex[s]
                    guard !keep.contains(tile) else { continue }
                    if !wanted.contains(tile) { chosen = s; break }
                    if fallback == nil { fallback = s }
                }
                if chosen == nil { chosen = fallback }
            }
            guard let slot = chosen else { return nil }
            b.state[slot] = .generating
            b.tileIndex[slot] = index
            b.counts[slot] = Counts()
            return (slot, b.epoch)
        }
    }

    /// A worker finished writing CPU-side data into the slot's buffers.
    /// Rejected if the board moved on while it was working.
    func markFilled(_ slot: Int, index: Int, counts newCounts: Counts,
                    epoch workerEpoch: UInt64) -> Bool {
        book.withLock { b in
            guard workerEpoch == b.epoch, b.tileIndex[slot] == index,
                  b.state[slot] == .generating else { return false }
            b.counts[slot] = newCounts
            b.state[slot] = .filled
            return true
        }
    }

    /// A worker gave up — release the slot rather than stranding it.
    func abandon(_ slot: Int, index: Int, epoch workerEpoch: UInt64) {
        book.withLock { b in
            guard workerEpoch == b.epoch, b.tileIndex[slot] == index,
                  b.state[slot] == .generating else { return }
            b.state[slot] = .free
            b.tileIndex[slot] = Int.min
        }
    }

    func markReady(_ slot: Int, epoch expandEpoch: UInt64) {
        book.withLock { b in
            if expandEpoch == b.epoch, b.state[slot] == .filled { b.state[slot] = .ready }
        }
    }

    /// The next tile whose data is in place but whose capsules have not been
    /// expanded. Nearest to the viewport first, so what the eye is about to
    /// reach is finished first.
    /// `rank` scores a tile index in tiles-from-the-viewport; lower goes first.
    /// It is a closure because only the renderer knows how each parallax layer
    /// maps its shifted indices back onto the strip.
    func nextToExpand(rank: (Int) -> Int) -> (slot: Int, index: Int)? {
        // The candidates come out of the lock and `rank` is applied outside it.
        // `rank` is the renderer's closure, not `Sendable`, and the compiler is
        // right to stop it crossing — but it is also caller code, and running
        // caller code while holding a lock the render thread wants is how a
        // small closure turns into a stall.
        let filled = book.withLock { b -> [(slot: Int, index: Int)] in
            var hits: [(slot: Int, index: Int)] = []
            for s in 0..<b.slotCount where b.state[s] == .filled {
                hits.append((s, b.tileIndex[s]))
            }
            return hits
        }
        var best: (slot: Int, index: Int)?
        var bestDistance = Int.max
        for candidate in filled {
            let d = rank(candidate.index)
            if d < bestDistance { bestDistance = d; best = candidate }
        }
        return best
    }

    /// Tiles that are `ready` and inside `range`, **far to near**, collected
    /// into a caller-owned buffer. The buffer is reused frame after frame —
    /// this runs on the render thread, which should not be allocating at all.
    ///
    /// The order is the point, and it belongs here rather than at the call site.
    /// There is no depth buffer: `BoardView` sets `depthStencilPixelFormat` to
    /// `.invalid` because every edge is antialiased analytically, so this is a
    /// painter's algorithm and **the order these come back in is the only depth
    /// information the board pass has**. Blending is source-over, and adjacent
    /// tiles overlap on purpose — `Draw.substrateOverlap` is a quarter of a
    /// tile, and seam traces run past their own edge — so a farther tile handed
    /// over after a nearer one paints its substrate across the nearer one's
    /// copper.
    ///
    /// Slot order is not that order and never was. `claim` takes the first free
    /// or first evictable slot, so the mapping permutes every time new board
    /// arrives at the back, and the composite flipped with it: once every
    /// `tileHeight / scrollSpeed` seconds, at the top of the frame. Measured on
    /// the old code the ring returned `[0, -5, -4, -3, -2, -1]` — the nearest
    /// tile first, then the five behind it painting over it.
    ///
    /// Sorting is bounded by the visible span, at most eight tiles a layer, so
    /// the cost is nothing next to being wrong.
    func collectReady(in range: ClosedRange<Int>, into out: inout [(index: Int, slot: Int)]) {
        out.removeAll(keepingCapacity: true)
        // `out` is an `inout` the caller owns, so it cannot cross into the
        // closure — collect inside the lock, hand back, append outside.
        let found = book.withLock { b -> [(index: Int, slot: Int)] in
            var hits: [(index: Int, slot: Int)] = []
            for s in 0..<b.slotCount
            where b.state[s] == .ready && range.contains(b.tileIndex[s]) {
                hits.append((b.tileIndex[s], s))
            }
            return hits
        }
        out.append(contentsOf: found)
        // Ascending index is far-to-near: the strip runs away from the eye
        // toward *lower* indices, which is the direction new board arrives from.
        out.sort { $0.index < $1.index }
    }
}
