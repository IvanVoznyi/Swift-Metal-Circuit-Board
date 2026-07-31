import Foundation
import simd

/// The routing substrate.
///
///   occ — hard blocked: copper plus its own half of the clearance.
///   hug — the free channel immediately outside occ, held as a *graded*
///         Chebyshev distance 1…hugRadius rather than an adjacent-or-not flag.
///         A binary flag only ever paid off when the shortest path already
///         grazed existing copper, so nothing bundled; a graded field pulls a
///         route toward a neighbouring corridor from a few cells away, which
///         is what makes traces run together and then peel off once the
///         detour stops paying.
final class RoutingGrid {
    let cols: Int
    let rows: Int
    let count: Int

    /// One bit per cell instead of one byte. `occ` and `hug` together were
    /// 32.8 KB against a 64 KB L1, competing with A*'s 345 KB of per-cell
    /// state; this takes the pair to 18.5 KB.
    private let occWords: UnsafeMutablePointer<UInt64>
    private let wordCount: Int
    private let hugStore: UnsafeMutableBufferPointer<UInt8>

    /// Bumped by every change `blocked` can see — occupancy and the hug field
    /// both. It lets a caller cache something derived from the grid and know
    /// exactly when to throw it away; `Router`'s component labels are the one
    /// that matters, and they are only sound because this counts hug changes
    /// too, `blocked(_:_:1)` being a hug lookup.
    private(set) var version: UInt64 = 0

    /// Visited marks for the local hug repair. Separate from the field itself
    /// because the repair has to propagate *through* cells whose value is
    /// already correct, which it could not do if the value were the mark.
    ///
    /// Sixteen bits, not thirty-two. The width buys nothing but a longer gap
    /// between wraps, and a wrap is already handled exactly — clear the marks,
    /// restart at 1 — so it only decides how often that clear runs. At the
    /// measured eighteen repairs a tile it is one 33 KB memset every ~3,600
    /// tiles, against 33 KB of resident memory saved on every worker for the
    /// whole run. `initialize` touches every page, so this buffer is resident
    /// in full whether the marks are used or not.
    private let repairStamp: UnsafeMutableBufferPointer<UInt16>
    private var repairGeneration: UInt16 = 0

    /// Cells that became occupied since the last hug settle. `hug` is a pure
    /// function of `occ`, and Chebyshev dilation is local, so re-deriving only
    /// the neighbourhood of these cells is identical to rebuilding the whole
    /// field — which the HTML did after every single `claim`.
    ///
    /// These three are raw buffers rather than `[Int32]`. They are written from
    /// the innermost loop of every field update, where a Swift array costs an
    /// exclusivity check on the class property, a uniqueness check on the
    /// buffer and a capacity check per append. A dilation can enqueue a cell at
    /// most once per level, so `count` is a hard bound and no growth is ever
    /// possible — the checks were guarding against something that cannot happen.
    private var pendingSources: UnsafeMutablePointer<Int32>
    private var pendingCount = 0
    private var frontier: UnsafeMutablePointer<Int32>
    private var nextFrontier: UnsafeMutablePointer<Int32>
    private var frontCount = 0

    /// Frontier entries are packed `x | y << 16`, not linear indices. Unpacking
    /// an index costs `i % cols` and `i / cols` — a real integer division,
    /// twenty thousand times a tile, because `cols` is not known at compile
    /// time. Every producer already holds x and y when it enqueues.
    @inline(__always) private static func pack(_ x: Int, _ y: Int) -> Int32 {
        Int32(x | (y << 16))
    }

    init(cols: Int, rows: Int) {
        self.cols = cols
        self.rows = rows
        count = cols * rows
        wordCount = (count + 63) / 64
        occWords = .allocate(capacity: wordCount)
        hugStore = .allocate(capacity: count)
        repairStamp = .allocate(capacity: count)
        pendingSources = .allocate(capacity: count)
        frontier = .allocate(capacity: count)
        nextFrontier = .allocate(capacity: count)
        occWords.initialize(repeating: 0, count: wordCount)
        hugStore.initialize(repeating: 0)
        repairStamp.initialize(repeating: 0)
    }

    deinit {
        occWords.deallocate()
        hugStore.deallocate()
        repairStamp.deallocate()
        pendingSources.deallocate()
        frontier.deallocate()
        nextFrontier.deallocate()
    }

    var hug: UnsafeMutableBufferPointer<UInt8> { hugStore }

    @inline(__always) func isSet(_ i: Int) -> Bool {
        occWords[i >> 6] & (1 << UInt64(i & 63)) != 0
    }
    @inline(__always) private func set(_ i: Int) {
        occWords[i >> 6] |= 1 << UInt64(i & 63)
    }
    @inline(__always) private func unset(_ i: Int) {
        occWords[i >> 6] &= ~(1 << UInt64(i & 63))
    }

    @inline(__always) func index(_ x: Int, _ y: Int) -> Int { y * cols + x }
    @inline(__always) func inBounds(_ x: Int, _ y: Int) -> Bool {
        x >= 0 && y >= 0 && x < cols && y < rows
    }

    func clear() {
        version &+= 1
        occWords.update(repeating: 0, count: wordCount)
        hugStore.update(repeating: 0)
        pendingCount = 0
    }

    // MARK: - Occupancy

    /// Out-of-bounds counts as NOT free, so a footprint never straddles an edge.
    func boxFree(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) -> Bool {
        if x0 < 0 || y0 < 0 || x1 >= cols || y1 >= rows { return false }
        for y in y0...y1 {
            let row = y * cols
            for x in x0...x1 where isSet(row + x) { return false }
        }
        return true
    }

    func markBox(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
        version &+= 1
        let ly = max(0, y0), hy = min(rows - 1, y1)
        let lx = max(0, x0), hx = min(cols - 1, x1)
        guard ly <= hy, lx <= hx else { return }
        for y in ly...hy {
            let row = y * cols
            for x in lx...hx {
                let i = row + x
                if !isSet(i) {
                    set(i)
                    pendingSources[pendingCount] = RoutingGrid.pack(x, y)
                    pendingCount += 1
                }
                hugStore[i] = 0
            }
        }
    }

    func markCell(_ gx: Int, _ gy: Int, _ r: Int) {
        markBox(gx - r, gy - r, gx + r, gy + r)
    }

    /// The 3×3 scan `blocked(x, y, 1)` replaces with a single load. Kept so a
    /// test can prove the two agree on random grids.
    func blockedByScan(_ x: Int, _ y: Int, _ r: Int) -> Bool {
        let ly = max(0, y - r), hy = min(rows - 1, y + r)
        let lx = max(0, x - r), hx = min(cols - 1, x + r)
        guard ly <= hy, lx <= hx else { return false }
        for ay in ly...hy {
            let row = ay * cols
            for ax in lx...hx where isSet(row + ax) { return true }
        }
        return false
    }

    /// Claim a single cell outright — no keepout, no hug bookkeeping. The seam
    /// lanes want exactly this: reserved before the field is built, released
    /// one at a time as each port takes its turn.
    ///
    /// Returns false if it was already copper. It exists so that writing to
    /// `occ` goes through something that bumps `version`; a raw write would
    /// leave a cached component label looking valid after the grid moved under
    /// it.
    @discardableResult
    func reserve(_ i: Int) -> Bool {
        guard !isSet(i) else { return false }
        version &+= 1
        set(i)
        return true
    }

    /// Release a whole reserved lane and bring `hug` back to exactly the field
    /// a full rebuild would produce — touching only the part that can have
    /// changed.
    ///
    /// `hug[p]` is the Chebyshev distance from p to the nearest copper, capped
    /// at `hugRadius`. Removing copper can only affect cells within that radius
    /// of what was removed, and the new value of such a cell is decided by
    /// copper within that radius of *it* — so the box grown by `hugRadius` is
    /// what must be recomputed, and the box grown by twice it holds every
    /// source that can decide the answer. Everything outside is already exact,
    /// and is propagated through rather than rewritten.
    ///
    /// For the 3×9 seam lane that is ~107 cells against the grid's 16,800, and
    /// this runs once per seam port — about nineteen times per tile, which is
    /// what made the full rebuild a fifth of generation.
    func release(_ cells: [Int]) {
        guard !cells.isEmpty else { return }
        // The repair assumes the field is exact for the copper still standing.
        // Anything marked and not yet settled would be lost outside the box,
        // where nothing recomputes it.
        if pendingCount > 0 { settleHug() }
        version &+= 1

        var x0 = cols, y0 = rows, x1 = -1, y1 = -1
        for i in cells {
            unset(i)
            let x = i % cols, y = i / cols
            x0 = min(x0, x); x1 = max(x1, x)
            y0 = min(y0, y); y1 = max(y1, y)
        }

        let radius = Routing.hugRadius
        let dx0 = max(0, x0 - radius), dy0 = max(0, y0 - radius)
        let dx1 = min(cols - 1, x1 + radius), dy1 = min(rows - 1, y1 + radius)
        let sx0 = max(0, x0 - 2 * radius), sy0 = max(0, y0 - 2 * radius)
        let sx1 = min(cols - 1, x1 + 2 * radius), sy1 = min(rows - 1, y1 + 2 * radius)

        for y in dy0...dy1 {
            let row = y * cols
            for x in dx0...dx1 { hugStore[row + x] = 0 }
        }

        repairGeneration &+= 1
        if repairGeneration == 0 { repairStamp.update(repeating: 0); repairGeneration = 1 }
        let gen = repairGeneration

        var m = 0
        for y in sy0...sy1 {
            let row = y * cols
            for x in sx0...sx1 where isSet(row + x) {
                repairStamp[row + x] = gen
                frontier[m] = RoutingGrid.pack(x, y)
                m += 1
            }
        }
        frontCount = m
        guard m > 0 else { return }

        dilate { j, x, y, l in
            if repairStamp[j] == gen { return false }
            repairStamp[j] = gen
            // Outside the dilation box the stored value is already exact, so
            // the ripple passes through without rewriting it.
            if x >= dx0, x <= dx1, y >= dy0, y <= dy1 { hugStore[j] = l }
            return true
        }
    }

    /// `blocked` for a caller that already holds the linear index. The
    /// coordinate form recomputes `y * cols + x` on every call, and the
    /// reachability pass makes eight of them per cell.
    @inline(__always)
    func blockedAt(_ i: Int, _ r: Int) -> Bool {
        if r == 0 { return isSet(i) }
        if r == 1 { return isSet(i) || hugStore[i] == 1 }
        return blocked(i % cols, i / cols, r)
    }

    /// A cell is illegal for a trace of keepout radius `r` if any cell within
    /// `r` is copper. Off-grid counts as FREE, not blocked: the board continues
    /// past the tile, and treating it as blocked made rows 0 and rows-1
    /// unreachable for every r > 0 trace — which silently killed every
    /// main-class seam port.
    @inline(__always)
    func blocked(_ x: Int, _ y: Int, _ r: Int) -> Bool {
        let i = index(x, y)
        if r == 0 { return isSet(i) }
        // `hug` is already the Chebyshev distance to the nearest copper, capped
        // at hugRadius and zero on copper itself — so "is there copper within
        // one cell" is exactly "occupied, or hug says the nearest is one away".
        // One load instead of a 3×3 scan, in A*'s innermost loop, and exact
        // rather than an approximation. Guarded by a test against the scan.
        if r == 1 { return isSet(i) || hugStore[i] == 1 }
        let ly = max(0, y - r), hy = min(rows - 1, y + r)
        let lx = max(0, x - r), hx = min(cols - 1, x + r)
        guard ly <= hy, lx <= hx else { return false }
        for ay in ly...hy {
            let row = ay * cols
            for ax in lx...hx where isSet(row + ax) { return true }
        }
        return false
    }

    /// Reserve the corridor a routed path occupies, then bring `hug` up to date.
    func claim(_ path: [SIMD2<Int32>], width: Float) {
        let r = RoutingGrid.keepout(width)
        for p in path { markCell(Int(p.x), Int(p.y), r) }
        settleHug()
    }

    // MARK: - Hug field

    /// The one Chebyshev dilation in the file. All three field updates — the
    /// incremental settle, the full rebuild and the local repair — are the same
    /// breadth-first ripple outward from copper and differ only in the test for
    /// "is this neighbour new", so that test is the parameter.
    ///
    /// `frontier` holds the cells at level `l-1`, packed. `admit(j, x, y, l)`
    /// decides a neighbour, records whatever it wants to record, and returns
    /// whether the ripple continues through it — taking the coordinates as well
    /// as the index so that no caller has to divide them back out. Copper is
    /// never a step: a cell whose
    /// line to the nearest copper crosses other copper is nearer to *that* one
    /// instead, so stopping there changes no answer, and the check is common to
    /// all three callers.
    ///
    /// The neighbour range is clamped once per cell rather than tested once per
    /// neighbour. Six branches become four selects, and every interior cell —
    /// which is nearly all of them — walks a full 3×3 with no bounds logic at
    /// all.
    @inline(__always)
    private func dilate(_ admit: (Int, Int, Int, UInt8) -> Bool) {
        for level in 1...Routing.hugRadius {
            let src = frontier, dst = nextFrontier
            let n = frontCount
            var m = 0
            let l = UInt8(level)
            for k in 0..<n {
                let e = src[k]
                let x = Int(e & 0xFFFF), y = Int(e >> 16)
                let y0 = y > 0 ? y - 1 : y, y1 = y < rows - 1 ? y + 1 : y
                let x0 = x > 0 ? x - 1 : x, x1 = x < cols - 1 ? x + 1 : x
                for ny in y0...y1 {
                    let row = ny * cols
                    for nx in x0...x1 {
                        let j = row + nx
                        if isSet(j) { continue }
                        // The centre cell needs no special case: at level 1 it
                        // is copper and `isSet` drops it, and deeper it already
                        // carries a value `admit` rejects.
                        if !admit(j, nx, ny, l) { continue }
                        dst[m] = RoutingGrid.pack(nx, ny)
                        m += 1
                    }
                }
            }
            frontier = dst
            nextFrontier = src
            frontCount = m
            if m == 0 { break }
        }
        frontCount = 0
    }

    /// Incremental, decrease-only dilation from the cells that just became
    /// copper. Chebyshev distance to a union of sets is the min of the
    /// distances, and every previously stored value was already exact, so this
    /// lands on precisely the field a full rebuild would produce.
    func settleHug() {
        version &+= 1
        guard pendingCount > 0 else { return }
        frontier.update(from: pendingSources, count: pendingCount)
        frontCount = pendingCount
        pendingCount = 0
        dilate { j, _, _, l in
            if hugStore[j] != 0 && hugStore[j] <= l { return false }
            hugStore[j] = l
            return true
        }
    }

    /// Full multi-source rebuild. Only needed after cells are *released*
    /// (the per-port seam lanes), where distances can grow.
    func rebuildHug() {
        version &+= 1
        hugStore.update(repeating: 0)
        pendingCount = 0

        // Seed from every occupied cell. Walking the bitset word by word and
        // stepping the set bits out of it touches one word per sixty-four cells
        // instead of testing all sixteen thousand one at a time; the row stays
        // in step with the scan, so no cell costs a division.
        var m = 0, y = 0, rowEnd = cols
        for w in 0..<wordCount {
            var bits = occWords[w]
            while bits != 0 {
                let i = (w << 6) | bits.trailingZeroBitCount
                bits &= bits &- 1
                while i >= rowEnd { y += 1; rowEnd += cols }
                frontier[m] = RoutingGrid.pack(i - rowEnd + cols, y)
                m += 1
            }
        }
        frontCount = m
        guard m > 0 else { return }

        dilate { j, _, _, l in
            if hugStore[j] != 0 { return false }
            hugStore[j] = l
            return true
        }
    }

    // MARK: - Free space

    func freeCell(_ rng: Rng, keepout r: Int) -> SIMD2<Int32>? {
        for _ in 0..<Routing.freeCellTries {
            let x = rng.int(1, cols - 2), y = rng.int(1, rows - 2)
            if !blocked(x, y, r) { return SIMD2(Int32(x), Int32(y)) }
        }
        return nil
    }

    // MARK: - Keepout sizing

    /// Keepout radius in cells for a trace of stroke width `w`. Sized against
    /// `pitch` so that for any pair of widths a, b the reserved corridors
    /// guarantee centre separation ≥ a/2 + b/2 + clearance — copper never
    /// touches. fine/signal/bus all land on r = 0 and may share adjacent lanes
    /// (that is what lets them bundle); only main roads reserve wider.
    static func keepout(_ w: Float) -> Int {
        let v = (Double(w) / 2 + Double(Board.clearance) / 2) / Double(Board.pitch) - 0.5
        return max(0, Int(ceil(v)))
    }

    /// Keepout radius for a pad whose copper reaches `ext` px from centre along
    /// that axis. A route may only occupy cells at Chebyshev ≥ r+1, and a
    /// polyline through such cells stays ≥ (r+1)·GS from the centre, so
    /// (r+1)·GS ≥ ext + widestThinTrace/2 + clearance is the whole requirement.
    /// The obvious `ceil(pr/GS)+1` is about twice this and turned every
    /// footprint into a wall.
    static func padKeepout(_ ext: Float) -> Int {
        let v = (Double(ext) + Double(TraceClass.widestThin) / 2 + Double(Board.clearance))
            / Double(Board.gridSize) - 1
        return max(1, Int(ceil(v)))
    }
}
