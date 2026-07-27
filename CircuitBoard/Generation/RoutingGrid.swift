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

    private let occStore: UnsafeMutableBufferPointer<UInt8>
    private let hugStore: UnsafeMutableBufferPointer<UInt8>

    /// Cells that became occupied since the last hug settle. `hug` is a pure
    /// function of `occ`, and Chebyshev dilation is local, so re-deriving only
    /// the neighbourhood of these cells is identical to rebuilding the whole
    /// field — which the HTML did after every single `claim`.
    private var pendingSources: [Int32] = []
    private var frontier: [Int32] = []
    private var nextFrontier: [Int32] = []

    init(cols: Int, rows: Int) {
        self.cols = cols
        self.rows = rows
        count = cols * rows
        occStore = .allocate(capacity: count)
        hugStore = .allocate(capacity: count)
        occStore.initialize(repeating: 0)
        hugStore.initialize(repeating: 0)
        pendingSources.reserveCapacity(1024)
        frontier.reserveCapacity(4096)
        nextFrontier.reserveCapacity(4096)
    }

    deinit {
        occStore.deallocate()
        hugStore.deallocate()
    }

    var occ: UnsafeMutableBufferPointer<UInt8> { occStore }
    var hug: UnsafeMutableBufferPointer<UInt8> { hugStore }

    @inline(__always) func index(_ x: Int, _ y: Int) -> Int { y * cols + x }
    @inline(__always) func inBounds(_ x: Int, _ y: Int) -> Bool {
        x >= 0 && y >= 0 && x < cols && y < rows
    }

    func clear() {
        occStore.update(repeating: 0)
        hugStore.update(repeating: 0)
        pendingSources.removeAll(keepingCapacity: true)
    }

    // MARK: - Occupancy

    /// Out-of-bounds counts as NOT free, so a footprint never straddles an edge.
    func boxFree(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) -> Bool {
        if x0 < 0 || y0 < 0 || x1 >= cols || y1 >= rows { return false }
        for y in y0...y1 {
            let row = y * cols
            for x in x0...x1 where occStore[row + x] != 0 { return false }
        }
        return true
    }

    func markBox(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) {
        let ly = max(0, y0), hy = min(rows - 1, y1)
        let lx = max(0, x0), hx = min(cols - 1, x1)
        guard ly <= hy, lx <= hx else { return }
        for y in ly...hy {
            let row = y * cols
            for x in lx...hx {
                let i = row + x
                if occStore[i] == 0 {
                    occStore[i] = 1
                    pendingSources.append(Int32(i))
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
            for ax in lx...hx where occStore[row + ax] != 0 { return true }
        }
        return false
    }

    /// Release a reserved cell. Distances can grow again, so callers must
    /// follow with `rebuildHug()` rather than the incremental settle.
    func release(_ i: Int) {
        occStore[i] = 0
    }

    /// A cell is illegal for a trace of keepout radius `r` if any cell within
    /// `r` is copper. Off-grid counts as FREE, not blocked: the board continues
    /// past the tile, and treating it as blocked made rows 0 and rows-1
    /// unreachable for every r > 0 trace — which silently killed every
    /// main-class seam port.
    @inline(__always)
    func blocked(_ x: Int, _ y: Int, _ r: Int) -> Bool {
        let i = index(x, y)
        if r == 0 { return occStore[i] == 1 }
        // `hug` is already the Chebyshev distance to the nearest copper, capped
        // at hugRadius and zero on copper itself — so "is there copper within
        // one cell" is exactly "occupied, or hug says the nearest is one away".
        // One load instead of a 3×3 scan, in A*'s innermost loop, and exact
        // rather than an approximation. Guarded by a test against the scan.
        if r == 1 { return occStore[i] != 0 || hugStore[i] == 1 }
        let ly = max(0, y - r), hy = min(rows - 1, y + r)
        let lx = max(0, x - r), hx = min(cols - 1, x + r)
        guard ly <= hy, lx <= hx else { return false }
        for ay in ly...hy {
            let row = ay * cols
            for ax in lx...hx where occStore[row + ax] != 0 { return true }
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

    /// Incremental, decrease-only dilation from the cells that just became
    /// copper. Chebyshev distance to a union of sets is the min of the
    /// distances, and every previously stored value was already exact, so this
    /// lands on precisely the field a full rebuild would produce.
    func settleHug() {
        guard !pendingSources.isEmpty else { return }
        frontier.removeAll(keepingCapacity: true)
        frontier.append(contentsOf: pendingSources)
        pendingSources.removeAll(keepingCapacity: true)

        for level in 1...Routing.hugRadius {
            nextFrontier.removeAll(keepingCapacity: true)
            let l = UInt8(level)
            for i in frontier {
                let x = Int(i) % cols, y = Int(i) / cols
                for dy in -1...1 {
                    let ny = y + dy
                    if ny < 0 || ny >= rows { continue }
                    let row = ny * cols
                    for dx in -1...1 {
                        let nx = x + dx
                        if nx < 0 || nx >= cols { continue }
                        let j = row + nx
                        if occStore[j] != 0 { continue }
                        if hugStore[j] != 0 && hugStore[j] <= l { continue }
                        hugStore[j] = l
                        nextFrontier.append(Int32(j))
                    }
                }
            }
            swap(&frontier, &nextFrontier)
            if frontier.isEmpty { break }
        }
        frontier.removeAll(keepingCapacity: true)
    }

    /// Full multi-source rebuild. Only needed after cells are *released*
    /// (the per-port seam lanes), where distances can grow.
    func rebuildHug() {
        hugStore.update(repeating: 0)
        pendingSources.removeAll(keepingCapacity: true)
        frontier.removeAll(keepingCapacity: true)
        for i in 0..<count where occStore[i] != 0 { frontier.append(Int32(i)) }
        for level in 1...Routing.hugRadius {
            nextFrontier.removeAll(keepingCapacity: true)
            let l = UInt8(level)
            for i in frontier {
                let x = Int(i) % cols, y = Int(i) / cols
                for dy in -1...1 {
                    let ny = y + dy
                    if ny < 0 || ny >= rows { continue }
                    let row = ny * cols
                    for dx in -1...1 {
                        let nx = x + dx
                        if nx < 0 || nx >= cols { continue }
                        let j = row + nx
                        if occStore[j] != 0 || hugStore[j] != 0 { continue }
                        hugStore[j] = l
                        nextFrontier.append(Int32(j))
                    }
                }
            }
            swap(&frontier, &nextFrontier)
            if frontier.isEmpty { break }
        }
        frontier.removeAll(keepingCapacity: true)
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
