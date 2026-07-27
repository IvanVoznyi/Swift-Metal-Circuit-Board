import Foundation
import simd

/// Binary-heap A* over the routing grid: octile moves, no corner cutting, a
/// turn penalty, and a per-hug-level step multiplier that makes riding an
/// existing corridor cheaper than crossing open board.
///
/// The four per-node arrays are allocated once for the whole tile and reused
/// across every route through a generation stamp, instead of being refilled
/// per call — a tile routes a few hundred times, and `Float32Array(N).fill()`
/// four times per route was pure overhead.
final class Router {
    private let grid: RoutingGrid
    private let count: Int

    private let gScore: UnsafeMutableBufferPointer<Float>
    private let cameFrom: UnsafeMutableBufferPointer<Int32>
    private let dirOf: UnsafeMutableBufferPointer<Int8>
    private let visitStamp: UnsafeMutableBufferPointer<UInt32>
    private let closedStamp: UnsafeMutableBufferPointer<UInt32>
    private var generation: UInt32 = 0

    /// The heap is the hottest structure in the search — every push sifts and
    /// every pop sifts back — so it lives in the same manually-managed arena as
    /// the score arrays rather than paying an `Array` bounds check per swap.
    private let heapF: UnsafeMutableBufferPointer<Float>
    private let heapV: UnsafeMutableBufferPointer<Int32>
    private var heapCount = 0
    private let heapCapacity: Int

    /// `HUGMUL` — also the heuristic's scale, so A* stays admissible against
    /// the cheapest possible lane.
    var hugMultiplier: Float = 0.72 {
        didSet { rebuildLanes() }
    }
    /// Per-hug-level step multiplier. `.0` is open board, `.1` is right
    /// alongside existing copper.
    private var lanes: (Float, Float, Float, Float) = (1, 0.72, 0.81, 0.91)

    init(grid: RoutingGrid) {
        self.grid = grid
        count = grid.count
        gScore = .allocate(capacity: count)
        cameFrom = .allocate(capacity: count)
        dirOf = .allocate(capacity: count)
        visitStamp = .allocate(capacity: count)
        closedStamp = .allocate(capacity: count)
        gScore.initialize(repeating: .infinity)
        cameFrom.initialize(repeating: -1)
        dirOf.initialize(repeating: -1)
        visitStamp.initialize(repeating: 0)
        closedStamp.initialize(repeating: 0)
        // Every open-list entry is a distinct cell at most once per push, and
        // a cell can be pushed once per incoming direction.
        heapCapacity = count * 8 + 16
        heapF = .allocate(capacity: heapCapacity)
        heapV = .allocate(capacity: heapCapacity)
        heapF.initialize(repeating: 0)
        heapV.initialize(repeating: 0)
        rebuildLanes()
    }

    deinit {
        gScore.deallocate()
        cameFrom.deallocate()
        dirOf.deallocate()
        visitStamp.deallocate()
        closedStamp.deallocate()
        heapF.deallocate()
        heapV.deallocate()
    }

    private func rebuildLanes() {
        let m = hugMultiplier
        lanes = (1, m, m + (1 - m) / 3, m + 2 * (1 - m) / 3)
    }

    /// Indexed by `hug` level, 0...3. A tuple rather than an array so the
    /// innermost loop does not bounds-check a four-element lookup.
    @inline(__always)
    private func lane(_ level: UInt8) -> Float {
        switch level {
        case 0: return lanes.0
        case 1: return lanes.1
        case 2: return lanes.2
        default: return lanes.3
        }
    }

    /// The eight octile steps, clockwise, as functions rather than a global
    /// array — the innermost loop should not index a heap-allocated table.
    @inline(__always) static func dx(_ i: Int) -> Int {
        switch i { case 0, 1, 7: return 1; case 3, 4, 5: return -1; default: return 0 }
    }
    @inline(__always) static func dy(_ i: Int) -> Int {
        switch i { case 1, 2, 3: return 1; case 5, 6, 7: return -1; default: return 0 }
    }

    // MARK: - Heap
    // Ported comparison-for-comparison so tie-breaking, and therefore the
    // chosen path among equal-cost routes, is reproducible.

    @inline(__always)
    private func heapPush(_ f: Float, _ v: Int32) {
        guard heapCount < heapCapacity else { return }
        heapF[heapCount] = f
        heapV[heapCount] = v
        var i = heapCount
        heapCount += 1
        while i > 0 {
            let p = (i - 1) >> 1
            if heapF[p] <= heapF[i] { break }
            let tf = heapF[p], tv = heapV[p]
            heapF[p] = heapF[i]; heapV[p] = heapV[i]
            heapF[i] = tf; heapV[i] = tv
            i = p
        }
    }

    @inline(__always)
    private func heapPop() -> Int32 {
        let top = heapV[0]
        let last = heapCount - 1
        heapF[0] = heapF[last]
        heapV[0] = heapV[last]
        heapCount = last
        let n = heapCount
        var i = 0
        while true {
            let l = 2 * i + 1, r = l + 1
            var m = i
            if l < n && heapF[l] < heapF[m] { m = l }
            if r < n && heapF[r] < heapF[m] { m = r }
            if m == i { break }
            let tf = heapF[m], tv = heapV[m]
            heapF[m] = heapF[i]; heapV[m] = heapV[i]
            heapF[i] = tf; heapV[i] = tv
            i = m
        }
        return top
    }

    // MARK: - Route

    /// Returns the centreline in grid cells, or nil if no legal corridor of
    /// keepout `r` exists inside the search budget.
    func route(from s: SIMD2<Int32>, to e: SIMD2<Int32>, keepout r: Int) -> [SIMD2<Int32>]? {
        let cols = grid.cols, rows = grid.rows
        let start = Int(s.y) * cols + Int(s.x)
        let goal = Int(e.y) * cols + Int(e.x)

        // Refuse an already-claimed start. Callers used to be trusted to check
        // this, and the one that forgot — seamFallback, on a port a through-
        // trace had just landed on — produced traces starting exactly on top
        // of another trace.
        if start != goal && grid.blocked(Int(s.x), Int(s.y), r) { return nil }

        generation &+= 1
        if generation == 0 {  // wrapped: stale stamps could alias
            visitStamp.update(repeating: 0)
            closedStamp.update(repeating: 0)
            generation = 1
        }
        let gen = generation
        heapCount = 0

        let hw = hugMultiplier
        let ex = Float(e.x), ey = Float(e.y)
        @inline(__always) func heuristic(_ x: Int, _ y: Int) -> Float {
            let dx = abs(Float(x) - ex), dy = abs(Float(y) - ey)
            return (max(dx, dy) + Routing.diagonalExtra * min(dx, dy)) * hw
        }

        gScore[start] = 0
        visitStamp[start] = gen
        cameFrom[start] = -1
        dirOf[start] = -1
        heapPush(heuristic(Int(s.x), Int(s.y)), Int32(start))

        var pops = 0
        var found = false
        let hug = grid.hug

        while heapCount > 0 {
            let cur = Int(heapPop())
            if closedStamp[cur] == gen { continue }
            closedStamp[cur] = gen
            if cur == goal { found = true; break }
            pops += 1
            if pops > Routing.budget { break }

            let cx = cur % cols, cy = cur / cols
            let cd = Int(dirOf[cur])
            let curG = gScore[cur]

            for di in 0..<8 {
                let dx = Router.dx(di), dy = Router.dy(di)
                let nx = cx + dx, ny = cy + dy
                if nx < 0 || ny < 0 || nx >= cols || ny >= rows { continue }
                let ni = ny * cols + nx
                if closedStamp[ni] == gen { continue }
                if ni != goal && grid.blocked(nx, ny, r) { continue }
                // No corner cutting. Relaxing this — allowing a diagonal past
                // an occupied orthogonal neighbour — roughly doubled seam
                // continuity and trace count, and was reverted: measurement
                // showed it reintroduced real copper overlaps. A trace never
                // touching another trace is the rule that matters most. The
                // connectivity it bought is recovered instead by not walling
                // off whole footprints (see padKeepout).
                if dx != 0 && dy != 0 {
                    if grid.blocked(cx + dx, cy, r) || grid.blocked(cx, cy + dy, r) { continue }
                }
                let step: Float = (dx != 0 && dy != 0) ? Routing.diagonalStep : 1
                let lane = lane(hug[ni])
                var turn: Float = 0
                if cd >= 0 {
                    let raw = abs(di - cd)
                    let dd = raw > 4 ? 8 - raw : raw
                    turn = dd == 0 ? 0 : (dd == 1 ? Routing.turn45
                                                  : Routing.turn90 * Float(dd) * 0.5)
                }
                let ng = curG + step * lane + turn
                let known = visitStamp[ni] == gen ? gScore[ni] : .infinity
                if ng < known {
                    gScore[ni] = ng
                    visitStamp[ni] = gen
                    cameFrom[ni] = Int32(cur)
                    dirOf[ni] = Int8(di)
                    heapPush(ng + heuristic(nx, ny), Int32(ni))
                }
            }
        }

        guard found else { return nil }
        var path: [SIMD2<Int32>] = []
        var c = goal
        while c != -1 {
            path.append(SIMD2(Int32(c % cols), Int32(c / cols)))
            let prev = visitStamp[c] == gen ? Int(cameFrom[c]) : -1
            c = prev
        }
        path.reverse()
        return path
    }
}
