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
    /// The last `Routing.spikeWindow` directions of the path reaching each cell,
    /// one nibble each, newest in the low bits, 0xF for "no history". Six
    /// dependent loads down the `cameFrom` chain per neighbour cost 70% of a
    /// tile; this is one load per pop and one AND per neighbour.
    private let recentDirs: UnsafeMutableBufferPointer<UInt32>
    private let visitStamp: UnsafeMutableBufferPointer<UInt32>
    private let closedStamp: UnsafeMutableBufferPointer<UInt32>
    private var generation: UInt32 = 0

    /// The reachability pre-pass: its own visited stamp and a plain FIFO.
    private let reachStamp: UnsafeMutableBufferPointer<UInt32>
    private let reachQueue: UnsafeMutableBufferPointer<Int32>
    private var reachGeneration: UInt32 = 0

    /// Component id per cell, written by a pre-pass that drained without
    /// finding its goal — at that point the queue it walked *is* the component.
    /// Ids are minted monotonically and never reused, so an epoch is discarded
    /// by moving `firstValidLabel` forward rather than by clearing 16k cells.
    private let label: UnsafeMutableBufferPointer<UInt32>
    private var nextLabel: UInt32 = 1
    private var firstValidLabel: UInt32 = 1
    private var labelVersion: UInt64 = .max

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
        recentDirs = .allocate(capacity: count)
        visitStamp = .allocate(capacity: count)
        closedStamp = .allocate(capacity: count)
        reachStamp = .allocate(capacity: count)
        reachQueue = .allocate(capacity: count)
        label = .allocate(capacity: count)
        gScore.initialize(repeating: .infinity)
        cameFrom.initialize(repeating: -1)
        dirOf.initialize(repeating: -1)
        recentDirs.initialize(repeating: Router.noHistory)
        visitStamp.initialize(repeating: 0)
        closedStamp.initialize(repeating: 0)
        reachStamp.initialize(repeating: 0)
        reachQueue.initialize(repeating: 0)
        label.initialize(repeating: 0)
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
        recentDirs.deallocate()
        visitStamp.deallocate()
        closedStamp.deallocate()
        reachStamp.deallocate()
        reachQueue.deallocate()
        label.deallocate()
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

    /// Every nibble a sentinel: a path with nothing behind it yet.
    static let historyMask: UInt32 = (1 << UInt32(4 * Routing.spikeWindow)) - 1
    static let noHistory: UInt32 = Router.historyMask

    /// Directions that would turn 135 degrees or more away from `d`, as a bit
    /// per direction. Built once so the innermost loop tests a mask instead of
    /// walking anything.
    static let opposed: [UInt32] = (0..<8).map { d in
        var m: UInt32 = 0
        for k in 3...5 { m |= 1 << UInt32((d + k) & 7) }
        return m
    }

    /// Turn magnitude between two of the eight directions, in 45 degree steps.
    @inline(__always) static func turnSteps(_ a: Int, _ b: Int) -> Int {
        let raw = abs(a - b)
        return raw > 4 ? 8 - raw : raw
    }

    /// Directions that would double the line back on itself, given where it has
    /// been for the last few cells.
    ///
    /// The window is what the eye reads as one place — six cells, the ~44 px a
    /// wedge spans before it stops looking like a kink and starts looking like a
    /// turn. A drawn segment is a whole run of same-direction cells, so two
    /// vertices of the drawn line can be five cells apart; a shorter window
    /// misses exactly the shape this is here to forbid, and a wider one starts
    /// refusing honest detours.
    @inline(__always)
    private func forbiddenDirections(_ history: UInt32) -> UInt32 {
        var h = history
        var mask: UInt32 = 0
        for _ in 0..<Routing.spikeWindow {
            let d = h & 0xF
            if d != 0xF { mask |= Router.opposed[Int(d)] }
            h >>= 4
        }
        return mask
    }

    // MARK: - Reachability

    /// Is `goal` reachable from `start` under the router's own move rule?
    ///
    /// Ninety-five percent of route requests have no corridor at all — a pad
    /// walled in by its own footprint's keepout, a pocket the last claim sealed
    /// — and A* proves that the expensive way: it drains its open list through
    /// the whole pocket, evaluating a lane multiplier and a turn cost at every
    /// one of some hundreds of pops, because a heap cannot know it is enclosed
    /// until it is empty. This answers the same question with a FIFO and two
    /// byte loads per neighbour.
    ///
    /// Only a false *reject* could change a board, so the move rule below is
    /// the router's own, unrelaxed. Over-accepting is free: A* then runs and
    /// fails exactly as it used to, which across the whole test matrix happens
    /// seven times in 33,000 routes.
    private func reachable(from start: Int, to goal: Int, keepout r: Int) -> Bool {
        if start == goal { return true }
        let cols = grid.cols, rows = grid.rows

        // A drained probe leaves the pocket it walked labelled, and a failed
        // route changes nothing on the grid — so the next probe out of that
        // same pocket, and there are usually several before anything is
        // claimed, costs two loads instead of another walk.
        if grid.version != labelVersion {
            firstValidLabel = nextLabel
            labelVersion = grid.version
        }
        let known = label[start]
        if known >= firstValidLabel { return component(known, touches: goal) }

        reachGeneration &+= 1
        if reachGeneration == 0 { reachStamp.update(repeating: 0); reachGeneration = 1 }
        let gen = reachGeneration

        var head = 0, tail = 0
        reachStamp[start] = gen
        // Packed (y << 16 | x): the queue carries the coordinates it already
        // knows rather than making the next pop divide them back out.
        reachQueue[tail] = Int32(bitPattern: UInt32((start / cols) << 16 | (start % cols)))
        tail += 1

        // Enqueue if unvisited and enterable. Reports the goal the moment it is
        // touched — and the goal is exempt from the occupancy test exactly as it
        // is in `route`, because a trace is allowed to land on copper there.
        // Blocked cells get stamped too: they are never enqueued, but stamping
        // them keeps the next neighbour that looks at them from testing again.
        @inline(__always)
        func step(_ ni: Int, _ nx: Int, _ ny: Int) -> Bool {
            if ni == goal { return true }
            if reachStamp[ni] == gen { return false }
            reachStamp[ni] = gen
            if grid.blockedAt(ni, r) { return false }
            reachQueue[tail] = Int32(bitPattern: UInt32(ny << 16 | nx))
            tail += 1
            return false
        }

        while head < tail {
            let packed = UInt32(bitPattern: reachQueue[head]); head += 1
            let cx = Int(packed & 0xFFFF), cy = Int(packed >> 16)
            let cur = cy * cols + cx
            let up = cur - cols, down = cur + cols

            if cx > 0, cy > 0, cx < cols - 1, cy < rows - 1 {
                // The four orthogonal neighbours are also the four corner cells
                // the diagonals have to test, so they are tested once here
                // instead of twice more inside each diagonal case.
                let bE = grid.blockedAt(cur + 1, r), bW = grid.blockedAt(cur - 1, r)
                let bS = grid.blockedAt(down, r),    bN = grid.blockedAt(up, r)

                if step(cur + 1, cx + 1, cy) { return true }
                if step(cur - 1, cx - 1, cy) { return true }
                if step(down, cx, cy + 1) { return true }
                if step(up, cx, cy - 1) { return true }
                if !bE && !bS, step(down + 1, cx + 1, cy + 1) { return true }
                if !bW && !bS, step(down - 1, cx - 1, cy + 1) { return true }
                if !bW && !bN, step(up - 1, cx - 1, cy - 1) { return true }
                if !bE && !bN, step(up + 1, cx + 1, cy - 1) { return true }
            } else {
                // Edge cell: the same eight steps with the bounds tests the
                // interior case is allowed to skip.
                for di in 0..<8 {
                    let dx = Router.dx(di), dy = Router.dy(di)
                    let nx = cx + dx, ny = cy + dy
                    if nx < 0 || ny < 0 || nx >= cols || ny >= rows { continue }
                    if dx != 0 && dy != 0 {
                        if grid.blocked(cx + dx, cy, r) || grid.blocked(cx, cy + dy, r) { continue }
                    }
                    if step(ny * cols + nx, nx, ny) { return true }
                }
            }
        }

        // Drained: everything enqueued is exactly one component and nothing
        // routed out of it. Write that down for the probes still to come.
        if nextLabel == .max {
            label.update(repeating: 0)
            nextLabel = 1
            firstValidLabel = 1
        }
        let c = nextLabel
        nextLabel &+= 1
        for k in 0..<tail {
            let packed = UInt32(bitPattern: reachQueue[k])
            label[Int(packed >> 16) * cols + Int(packed & 0xFFFF)] = c
        }
        return false
    }

    /// Can a trace inside component `c` land on `goal`? Either the goal is in
    /// it, or it is enterable from a cell that is — the goal being exempt from
    /// the occupancy test, it may well be copper itself.
    private func component(_ c: UInt32, touches goal: Int) -> Bool {
        if label[goal] == c { return true }
        let cols = grid.cols, rows = grid.rows
        let gx = goal % cols, gy = goal / cols
        for di in 0..<8 {
            let dx = Router.dx(di), dy = Router.dy(di)
            let nx = gx + dx, ny = gy + dy
            if nx < 0 || ny < 0 || nx >= cols || ny >= rows { continue }
            if label[ny * cols + nx] != c { continue }
            // The same no-corner-cutting rule, at r = 0: a weaker test than the
            // caller's own r can only accept more, and accepting more is free.
            if dx != 0 && dy != 0 {
                if grid.blocked(nx, gy, 0) || grid.blocked(gx, ny, 0) { continue }
            }
            return true
        }
        return false
    }

    // MARK: - Route

    /// Returns the centreline in grid cells, or nil if no legal corridor of
    /// keepout `r` exists inside the search budget.
    /// `heading` is the direction the line was already travelling when it
    /// reached `s` — for a path whose first cells were laid down by the caller.
    /// Without it the search starts with no history and is free to reverse on
    /// its very first step, which is exactly the wedge the rest of this guards
    /// against.
    func route(from s: SIMD2<Int32>, to e: SIMD2<Int32>, keepout r: Int,
               heading: Int = -1) -> [SIMD2<Int32>]? {
        let cols = grid.cols, rows = grid.rows
        let start = Int(s.y) * cols + Int(s.x)
        let goal = Int(e.y) * cols + Int(e.x)

        // Refuse an already-claimed start. Callers used to be trusted to check
        // this, and the one that forgot — seamFallback, on a port a through-
        // trace had just landed on — produced traces starting exactly on top
        // of another trace.
        if start != goal && grid.blocked(Int(s.x), Int(s.y), r) { return nil }

        // Nothing below can find a corridor the pre-pass says is not there.
        if !reachable(from: start, to: goal, keepout: r) { return nil }

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
        dirOf[start] = Int8(heading)
        recentDirs[start] = heading >= 0 ? (Router.noHistory << 4 | UInt32(heading)) & Router.historyMask
                                         : Router.noHistory
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
            let history = recentDirs[cur]
            let forbidden = forbiddenDirections(history)

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
                // No spike: never leave more than 90 degrees off the way the
                // line arrived, and never off the way it was going two or three
                // cells back either. A single 135 degree vertex and a pair of
                // 90s over five cells draw the same thing — a wedge with the
                // line doubling back beside itself — and the turn cost alone
                // only prices those, it does not forbid them. A wide U-turn is
                // untouched: this looks back three cells, not thirty.
                // No spike: never leave more than 90 degrees off the way the
                // line arrived, nor off the way it was going a few cells back.
                // A single 135 degree vertex and a pair of 90s over five cells
                // draw the same thing — a wedge with the line doubling back
                // beside itself — and the turn cost alone only prices those.
                if forbidden & (1 << UInt32(di)) != 0 { continue }
                var turn: Float = 0
                if cd >= 0 {
                    let dd = Router.turnSteps(di, cd)
                    turn = dd == 0 ? 0 : (dd == 1 ? Routing.turn45
                                                  : Routing.turn90 * Float(dd) * 0.5)
                }
                let step: Float = (dx != 0 && dy != 0) ? Routing.diagonalStep : 1
                let lane = lane(hug[ni])
                let ng = curG + step * lane + turn
                let known = visitStamp[ni] == gen ? gScore[ni] : .infinity
                if ng < known {
                    gScore[ni] = ng
                    visitStamp[ni] = gen
                    cameFrom[ni] = Int32(cur)
                    dirOf[ni] = Int8(di)
                    recentDirs[ni] = (history << 4 | UInt32(di)) & Router.historyMask
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
