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

    /// A monotone bucket queue, in place of the binary heap this used to be.
    ///
    /// A* pops `f` in roughly non-decreasing order, so the open list never
    /// needed a total ordering — only the cheapest non-empty bucket. Push and
    /// pop become a linked-list splice rather than a sift, which is what the
    /// traffic asked for: measured 1.58 pushes per pop, so more than a third of
    /// the heap's work was on entries that were superseded before they came
    /// back out. 15.4 ms a tile to 12.5.
    ///
    /// Cells inside one bucket come back in an arbitrary order. That is the
    /// price, and it is not free: paths come out slightly looser — 4.63 bends a
    /// trace to 4.84, mean copper 356 px to 359 — though it also lands *more*
    /// traces, 2471 to 2499, and breaks none of the invariants (no spike, no
    /// clearance violation). A narrower bucket does not buy the quality back:
    /// what differs from the heap is the order within a bucket, not the
    /// resolution between them. Measured at 0.0625 it is no better and slower.
    private static let bucketCount = 2048
    private static let bucketWidth: Float = 0.25
    private let bucketHead: UnsafeMutablePointer<Int32>
    private let entryCell: UnsafeMutablePointer<Int32>
    private let entryNext: UnsafeMutablePointer<Int32>
    private var entryCount = 0
    private var scanBucket = 0
    private var queued = 0

    /// How many entries the queue arena holds: a cell can be pushed once per
    /// incoming direction, and never more than that.
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
        heapCapacity = count * 8 + 16
        bucketHead = .allocate(capacity: Router.bucketCount)
        entryCell = .allocate(capacity: heapCapacity)
        entryNext = .allocate(capacity: heapCapacity)
        bucketHead.initialize(repeating: -1, count: Router.bucketCount)
        entryCell.initialize(repeating: 0, count: heapCapacity)
        entryNext.initialize(repeating: -1, count: heapCapacity)
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
        bucketHead.deallocate()
        entryCell.deallocate()
        entryNext.deallocate()
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

    // -------------------------------------------------------------------------------------------------------------------------
    //  Direction lookup tables used by dx(i) and dy(i)
    //
    //      Index      Direction      dx(i)      dy(i)
    //      ------------------------------------------------
    //         5            NW          -1         -1
    //         6             N           0         -1
    //         7            NE          +1         -1
    //         4             W          -1          0
    //         0             E          +1          0
    //         3            SW          -1         +1
    //         2             S           0         +1
    //         1            SE          +1         +1
    //
    //                 dx = -1            dx = 0             dx = +1
    //              ┌──────────────────┬──────────────────┬──────────────────┐
    //      dy = -1 │  Index 5 (NW)    │  Index 6 (N)     │  Index 7 (NE)    │
    //              │  dx = -1         │  dx =  0         │  dx = +1         │
    //              │  dy = -1         │  dy = -1         │  dy = -1         │
    //              ├──────────────────┼──────────────────┼──────────────────┤
    //      dy =  0 │  Index 4 (W)     │      CELL        │  Index 0 (E)     │
    //              │  dx = -1         │                  │  dx = +1         │
    //              │  dy =  0         │                  │  dy =  0         │
    //              ├──────────────────┼──────────────────┼──────────────────┤
    //      dy = +1 │  Index 3 (SW)    │  Index 2 (S)     │  Index 1 (SE)    │
    //              │  dx = -1         │  dx =  0         │  dx = +1         │
    //              │  dy = +1         │  dy = +1         │  dy = +1         │
    //              └──────────────────┴──────────────────┴──────────────────┘
    //
    //  Direction index order (clockwise):
    //
    //                 5 ───── 6 ───── 7
    //                 │               │
    //                 │               │
    //                 4     CELL      0
    //                 │               │
    //                 │               │
    //                 3 ───── 2 ───── 1
    //
    //  Therefore:
    //
    //      dx = ( +1, +1,  0, -1, -1, -1,  0, +1 )
    //      dy = (  0, +1, +1, +1,  0, -1, -1, -1 )
    // -------------------------------------------------------------------------------------------------------------------------
    
    @inline(__always)
    static func dx(_ i: Int) -> Int {
        let table = (1, 1, 0, -1, -1, -1, 0, 1)
        return withUnsafeBytes(of: table) { $0.bindMemory(to: Int.self)[i & 7] }
    }
    
    @inline(__always)
    static func dy(_ i: Int) -> Int {
        let table = (0, 1, 1, 1, 0, -1, -1, -1)
        return withUnsafeBytes(of: table) { $0.bindMemory(to: Int.self)[i & 7] }
    }

    // MARK: - Open list
    // Ported comparison-for-comparison so tie-breaking, and therefore the
    // chosen path among equal-cost routes, is reproducible.

    @inline(__always)
    private func enqueue(_ cost: Float, _ nodeCellIndex: Int32) {
        guard entryCount < heapCapacity else { return }
        var b = Int(cost / Router.bucketWidth)
        // Never behind the scan, and never more than one lap ahead of it.
        if b < scanBucket { b = scanBucket }
        if b >= scanBucket + Router.bucketCount { b = scanBucket + Router.bucketCount - 1 }
        let slot = b & (Router.bucketCount - 1)
        let e = entryCount
        entryCell[e] = nodeCellIndex
        entryNext[e] = bucketHead[slot]
        bucketHead[slot] = Int32(e)
        entryCount += 1
        queued += 1
    }

    @inline(__always)
    private func dequeue() -> Int32 {
        while queued > 0 {
            let slot = scanBucket & (Router.bucketCount - 1)
            let e = bucketHead[slot]
            if e >= 0 {
                bucketHead[slot] = entryNext[Int(e)]
                queued -= 1
                return entryCell[Int(e)]
            }
            scanBucket += 1
        }
        return -1
    }

    static let historyMask: UInt32 = (1 << UInt32(4 * Routing.spikeWindow)) - 1
    /// Every nibble a sentinel: a path with nothing behind it yet.
    static let noHistory: UInt32 = Router.historyMask

    /// Precomputed bitmasks representing the "rear blindspot" (135°, 180°, and 225°
    /// reverse angles) for each of the 8 compass directions.
    ///
    /// Each bit in the UInt32 mask corresponds to one of the 8 grid directions (0 to 7).
    /// When a bit is set to 1, that direction is blocked, preventing the pathfinder
    /// from making sharp backtracking or U-turn movements relative to its current heading.
    // For each current heading d, the three directions behind it
    // (225°, 180°, and 135° relative to the heading) are marked as
    // "opposed" by setting their bits to 1.
    //
    //  d = 0 (E)  -> W,  NW, N   (bits 4,5,6) -> 0x70
    //  d = 1 (SE) -> NW, N,  NE  (bits 5,6,7) -> 0xE0
    //  d = 2 (S)  -> N,  NE, E   (bits 6,7,0) -> 0xC1
    //  d = 3 (SW) -> NE, E,  SE  (bits 7,0,1) -> 0x83
    //  d = 4 (W)  -> E,  SE, S   (bits 0,1,2) -> 0x07
    //  d = 5 (NW) -> SE, S,  SW  (bits 1,2,3) -> 0x0E
    //  d = 6 (N)  -> S,  SW, W   (bits 2,3,4) -> 0x1C
    //  d = 7 (NE) -> SW, W,  NW  (bits 3,4,5) -> 0x38
    //
    // -----------------------------------------------------------------------------------------------
    // [d] = Current heading
    // [X] = Opposed direction (bit = 1)
    //  .  = Allowed direction (bit = 0)
    // -----------------------------------------------------------------------------------------------
    //                                 Direction index mapping used throughout the router:
    //
    //                                                dx=-1           dx=0           dx=+1
    //                                            ┌────────────┬────────────┬────────────┐
    //                                dy = -1     │ NW (5)     │ N  (6)     │ NE (7)     │
    //                                            ├────────────┼────────────┼────────────┤
    //                                dy =  0     │ W  (4)     │  (CELL)    │ E  (0)     │
    //                                            ├────────────┼────────────┼────────────┤
    //                                dy = +1     │ SW (3)     │ S  (2)     │ SE (1)     │
    //                                            └────────────┴────────────┴────────────┘
    // ┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
    // │         d = 0 (E)            │        d = 1 (SE)            │         d = 2 (S)            │        d = 3 (SW)            │
    // │                              │                              │                              │                              │
    // │ ┌────────┬────────┬────────┐ │ ┌────────┬────────┬────────┐ │ ┌────────┬────────┬────────┐ │ ┌────────┬────────┬────────┐ │
    // │ │  [X]   │   .    │   .    │ │ │  [X]   │  [X]   │   .    │ │ │  [X]   │  [X]   │  [X]   │ │ │   .    │  [X]   │  [X]   │ │
    // │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │
    // │ │  [X]   │ (Cell) │   [d]  │ │ │  [X]   │ (Cell) │   .    │ │ │   .    │ (Cell) │   .    │ │ │   .    │ (Cell) │  [X]   │ │
    // │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │
    // │ │  [X]   │   .    │   .    │ │ │   .    │   .    │   [d]  │ │ │   .    │  [d]   │   .    │ │ │  [d]   │   .    │   .    │ │
    // │ └────────┴────────┴────────┘ │ └────────┴────────┴────────┘ │ └────────┴────────┴────────┘ │ └────────┴────────┴────────┘ │
    // │ Bits: 3,4,5                  │ Bits: 4,5,6                  │ Bits: 5,6,7                  │ Bits: 6,7,0                  │
    // | Byte : [ 0 0 1 1 1 0 0 0 ]   | Byte : [ 0 1 1 1 0 0 0 0 ]   | Byte : [ 1 1 1 0 0 0 0 0 ]   | Byte : [ 1 1 0 0 0 0 0 1 ]   |
    // |          │ │ │ │ │ │ │ │     |          │ │ │ │ │ │ │ │     |          │ │ │ │ │ │ │ │     |          │ │ │ │ │ │ │ │     |
    // |          7 6 5 4 3 2 1 0     |          7 6 5 4 3 2 1 0     |          7 6 5 4 3 2 1 0     |          7 6 5 4 3 2 1 0     |
    // │ Mask: 0x38                   │ Mask: 0x70                   │ Mask: 0xE0                   │ Mask: 0xC1                   │
    // ├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
    // │         d = 4 (W)            │        d = 5 (NW)            │         d = 6 (N)            │        d = 7 (NE)            │
    // │                              │                              │                              │                              │
    // │ ┌────────┬────────┬────────┐ │ ┌────────┬────────┬────────┐ │ ┌────────┬────────┬────────┐ │ ┌────────┬────────┬────────┐ │
    // │ │   .    │   .    │  [X]   │ │ │  [d]   │   .    │   .    │ │ │   .    │  [d]   │   .    │ │ │   .    │   .    │  [d]   │ │
    // │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │
    // │ │  [d]   │ (Cell) │  [X]   │ │ │   .    │ (Cell) │  [X]   │ │ │   .    │ (Cell) │   .    │ │ │  [X]   │ (Cell) │   .    │ │
    // │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │ ├────────┼────────┼────────┤ │
    // │ │   .    │   .    │  [X]   │ │ │   .    │  [X]   │  [X]   │ │ │  [X]   │  [X]   │  [X]   │ │ │  [X]   │  [X]   │   .    │ │
    // │ └────────┴────────┴────────┘ │ └────────┴────────┴────────┘ │ └────────┴────────┴────────┘ │ └────────┴────────┴────────┘ │
    // │ Bits: 7,0,1                  │ Bits: 0,1,2                  │ Bits: 1,2,3                  │ Bits: 2,3,4                  │
    // | Byte : [ 1 0 0 0 0 0 1 1 ]   | Byte : [ 0 0 0 0 0 1 1 1 ]   | Byte : [ 0 0 0 0 1 1 1 0 ]   | Byte : [ 0 0 0 1 1 1 0 0 ]   |
    // |          │ │ │ │ │ │ │ │     |          │ │ │ │ │ │ │ │     |          │ │ │ │ │ │ │ │     |          │ │ │ │ │ │ │ │     |
    // |          7 6 5 4 3 2 1 0     |          7 6 5 4 3 2 1 0     |          7 6 5 4 3 2 1 0     |          7 6 5 4 3 2 1 0     |
    // │ Mask: 0x83                   │ Mask: 0x07                   │ Mask: 0x0E                   │ Mask: 0x1C                   │
    // └──────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
    //
    // A set bit (1) means that direction is considered "opposed"
    // to the current heading and may be skipped by the router.
    //
    static let opposed: [UInt32] = [
        0x38, // E
        0x70, // SE
        0xE0, // S
        0xC1, // SW
        0x83, // W
        0x07, // NW
        0x0E, // N
        0x1C  // NE
    ]

    // --- The Octile Turn Step Distance Calculator (Shortest Angular Delta) ---
    // Calculates the minimum discrete turn steps required to transition between two 8-way directional indices (0 through 7).
    // Replaces expensive floating-point trigonometric calculations (like atan2 or acos) with ultra-fast scalar modular
    // arithmetic to compute directional turn penalties instantly during path finding.
    //
    // 1. Angular Index Representation:
    //    - The 8 movement directions are mapped sequentially from 0 through 7 around the circle in 45-degree steps.
    //    - One turn step equals a 45-degree shift (e.g., direct move to gentle turn).
    //    - Two turn steps equal 90 degrees, three equal 135 degrees, and four equal a full 180-degree reversal.
    //
    // 2. Circular Wrap-Around Correction:
    //    - Simple absolute subtraction `abs(a - b)` fails across the 0/7 index boundary (e.g., from North-East [7] to East [0]).
    //    - If the raw delta exceeds 4 steps (180 degrees), taking the opposite circular arc `8 - raw` yields the true
    //      shortest turn distance (e.g., a raw difference of 7 steps across the boundary simplifies to 1 step).
    //
    // 3. Performance Inlining (@inline(__always)):
    //    - Marking this function for mandatory inlining allows the Swift compiler to insert the simple arithmetic logic
    //      directly into hot pathfinding loops, eliminating function call overhead during millions of neighbor evaluations.
    //
    // ------------------------------------------------------------------------------------
    //
    //    [ DIRECTION KEY ]
    //    0 = North (0°)       2 = East (90°)       4 = South (180°)     6 = West (270°)
    //    1 = Northeast (45°)  3 = Southeast (135°) 5 = Southwest (225°) 7 = Northwest (315°)
    //
    //    --------------------------------------------------------------------------------
    //    [ TURN STEPS LOOKUP TABLE ]
    //    Rows = Starting Direction (a) | Columns = Target Direction (b)
    //    --------------------------------------------------------------------------------
    //           |  0(N)  1(NE)  2(E)  3(SE)  4(S)  5(SW)  6(W)  7(NW)
    //    ------ + -------------------------------------------------
    //      0(N) |   0      1     2     3      4     3     2     1
    //      1(NE)|   1      0     1     2      3     4     3     2
    //      2(E) |   2      1     0     1      2     3     4     3
    //      3(SE)|   3      2     1     0      1     2     3     4
    //      4(S) |   4      3     2     1      0     1     2     3
    //      5(SW)|   3      4     3     2      1     0     1     2
    //      6(W) |   2      3     4     3      2     1     0     1
    //      7(NW)|   1      2     3     4      3     2     1     0
    //
    //    --------------------------------------------------------------------------------
    //    [ WHAT THE VALUES MEAN & MOVEMENT RULES ]
    //    --------------------------------------------------------------------------------
    //    Value | Angle Diff | Description           | Blocked Status (If applicable)
    //    ----- + ---------- + --------------------- + ----------------------------
    //      0   |    0°      | No turn / Straight    | Allowed
    //      1   |   45°      | Slight turn / Diagonal| Allowed
    //      2   |   90°      | Perpendicular turn    | Allowed
    //      3   |  135°      | Sharp turn            | BLOCKED (Also covers 225° wrap)
    //      4   |  180°      | U-turn / Opposite     | BLOCKED
    //    --------------------------------------------------------------------------------
    //    Turns from 0° to 90° are fully allowed
    @inline(__always)
    static func turnSteps(_ originDirectionIndex: Int, _ targetDirectionIndex: Int) -> Int {
        let rawIndexDifference = abs(originDirectionIndex - targetDirectionIndex)
        return rawIndexDifference > 4 ? 8 - rawIndexDifference : rawIndexDifference
    }
    
    // --- The Directional History Bitmask Filter (Spike & Reversal Suppression Engine) ---
    // Scans the recent directional history of a path node to extract a combined bitmask of all forbidden directions.
    // By evaluating recent steps stored in a nibble-packed history variable, this method blocks path movements
    // that would cause sharp back-turns, U-turns, or tight zigzagging loops within a specified step window.
    //
    // 1. Nibble-Packed History Unrolling:
    //    - Reads directional history from a 32-bit bitmask (`history`) where each 4-bit segment (nibble) stores an 8-way directional index (0 through 7).
    //    - Unrolls the history step-by-step using cheap bitwise shifts (`h >>= 4`) and bitmasks (`h & 0xF`), inspecting up to `Routing.spikeWindow` previous steps.
    //
    // 2. Sentinel Check & Mask Accumulation:
    //    - Compares each extracted direction index against the sentinel value `0xF` (which flags an uninitialized or empty history slot).
    //    - For every valid directional step found, looks up its opposed direction bitmask in `Router.opposedDirectionMasks` (covering 135-degree to 180-degree back-turns)
    //      and merges it into the cumulative `forbiddenMask` via bitwise OR (`|=`).
    //
    // 3. Early Turn Pruning in Pathfinding Loops:
    //    - Returns a single 32-bit bitmask where each set bit represents an illegal directional choice for the candidate step.
    //    - Enables candidate neighbor steps to validate directional legality with a single bitwise test (`(forbiddenMask & (1 << candidateDir)) != 0`),
    //      eliminating costly branching logic inside the inner pathfinding loop.
    // -------------------------------------------------------------------------------------------------------------------------
    //  spikeWindow = 3                                                   Direction index order (clockwise):
    //                                 ┌────────┬────────┬────────┐               5 ───── 6 ───── 7
    //                                 │   NW   │   N    │   NE   │               │               │
    //                                 ├────────┼────────┼────────┤               │               │
    //                                 │   W    │ (Cell) │   E    │               4     CELL      0
    //                                 ├────────┼────────┼────────┤               │               │
    //                                 │   SW   │   S    │   SE   │               │               │
    //                                 └────────┴────────┴────────┘               3 ───── 2 ───── 1
    // -------------------------------------- + -------------------------------------- + -------------------------------------- +
    //            recentDirs[cur]             |            recentDirs[cur]             |            recentDirs[cur]             |
    //    +------+------+------+------+       |    +------+------+------+------+       |    +------+------+------+------+       |
    //    |  E   |  SE  |  S   | 0xF  |       |    |  E   |  SE  |  S   | 0xF  |       |    |  N   |  SE  |  S   | 0xF  |       |
    //    +------+------+------+------+       |    +------+------+------+------+       |    +------+------+------+------+       |
    //       d0     d1     d2   empty         |       d0     d1     d2   empty         |       d0     d1     d2   empty         |
    //       ^                                |              ^                         |                     ^                  |
    //       │                                |              │                         |                     │                  |
    //       └──────────────┐                 |              └──────┐                  |                     │                  |
    //                      │                 |                     │                  |                     │                  |
    //                      ▼                 |                     ▼                  |                     ▼                  |
    //            heading = h & 0xF           |           heading = h & 0xF            |           heading = h & 0xF            |
    // -------------------------------------- + -------------------------------------- + -------------------------------------- +
    //            Heading = E                 |           Heading = SE                 |           Heading = S                  |
    //                                        |                                        |                                        |
    //     ┌────────┬────────┬────────┐       |     ┌────────┬────────┬────────┐       |    ┌────────┬────────┬────────┐        |
    //     │  [X]   │   .    │   .    │       |     │  [X]   │  [X]   │   .    │       |    │  [X]   │  [X]   │  [X]   │        |
    //     ├────────┼────────┼────────┤       |     ├────────┼────────┼────────┤       |    ├────────┼────────┼────────┤        |
    //     │  [X]   │ (Cell) │  [d]   │       |     │  [X]   │ (Cell) │   .    │       |    │   .    │ (Cell) │  .     │        |
    //     ├────────┼────────┼────────┤       |     ├────────┼────────┼────────┤       |    ├────────┼────────┼────────┤        |
    //     │  [X]   │   .    │   .    │       |     │   .    │   .    │  [d]   │       |    │   .    │   [d]  │   .    │        |
    //     └────────┴────────┴────────┘       |     └────────┴────────┴────────┘       |    └────────┴────────┴────────┘        |
    //                                        |                                        |                                        |
    //     Mask : 0 0 1 1 1 0 0 0  (0x38)     |     Byte : 0 1 1 1 0 0 0 0  (0x70)     |    Byte : 1 1 1 0 0 0 0 0  (0xE0)      |
    //     Bits : 7 6 5 4 3 2 1 0             |     Bits : 7 6 5 4 3 2 1 0             |    Bits : 7 6 5 4 3 2 1 0              |
    //                                        |                                        |                                        |
    // -------------------------------------- + -------------------------------------- + -------------------------------------- +
    //                                        |                Result                  |                                        |
    // -------------------------------------- + -------------------------------------- + -------------------------------------- +
    //              Bitwise OR                |    ┌────────┬────────┬────────┐        |  Mask :      1  1  1  1  1  0  0  0    |
    //                                        |    │  [X]   │  [X]   │  [X]   │        |              │  │  │  │  │  │  │  │    |
    //         00111000   (0x38)              |    ├────────┼────────┼────────┤        |  Bits:       7  6  5  4  3  2  1  0    |
    //      OR 01110000   (0x70)              |    │  [X]   │ (Cell) │   .    │        |  Direction: NE  N NW  W SW  S SE  E    |
    //      OR 11100000   (0xE0)              |    ├────────┼────────┼────────┤        |                                        |
    //      --------------------              |    │  [X]   │   .    │   .    │        |  Allowed:    E, SE, S                  |
    //         11111000   (0xF8)              |    └────────┴────────┴────────┘        |  Blocked:    NE, N, NW, W, SW          |
    //                                        |                                        |                                        |
    // -------------------------------------- + -------------------------------------- + -------------------------------------- +
    @inline(__always)
    private func forbiddenDirectionsMask(_ directionalHistory: UInt32) -> UInt32 {
        var shiftableHistory = directionalHistory
        var cumulativeForbiddenMask: UInt32 = 0
        
        // Iterate through nibbles across the active spike window
        for _ in 0..<Routing.spikeWindow {
            let directionIndex = shiftableHistory & 0xF
            
            // Skip unset/sentinel history nibbles (0xF)
            if directionIndex != 0xF {
                cumulativeForbiddenMask |= Router.opposed[Int(directionIndex)]
            }
            
            // Shift to inspect the previous 4-bit directional step
            shiftableHistory >>= 4
        }
        
        return cumulativeForbiddenMask
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
        bucketHead.update(repeating: -1, count: Router.bucketCount)
        entryCount = 0
        scanBucket = 0
        queued = 0

        let hw = hugMultiplier
        let ex = Float(e.x), ey = Float(e.y)
        @inline(__always) func heuristic(_ x: Int, _ y: Int) -> Float {
            let dx = abs(Float(x) - ex), dy = abs(Float(y) - ey)
            return (max(dx, dy) + Routing.diagonalExtra * min(dx, dy))
                * hw * Routing.heuristicWeight
        }

        gScore[start] = 0
        visitStamp[start] = gen
        cameFrom[start] = -1
        dirOf[start] = Int8(heading)
        recentDirs[start] = heading >= 0 ? (Router.noHistory << 4 | UInt32(heading)) & Router.historyMask
                                         : Router.noHistory
        enqueue(heuristic(Int(s.x), Int(s.y)), Int32(start))

        var pops = 0
        var found = false
        let hug = grid.hug

        while queued > 0 {
            let cur = Int(dequeue())
            if closedStamp[cur] == gen { continue }
            closedStamp[cur] = gen
            if cur == goal { found = true; break }
            pops += 1
            if pops > Routing.budget { break }

            let cx = cur % cols, cy = cur / cols
            let cd = Int(dirOf[cur])
            let curG = gScore[cur]
            let history = recentDirs[cur]
            let forbidden = forbiddenDirectionsMask(history)

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
                    enqueue(ng + heuristic(nx, ny), Int32(ni))
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
