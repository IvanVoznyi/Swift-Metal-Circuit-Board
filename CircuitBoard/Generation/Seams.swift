import Foundation
import simd

/// A cross-tile seam port. Seam `k` sits between tile `k-1` and tile `k`, and
/// both neighbours derive the very same list from `hash(world, k)`, so column,
/// class and colour agree and the trace reads as one continuous line across
/// the boundary — including at negative `k`, now that the board grows upward.
struct SeamPort {
    var gx: Int
    var cls: TraceClass
    var colorIndex: Int
    var row: Int
    /// What both neighbours must agree on for this crossing's runner: a wave
    /// moving at `pulseSpeed` board units a second, standing at `pulsePhase`
    /// when the clock starts, travelling downward in world space when
    /// `pulseDown`. Each half turns these into its own clock from its own
    /// length — see `Pulse.clock(_:length:)`. Nothing else has to be agreed:
    /// the spacing of the crests is a constant, and where they fall is read off
    /// the boundary the two halves already share.
    var pulseSpeed: Float = 0
    var pulsePhase: Float = 0
    var pulseDown: Bool = true
}

extension TileGenerator {

    func edgePorts(_ k: Int) -> [SeamPort] {
        let save = rng.snapshot
        rng.snapshot = Int64(Rng.hash(params.world ^ Seam.worldSalt,
                                      signed: k + Int(Seam.indexSalt)))

        let n = max(2, rng.int(3, 4 + TileGenerator.jsRound(geo.width / 260)))
        var out: [SeamPort] = []
        out.reserveCapacity(n)
        // Ports must be spaced apart, not merely distinct: two ports a column
        // or two apart meant the first trace's keepout blocked the second, so
        // that column got a trace on one side of the seam and nothing on the
        // other. The loop stays fully deterministic, so both neighbouring
        // tiles still derive the same list.
        var i = 0
        while i < n * 8 && out.count < n {
            i += 1
            let gx = rng.int(2, grid.cols - 3)
            let u = rng.float()
            let ci = rng.int(0, palette.traces.count - 1)
            // The runner's terms, drawn here so both neighbours read the same
            // ones. Speed is pinned to the timing every other trace already
            // uses: a crossing of the reference length takes `travel` seconds.
            let travel = rng.float(Pulse.travel)
            let speed = Pulse.crossingReference / travel
            let phase = rng.float(0, Float(PCB_RUNNER_WAVELENGTH) / speed)
            let down = rng.coinFlip()
            if out.contains(where: { abs($0.gx - gx) < Seam.minColumnGap }) { continue }
            let cls: TraceClass = u < Seam.mainCut ? .main : (u < Seam.busCut ? .bus : .signal)
            out.append(SeamPort(gx: gx, cls: cls, colorIndex: ci, row: 0,
                                pulseSpeed: speed, pulsePhase: phase, pulseDown: down))
        }

        rng.snapshot = save
        return out
    }

    /// This half's share of the crossing's runner.
    ///
    /// A half lies upstream of the boundary when the wave travels toward it —
    /// downward through a port on this tile's bottom edge, or upward through one
    /// on its top edge. A seam trace is stored seam-end first, so an upstream
    /// half measures its arc *backwards* from the boundary, and that sign is all
    /// that distinguishes the two halves of one wave.
    func crossingPulse(_ port: SeamPort) -> Pulse.Crossing {
        Pulse.Crossing(speed: port.pulseSpeed,
                       phase: port.pulsePhase,
                       towardSeam: (port.row == 0) != port.pulseDown)
    }

    /// Route out of a seam port straight into the tile for one cell before A*
    /// gets a say.
    ///
    /// The two halves of a crossing are drawn by different tiles and meet on
    /// the boundary, where each contributes a stub from its own first cell to
    /// the edge — and those stubs are vertical, because a stub runs down its
    /// own column. So the only direction the halves can leave the seam without
    /// putting a kink either side of that vertical pair is straight. Letting
    /// each side pick its own angle gave a wedge; agreeing on a diagonal gave
    /// something worse, a diagonal into 8px of vertical into a diagonal, which
    /// reads as a step in the middle of the line. Straight through, both sides
    /// collinear with the stubs, is the one crossing with no vertex in it.
    ///
    /// Falls back to an unconstrained route if a cell is taken — one kinked
    /// crossing beats a port with no trace at all.
    ///
    /// Four cells, so the straight run reaches ±36 px either side of the
    /// boundary rather than ±20. It is free: the lane these cells sit in is
    /// reserved three wide and nine deep before a single footprint is placed,
    /// so forcing part of it takes nothing from anyone. Measured over 300
    /// crossings, real bends within 28 px of the boundary go 2.08 per crossing
    /// at a lead of two to 0.30 at four, with the seam trace count unmoved at
    /// 580. Six is no better and eight is worse — past that the constraint
    /// starts fighting the router and the fallback fires more often.
    func routeFromSeam(_ port: SeamPort, to target: SIMD2<Int32>,
                       keepout r: Int) -> [SIMD2<Int32>]? {
        let a = SIMD2(Int32(port.gx), Int32(port.row))
        let step: Int32 = port.row == 0 ? 1 : -1
        // The lead is a straight column, so it is fully described by its start,
        // its direction and its length — it never needed to be an array. As one
        // it cost a literal, its growth and the concatenation at the end, on
        // every pool probe of every port, which made this the heaviest
        // allocation site in generation.
        var ok = true
        for i in 1...Seam.straightLead {
            let y = a.y + step * Int32(i)
            guard grid.inBounds(Int(a.x), Int(y)), !grid.blocked(Int(a.x), Int(y), r) else {
                ok = false; break
            }
        }
        // Straight in, so the router knows the line is already travelling that
        // way and will not turn back on it.
        let heading = port.row == 0 ? 2 : 6          // (0, +1) / (0, -1)
        if ok {
            // The loop ran to completion, so the lead is exactly
            // `straightLead + 1` cells and the route starts on the last of them.
            let last = SIMD2(a.x, a.y + step * Int32(Seam.straightLead))
            if let p = router.route(from: last, to: target, keepout: r, heading: heading),
               !returnsToSeam(p, row: port.row),
               !crossesLead(p, x: a.x, from: a.y, step: step, count: Seam.straightLead) {
                var out = [SIMD2<Int32>]()
                out.reserveCapacity(Seam.straightLead + p.count)
                for i in 0..<Seam.straightLead { out.append(SIMD2(a.x, a.y + step * Int32(i))) }
                out.append(contentsOf: p)
                return out
            }
        }
        // Unconstrained fallback, but never one that turns straight back at the
        // boundary: the stub from the edge to the first cell is drawn too, and a
        // route returning to that row within a few cells puts a wedge on it.
        guard let p = router.route(from: a, to: target, keepout: r),
              !returnsToSeam(p, row: port.row) else { return nil }
        return p
    }

    /// Does the path come back to the seam row while the eye still reads it as
    /// the same place?
    private func returnsToSeam(_ path: [SIMD2<Int32>], row: Int) -> Bool {
        // OPTIMIZED: Direct index iteration avoids ArraySlice allocations
        let limit = min(path.count, Routing.stubWindow + 1)
        for i in 1..<limit {
            if Int(path[i].y) == row { return true }
        }
        return false
    }

    /// Does the route run back over the lead the trace is about to draw?
    ///
    /// The lead is real copper, but it is never claimed: it is checked free,
    /// then the router is started from its far end, so the grid the router
    /// consults has nothing in those cells. A route that comes back through
    /// them draws one line lying across itself — a loop with a tail, which is
    /// what it looks like on screen.
    ///
    /// `returnsToSeam` does not cover this. It watches the seam *row* for the
    /// first `stubWindow` cells, to catch a stub doubling back at the boundary;
    /// a route that loops away and returns twenty cells later is a different
    /// shape and passes it. Rejecting here costs nothing, because the fallback
    /// below starts at the port itself and so cannot repeat a cell at all.
    ///
    /// Only the cells that get prepended count — the `count` cells starting at
    /// `(x, from)` and stepping by `step`. The route legitimately begins on the
    /// one after them, which is why it is not included.
    private func crossesLead(_ path: [SIMD2<Int32>], x: Int32, from: Int32,
                             step: Int32, count: Int) -> Bool {
        // A step is one cell, and the lead is a straight column, so a route
        // cannot pass through it without standing on it — comparing cells is
        // enough, with no need to test segments for geometric intersection.
        // Being a column also makes it a range test rather than a search.
        guard count > 0 else { return false }
        let end = from + step * Int32(count - 1)
        let lo = min(from, end), hi = max(from, end)
        for c in path where c.x == x && c.y >= lo && c.y <= hi { return true }
        return false
    }

    /// Every seam port is its own net, ending at a pad inside this tile.
    /// Running one straight through to the opposite seam was tried and
    /// removed: the trace carried THIS port's colour and width onto a port the
    /// neighbouring tile colours from its own hash entry, so the line visibly
    /// changed colour and thickness exactly at the boundary.
    func edgeTrace(_ port: SeamPort, pool: inout [Int]) -> Trace? {
        let w = port.cls.width
        let r = RoutingGrid.keepout(w)
        let color = palette.traces[port.colorIndex]
        let a = SIMD2(Int32(port.gx), Int32(port.row))
        if grid.blocked(port.gx, port.row, r) { return nil }

        var b: SIMD2<Int32>?
        var seamPath: [SIMD2<Int32>]?
        var padB: Int?
        if !pool.isEmpty {
            for _ in 0..<Routing.poolProbeTries {
                let i = rng.int(0, pool.count - 1)
                let p = pool[i]
                if data.pads[p].taken { continue }
                guard let hit = routeToPad(from: a, pad: p, keepout: r, via: {
                    routeFromSeam(port, to: $0, keepout: r)
                }) else { continue }
                b = hit.port; seamPath = hit.path; padB = p
                
                // OPTIMIZED: O(1) removal instead of O(N) shifting
                pool.swapRemove(at: i)
                break
            }
        }
        guard let target = b else { return nil }
        guard let path = seamPath ?? routeFromSeam(port, to: target, keepout: r) else {
            if let padB { pool.append(padB) }   // give the pad back
            return nil
        }
        grid.claim(path, width: w)
        if let padB { data.pads[padB].taken = true }
        return Trace(path: path, color: color, width: w, cls: port.cls,
                     padA: nil, padB: padB, edgeA: true,
                     crossing: crossingPulse(port))
    }

    /// A seam port that fails to reach a pad would leave the trace dangling on
    /// one side of the boundary only. Drop a via just inside the tile and land
    /// on that instead, so the neighbour always has something to line up with.
    func seamFallback(_ port: SeamPort) -> Trace? {
        let w = port.cls.width
        let r = RoutingGrid.keepout(w)
        let color = palette.traces[port.colorIndex]
        let dir = port.row == 0 ? 1 : -1
        if grid.blocked(port.gx, port.row, r) { return nil }  // a through-trace took it

        for k in stride(from: Seam.fallbackDepth.upperBound,
                        through: Seam.fallbackDepth.lowerBound, by: -1) {
            let gy = port.row + dir * k
            if !grid.inBounds(port.gx, gy) { continue }
            guard grid.boxFree(port.gx - 1, gy - 1, port.gx + 1, gy + 1) else { continue }
            // Radius is drawn before the metal, matching the original's
            // object-literal evaluation order — the stream must not shift.
            let pr = rng.float(3.6, 5)
            var pad = Pad(gx: port.gx, gy: gy, type: .via,
                          metal: rng.pick(palette.metals))
            pad.pr = pr
            pad.rx = 1; pad.ry = 1
            grid.markBox(port.gx - 1, gy - 1, port.gx + 1, gy + 1)
            data.pads.append(pad)
            let padIndex = data.pads.count - 1
            grid.settleHug()

            // Same rule as everywhere else: the via's own stub is drawn, so it
            // has to carry on the way the route arrived.
            guard let hit = routeToPad(from: SIMD2(Int32(port.gx), Int32(port.row)),
                                       pad: padIndex, keepout: r, via: {
                routeFromSeam(port, to: $0, keepout: r)
            }) else { continue }
            let path = hit.path
            grid.claim(path, width: w)
            data.pads[padIndex].taken = true
            return Trace(path: path, color: color, width: w, cls: port.cls,
                         padA: nil, padB: padIndex, edgeA: true,
                         crossing: crossingPulse(port))
        }
        return nil
    }
}
