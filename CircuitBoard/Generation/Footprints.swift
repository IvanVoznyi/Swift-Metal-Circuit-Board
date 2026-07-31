import Foundation
import simd

/// Footprint placement — the vocabulary of packages seen on the reference
/// boards. Clusters test their whole footprint with `boxFree` so they never
/// land on top of something else, but only ever MARK the individual pad
/// keepouts. Marking the bounding box (as this first did) turned each
/// footprint into a solid impassable rectangle and cut the board into
/// disconnected regions.
/// The dimensions of the package vocabulary, in board units.
///
/// Hoisted out of the placement code because they are the *specification* of
/// what a board looks like, not incidental arithmetic: a via is 3.4-5 units
/// across, an SMD pad is 5-6.5 by 10-14, a connector's pins sit 4.5-5.5 apart.
/// Read together they are a parts catalogue; read scattered through `rng.float`
/// calls they were nineteen unexplained numbers.
///
/// Deliberately ranges rather than single values. Every board is generated from
/// a seed and has to look like a *different* board rather than the same one
/// nudged, so each footprint draws its own size from within the family.
enum Footprint {
    /// Pad radii by kind. Vias are the smallest hole on the board and
    /// through-hole pads the largest, which is what makes the two read as
    /// different objects at a glance.
    static let viaRadius: ClosedRange<Float> = 3.4...5
    static let throughHoleRadius: ClosedRange<Float> = 5.5...8.5
    static let chipPadRadius: ClosedRange<Float> = 3.4...4.8
    static let connectorPadRadius: ClosedRange<Float> = 5...6.5
    static let decorPadRadius: ClosedRange<Float> = 3...4.2

    /// SMD pads, which are rectangles rather than discs.
    static let smdPadWidth: ClosedRange<Float> = 5...6.5
    static let dipPadHeight: ClosedRange<Float> = 10...14
    static let sipPadHeight: ClosedRange<Float> = 9...13

    /// How far apart a connector's pins sit, per pin.
    static let pinPitch: ClosedRange<Float> = 4.5...5.5
    /// A package body is drawn slightly larger or smaller than its pad span, so
    /// a row of identical parts does not look stamped.
    static let bodyJitter: ClosedRange<Float> = 0.9...1.1

    /// How often a body is drawn filled rather than as an outline. Two
    /// different values because a chip and a connector are different objects:
    /// chips are mostly solid, connectors mostly open.
    static let chipSolidChance: Float = 0.6
    static let connectorPairChance: Float = 0.4

    /// Silkscreen decor stroke and its spacing.
    static let decorStrokeWidth: Float = 2.2
    static let decorStrokeSpacing: Float = 4
}

extension TileGenerator {

    /// `Math.round` rounds halves toward +∞; Swift's `.rounded()` rounds away
    /// from zero. Only this version reproduces the original for negative input.
    @inline(__always)
    static func jsRound(_ v: Float) -> Int { Int((v + 0.5).rounded(.down)) }

    // MARK: - Pad ports

    /// Ports sit on the four axes only, each at its own axis' keepout distance,
    /// so the un-routed stub from the port back to the pad centre runs along
    /// the pad's own row/column and never leaves the corridor the pad and
    /// `claim` reserved. Diagonal ports, or a single reach of max(rx, ry),
    /// would put the stub out over open board another trace is entitled to use.
    /// `toward` is where the trace runs to, or comes from. The port goes on
    /// that side when it can, because the stub from the port back to the pad
    /// centre is drawn as part of the trace but was never routed: put the port
    /// on the far side and that stub doubles back over the route's own last
    /// cells, two parallel runs a stroke width apart that read as one line
    /// ending in a spike. It also routes better — the escape starts pointing
    /// at where it is going instead of behind the pad.
    /// The four axis directions, as arithmetic rather than a table: index 0 is
    /// east, 1 west, 2 south, 3 north.
    @inline(__always) private static func portDX(_ k: Int) -> Int { k == 0 ? 1 : (k == 1 ? -1 : 0) }
    @inline(__always) private static func portDY(_ k: Int) -> Int { k == 2 ? 1 : (k == 3 ? -1 : 0) }

    /// Every legal port of `padIndex`, best first, handed to `body` one at a
    /// time. Returning true from `body` stops the walk.
    ///
    /// Allocation-free, which is the entire point. The array-returning version
    /// this replaced built five heap objects per call — a `dirs` literal, the
    /// rotated order, the `enumerated().sorted()` pair and the result — and it
    /// is called 171,000 times over twenty-five tiles, about a quarter of
    /// generation. Choosing between four directions should not touch the heap.
    ///
    /// The order is unchanged: rotate by `start`, then sort by score against
    /// `toward` with the rotation as the tie-break. That comparator is a total
    /// order, so selecting the maximum four times running yields exactly the
    /// permutation `sorted` did — the boards come out bit-identical.
    func forEachPadPort(_ padIndex: Int, keepout r: Int, toward: SIMD2<Int32>? = nil,
                        _ body: (SIMD2<Int32>) -> Bool) {
        let start = rng.int(0, 3)
        let p = data.pads[padIndex]

        var ux: Float = 0, uy: Float = 0, scored = false
        if let toward {
            let vx = Float(Int(toward.x) - p.gx), vy = Float(Int(toward.y) - p.gy)
            let n = (vx * vx + vy * vy).squareRoot()
            if n > 0 { ux = vx / n; uy = vy / n; scored = true }
        }
        @inline(__always) func score(_ k: Int) -> Float {
            Float(TileGenerator.portDX(k)) * ux + Float(TileGenerator.portDY(k)) * uy
        }

        // Selection over four, ties broken by position in the rotated order —
        // the same total order the comparator defined.
        var taken: UInt8 = 0
        for _ in 0..<4 {
            var bestOffset = -1
            var bestScore = -Float.greatestFiniteMagnitude
            for offset in 0..<4 where taken & (1 << UInt8(offset)) == 0 {
                let k = (offset + start) % 4
                let sc = scored ? score(k) : 0
                if bestOffset < 0 || sc > bestScore {
                    bestScore = sc
                    bestOffset = offset
                }
                if !scored { break }        // no target: the rotation is the order
            }
            guard bestOffset >= 0 else { break }
            taken |= 1 << UInt8(bestOffset)
            let k = (bestOffset + start) % 4
            let gx = p.gx + TileGenerator.portDX(k) * (p.rx + 1 + r)
            let gy = p.gy + TileGenerator.portDY(k) * (p.ry + 1 + r)
            if !grid.inBounds(gx, gy) { continue }
            if grid.blocked(gx, gy, r) { continue }
            if body(SIMD2(Int32(gx), Int32(gy))) { return }
        }
    }

    /// The best legal port, when the caller cannot try more than one.
    func padPort(_ padIndex: Int, keepout r: Int, toward: SIMD2<Int32>? = nil) -> SIMD2<Int32>? {
        var first: SIMD2<Int32>?
        forEachPadPort(padIndex, keepout: r, toward: toward) { first = $0; return true }
        return first
    }

    /// Does the un-routed stub from `port` to the pad centre run back against
    /// the way the route arrived? That is the spike: two parallel runs a
    /// stroke apart, reading as one line that ends in a point.
    func stubDoublesBack(_ path: [SIMD2<Int32>], port: SIMD2<Int32>, pad: Int,
                         fromStart: Bool = false) -> Bool {
        guard path.count >= 2 else { return false }
        let p = data.pads[pad]
        let stub = SIMD2<Float>(Float(p.gx) - Float(port.x), Float(p.gy) - Float(port.y))
        let sl = (stub.x * stub.x + stub.y * stub.y).squareRoot()
        guard sl > 0 else { return false }
        let sx = stub.x / sl, sy = stub.y / sl
        // Look back along the route as far as the eye reads the two runs as one
        // line — about five cells — not just at the final step.
        //
        // `fromStart` walks the other way for the pad a route *leaves*. The
        // callers used to hand over `Array(path.reversed())` for that, copying
        // a whole path — thirty cells on average, sometimes hundreds — to
        // inspect the eight at one end of it.
        var arc: Float = 0
        var i = fromStart ? 0 : path.count - 1
        while arc < Float(Routing.stubWindow) {
            let a: SIMD2<Int32>, b: SIMD2<Int32>
            if fromStart {
                guard i < path.count - 1 else { break }
                a = path[i]; b = path[i + 1]          // stepping away from the pad
            } else {
                guard i > 0 else { break }
                a = path[i]; b = path[i - 1]
            }
            let v = SIMD2<Float>(Float(a.x - b.x), Float(a.y - b.y))
            let l = (v.x * v.x + v.y * v.y).squareRoot()
            arc += l
            if l > 0, (v.x / l) * sx + (v.y / l) * sy < -0.7071 { return true }
            i += fromStart ? 1 : -1
        }
        return false
    }

    /// Route to a pad, preferring a port whose stub carries on the way the
    /// route came in. The first choice is right ~94% of the time and costs one
    /// route as before; only the rest pay for a second look.
    func routeToPad(from start: SIMD2<Int32>, pad: Int, keepout r: Int,
                    via: (SIMD2<Int32>) -> [SIMD2<Int32>]?) -> (path: [SIMD2<Int32>], port: SIMD2<Int32>)? {
        var hit: (path: [SIMD2<Int32>], port: SIMD2<Int32>)?
        forEachPadPort(pad, keepout: r, toward: start) { q in
            guard let path = via(q) else { return false }
            if !stubDoublesBack(path, port: q, pad: pad) { hit = (path, q); return true }
            return false
        }
        if let hit { return hit }
        // Every port would leave a stub doubling back over the route. Refuse the
        // pad rather than draw the wedge — the caller has nine more to try, and
        // a trace that does not exist is better than one that looks broken.
        return nil
    }

    func markPads(_ indices: [Int]) {
        for i in indices {
            let p = data.pads[i]
            grid.markBox(p.gx - p.rx, p.gy - p.ry, p.gx + p.rx, p.gy + p.ry)
        }
    }

    // MARK: - Loose pads

    /// The draw bag for loose pads, weighted by repetition. A literal here was
    /// a fresh heap array on every call; the contents never change.
    private static let loosePadTypes: [PadType] = [.via, .throughHole, .throughHole, .smd, .smd]

    func makePads(_ n: Int) {
        var placed = 0, tries = 0
        while placed < n && tries < n * 10 {
            tries += 1
            let gx = rng.int(2, grid.cols - 3)
            let gy = rng.int(2, Board.rows - 3)
            let type = rng.pick(TileGenerator.loosePadTypes)
            var p = Pad(gx: gx, gy: gy, type: type, metal: rng.pick(palette.metals))
            switch type {
            case .via:
                p.pr = rng.float(Footprint.viaRadius)
                p.rx = RoutingGrid.padKeepout(p.pr); p.ry = p.rx
            case .throughHole:
                p.pr = rng.float(Footprint.throughHoleRadius)
                p.rx = RoutingGrid.padKeepout(p.pr); p.ry = p.rx
            default:
                p.w = rng.float(13, 28)
                p.h = rng.float(7, 15)
                p.rx = RoutingGrid.padKeepout(p.w / 2)
                p.ry = RoutingGrid.padKeepout(p.h / 2)
            }
            if grid.boxFree(gx - p.rx, gy - p.ry, gx + p.rx, gy + p.ry) {
                grid.markBox(gx - p.rx, gy - p.ry, gx + p.rx, gy + p.ry)
                data.pads.append(p)
                placed += 1
            }
        }
    }

    /// Drop a via on a free cell so a trace always terminates on something
    /// rather than stopping in open board. (`endVia`)
    func endVia(_ gx: Int, _ gy: Int) -> Int? {
        let pr = rng.float(Footprint.chipPadRadius)
        let rr = RoutingGrid.padKeepout(pr)
        guard grid.boxFree(gx - rr, gy - rr, gx + rr, gy + rr) else { return nil }
        var pad = Pad(gx: gx, gy: gy, type: .via, metal: rng.pick(palette.metals))
        pad.pr = pr; pad.rx = rr; pad.ry = rr; pad.taken = true
        grid.markBox(gx - rr, gy - rr, gx + rr, gy + rr)
        data.pads.append(pad)
        grid.settleHug()
        return data.pads.count - 1
    }

    // MARK: - Clusters

    @discardableResult
    func clusterDIP() -> Bool {
        let n = rng.int(3, 8), pitch = 4
        let gap = rng.int(6, 10)
        let pr = rng.float(Footprint.connectorPadRadius)
        let metal = rng.pick(palette.metals)
        let h = (n - 1) * pitch
        let rr = RoutingGrid.padKeepout(pr)
        let gx = rng.int(3, grid.cols - 4 - gap)
        let gy = rng.int(3, Board.rows - 4 - h)
        guard gx >= 3, gy >= 3 else { return false }
        guard grid.boxFree(gx - rr, gy - rr, gx + gap + rr, gy + h + rr) else { return false }
        var made: [Int] = []
        for i in 0..<n {
            for s in 0..<2 {
                var p = Pad(gx: s == 1 ? gx + gap : gx, gy: gy + i * pitch,
                            type: (i == 0 && s == 0) ? .square : .throughHole,
                            metal: metal)
                p.pr = pr; p.rx = rr; p.ry = rr
                data.pads.append(p)
                made.append(data.pads.count - 1)
            }
        }
        markPads(made)
        return true
    }

    @discardableResult
    func clusterMatrix() -> Bool {
        let nx = rng.int(3, 6), ny = rng.int(3, 5), pitch = 3
        let pr = rng.float(Footprint.decorPadRadius)
        let metal = rng.pick(palette.metals)
        let rr = RoutingGrid.padKeepout(pr)
        let gx = rng.int(2, grid.cols - 3 - nx * pitch)
        let gy = rng.int(2, Board.rows - 3 - ny * pitch)
        guard gx >= 2, gy >= 2 else { return false }
        let x1 = gx + (nx - 1) * pitch, y1 = gy + (ny - 1) * pitch
        guard grid.boxFree(gx - rr, gy - rr, x1 + rr, y1 + rr) else { return false }
        var made: [Int] = []
        for j in 0..<ny {
            for i in 0..<nx {
                var p = Pad(gx: gx + i * pitch, gy: gy + j * pitch, type: .via, metal: metal)
                p.pr = pr; p.rx = rr; p.ry = rr
                data.pads.append(p)
                made.append(data.pads.count - 1)
            }
        }
        markPads(made)
        return true
    }

    @discardableResult
    func clusterRing() -> Bool {
        let pr = rng.float(11, 16)
        let rr = RoutingGrid.padKeepout(pr)
        let gx = rng.int(rr + 1, grid.cols - rr - 2)
        let gy = rng.int(rr + 1, Board.rows - rr - 2)
        guard gx >= rr + 1, gy >= rr + 1 else { return false }
        guard grid.boxFree(gx - rr, gy - rr, gx + rr, gy + rr) else { return false }
        grid.markCell(gx, gy, rr)
        var p = Pad(gx: gx, gy: gy, type: .ring, metal: rng.pick(palette.metals))
        p.pr = pr; p.rx = rr; p.ry = rr
        data.pads.append(p)
        return true
    }

    @discardableResult
    func clusterFingers() -> Bool {
        let n = rng.int(4, 8), pitch = 3
        let w = rng.float(26, 42), h = rng.float(6, 9)
        let rx = RoutingGrid.padKeepout(w / 2), ry = RoutingGrid.padKeepout(h / 2)
        let side = rng.coinFlip() ? 0 : 1
        let gx = side == 1 ? grid.cols - 2 - rx : 1 + rx
        let gy = rng.int(2, Board.rows - 3 - (n - 1) * pitch)
        guard gy >= 2 else { return false }
        let y1 = gy + (n - 1) * pitch
        guard grid.boxFree(gx - rx, gy - ry, gx + rx, y1 + ry) else { return false }
        let metal = rng.pick(palette.metals)
        var made: [Int] = []
        for i in 0..<n {
            var p = Pad(gx: gx, gy: gy + i * pitch, type: .smd, metal: metal)
            p.w = w; p.h = h; p.rx = rx; p.ry = ry
            data.pads.append(p)
            made.append(data.pads.count - 1)
        }
        markPads(made)
        return true
    }

    /// Fine-pitch SMD row — its escapes become the fan-out comb once routed
    /// in order.
    @discardableResult
    func clusterComb() -> Bool {
        let n = rng.int(5, 10), pitch = 3
        let w = rng.float(6, 9), h = rng.float(13, 20)
        let vert = rng.coinFlip()
        let rx = RoutingGrid.padKeepout((vert ? w : h) / 2)
        let ry = RoutingGrid.padKeepout((vert ? h : w) / 2)
        let gx = rng.int(2 + rx, grid.cols - 3 - rx - (vert ? (n - 1) * pitch : 0))
        let gy = rng.int(2 + ry, Board.rows - 3 - ry - (vert ? 0 : (n - 1) * pitch))
        guard gx >= 2, gy >= 2 else { return false }
        let x1 = gx + (vert ? (n - 1) * pitch : 0)
        let y1 = gy + (vert ? 0 : (n - 1) * pitch)
        guard grid.boxFree(gx - rx, gy - ry, x1 + rx, y1 + ry) else { return false }
        let metal = rng.pick(palette.metals)
        var row: [Int] = []
        for i in 0..<n {
            var p = Pad(gx: gx + (vert ? i * pitch : 0),
                        gy: gy + (vert ? 0 : i * pitch),
                        type: .smd, metal: metal)
            p.w = vert ? w : h; p.h = vert ? h : w
            p.rx = rx; p.ry = ry
            data.pads.append(p)
            row.append(data.pads.count - 1)
        }
        markPads(row)
        combs.append(row)
        return true
    }

    /// QFP / CPU package — square body with a pin row on all four sides.
    ///
    /// The body is a solid keepout, so nothing routes underneath it, and each
    /// pin is boxed in by its neighbours on both sides and by the body behind
    /// it: the only free port faces outward, which is what makes the escapes
    /// radiate the way they do on a real fan-out. The pin rows are handed to
    /// the comb router so those escapes bundle as they leave. Deliberately
    /// rare — one attempt per tile, and it needs a large clear square, so most
    /// tiles have none.
    func clusterChip() -> ChipBody? {
        let n = rng.int(4, 6), pitch = 2
        let span = (n - 1) * pitch
        // ±10% body-size variety, so no two chips are the same size. The pins
        // always sit one cell outside the scaled body, and the keepout IS the
        // body, so scaling can never make copper touch a pin.
        let bodyScale = rng.float(Footprint.bodyJitter)
        let halfSpan = Int((Float(span) / 2).rounded(.up))
        let hb = max(halfSpan + 1,
                     TileGenerator.jsRound(Float(halfSpan + 2) * bodyScale))
        let chk = hb + 4
        guard chk * 2 + 3 < grid.cols, chk * 2 + 3 < Board.rows else { return nil }

        for _ in 0..<Routing.clusterTries {
            let cx = rng.int(chk + 1, grid.cols - 2 - chk)
            let cy = rng.int(chk + 1, Board.rows - 2 - chk)
            guard grid.boxFree(cx - chk, cy - chk, cx + chk, cy + chk) else { continue }
            chipSeq += 1
            let id = chipSeq
            let metal = rng.pick(palette.metals)
            let pw = rng.float(Footprint.smdPadWidth), ph = rng.float(Footprint.dipPadHeight)
            let off = TileGenerator.jsRound(Float(span) / 2)

            func pin(_ gx: Int, _ gy: Int, vertical: Bool) -> Pad {
                var p = Pad(gx: gx, gy: gy, type: .smd, metal: metal)
                p.chip = id
                p.w = vertical ? pw : ph
                p.h = vertical ? ph : pw
                p.rx = RoutingGrid.padKeepout((vertical ? pw : ph) / 2)
                p.ry = RoutingGrid.padKeepout((vertical ? ph : pw) / 2)
                return p
            }
            // Appended top→bottom→left→right, because pool order decides which
            // pads later traces reach for. `pin` draws no randomness, so each
            // pad can be built where it is appended and the sequence is the
            // same one the four intermediate rows used to produce.
            var sides: [[Int]] = [[], [], [], []]
            for s in 0..<4 {
                sides[s].reserveCapacity(n)
                for i in 0..<n {
                    let d = -off + i * pitch
                    let p: Pad
                    switch s {
                    case 0:  p = pin(cx + d, cy - hb - 1, vertical: true)
                    case 1:  p = pin(cx + d, cy + hb + 1, vertical: true)
                    case 2:  p = pin(cx - hb - 1, cy + d, vertical: false)
                    default: p = pin(cx + hb + 1, cy + d, vertical: false)
                    }
                    data.pads.append(p)
                    sides[s].append(data.pads.count - 1)
                }
            }
            grid.markBox(cx - hb, cy - hb, cx + hb, cy + hb)  // nothing routes under the die
            for side in sides { markPads(side) }
            combs.append(contentsOf: sides)
            return ChipBody(x0: cx - hb, y0: cy - hb, x1: cx + hb, y1: cy + hb,
                            metal: metal,
                            // Two reference styles: a dark die or a hollow body.
                            solid: rng.coinFlip(),
                            // Independent 5–15% insets, so the two inner rings
                            // never coincide and successive chips differ.
                            insetSolid: rng.float(Draw.Chip.insetRange),
                            insetDash: rng.float(Draw.Chip.insetRange))
        }
        return nil
    }

    /// DIP / SOIC package — a rectangular IC with a pin row on only TWO
    /// opposite sides. A real routable chip: pins are pads handed to the comb
    /// router, body is keepout. Rotates 90° at random, and appears a little
    /// more often than the four-sided chip.
    func clusterDIPChip() -> ChipBody? {
        let n = rng.int(3, 7), pitch = 2
        let span = (n - 1) * pitch
        let shortHalf = rng.int(2, 3)
        let longHalf = Int((Float(span) / 2).rounded(.up)) + 1
        let vert = rng.coinFlip()
        let hbX = vert ? shortHalf : longHalf
        let hbY = vert ? longHalf : shortHalf
        let chk = max(hbX, hbY) + 4
        guard chk * 2 + 3 < grid.cols, chk * 2 + 3 < Board.rows else { return nil }

        for _ in 0..<Routing.clusterTries {
            let cx = rng.int(chk + 1, grid.cols - 2 - chk)
            let cy = rng.int(chk + 1, Board.rows - 2 - chk)
            guard grid.boxFree(cx - chk, cy - chk, cx + chk, cy + chk) else { continue }
            chipSeq += 1
            let id = chipSeq
            let metal = rng.pick(palette.metals)
            let pw = rng.float(Footprint.smdPadWidth), ph = rng.float(Footprint.sipPadHeight)
            let off = TileGenerator.jsRound(Float(span) / 2)

            func pin(_ gx: Int, _ gy: Int, vertical: Bool) -> Pad {
                var p = Pad(gx: gx, gy: gy, type: .smd, metal: metal)
                p.chip = id
                p.w = vertical ? pw : ph
                p.h = vertical ? ph : pw
                p.rx = RoutingGrid.padKeepout((vertical ? pw : ph) / 2)
                p.ry = RoutingGrid.padKeepout((vertical ? ph : pw) / 2)
                return p
            }
            // One side fully, then the other — pool order is load-bearing. As in
            // `clusterChip`, `pin` draws no randomness, so building each pad at
            // the point it is appended gives the identical sequence with no
            // intermediate rows.
            var sideA: [Int] = [], sideB: [Int] = []
            sideA.reserveCapacity(n); sideB.reserveCapacity(n)
            for i in 0..<n {
                let d = -off + i * pitch
                let p = vert ? pin(cx - hbX - 1, cy + d, vertical: false)
                             : pin(cx + d, cy - hbY - 1, vertical: true)
                data.pads.append(p); sideA.append(data.pads.count - 1)
            }
            for i in 0..<n {
                let d = -off + i * pitch
                let p = vert ? pin(cx + hbX + 1, cy + d, vertical: false)
                             : pin(cx + d, cy + hbY + 1, vertical: true)
                data.pads.append(p); sideB.append(data.pads.count - 1)
            }
            grid.markBox(cx - hbX, cy - hbY, cx + hbX, cy + hbY)
            markPads(sideA); markPads(sideB)
            combs.append(sideA)
            combs.append(sideB)
            return ChipBody(x0: cx - hbX, y0: cy - hbY, x1: cx + hbX, y1: cy + hbY,
                            metal: metal,
                            solid: rng.coinFlip(),
                            insetSolid: rng.float(Draw.Chip.insetRange),
                            insetDash: rng.float(Draw.Chip.insetRange))
        }
        return nil
    }

    // MARK: - Silkscreen decor

    /// Reserve a free box big enough for a component of the given px
    /// half-extents. (`decorFit`)
    private func decorFit(_ hw: Float, _ hh: Float) -> SIMD2<Float>? {
        let rx = Int((Double(hw + Board.clearance) / Double(Board.gridSize)).rounded(.up))
        let ry = Int((Double(hh + Board.clearance) / Double(Board.gridSize)).rounded(.up))
        guard rx * 2 + 3 < grid.cols, ry * 2 + 3 < Board.rows else { return nil }
        for _ in 0..<Routing.decorTries {
            let gx = rng.int(rx + 1, grid.cols - 2 - rx)
            let gy = rng.int(ry + 1, Board.rows - 2 - ry)
            guard grid.boxFree(gx - rx, gy - ry, gx + rx, gy + ry) else { continue }
            grid.markBox(gx - rx, gy - ry, gx + rx, gy + ry)
            return SIMD2(geo.px(gx), geo.py(gy))
        }
        return nil
    }

    /// The small decorative packages that dot the reference art around the
    /// chip. Purely cosmetic — nothing routes to them — but they reserve grid
    /// keepout exactly like a pad, placed before routing so the router flows
    /// around them.
    /// Likewise the decor draw bag — `.block` twice because it is the one that
    /// should recur.
    private static let decorKinds: [DecorItem.Kind] = [.soic, .block, .block, .dots, .frame,
                                                       .bars, .tinySquare, .diagonal, .dip]

    func makeDecor(_ n: Int) {
        var placed = 0, tries = 0
        while placed < n && tries < n * 10 {
            tries += 1
            let kind = rng.pick(TileGenerator.decorKinds)
            let metal = rng.pick(palette.metals)
            // Per-component size boost of up to +15%. Reserved at the scaled
            // size and applied at draw time, so a bigger part still keeps its
            // full keepout and never touches a trace or pad.
            let sc = rng.float(Draw.Decor.scaleRange)
            var item: DecorItem
            var at: SIMD2<Float>?

            switch kind {
            case .soic:
                let pn = rng.int(3, 5)
                let leg: Float = 3
                let bw = Float(pn) * rng.float(Footprint.pinPitch)
                let bh = rng.float(9, 13)
                let vert = rng.coinFlip()
                at = decorFit((vert ? bh / 2 : bw / 2 + leg) * sc,
                              (vert ? bw / 2 + leg : bh / 2) * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .soic, x: a.x, y: a.y, metal: metal)
                item.vertical = vert
                item.bw = bw; item.bh = bh; item.pinCount = pn; item.leg = leg
                item.solid = rng.chance(Footprint.chipSolidChance)

            case .block:
                let bw = rng.float(9, 17), bh = rng.float(5, 8)
                let pair = rng.chance(Footprint.connectorPairChance)
                at = decorFit(bw / 2 * sc, (pair ? bh + 2 : bh / 2) * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .block, x: a.x, y: a.y, metal: metal)
                item.bw = bw; item.bh = bh; item.paired = pair

            case .dots:
                let nx = rng.int(2, 4), ny = rng.int(2, 4)
                let sp: Float = 5
                at = decorFit((Float(nx - 1) * sp / 2 + 2) * sc,
                              (Float(ny - 1) * sp / 2 + 2) * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .dots, x: a.x, y: a.y, metal: metal)
                item.nx = nx; item.ny = ny; item.spacing = sp

            case .frame:
                let s = rng.float(10, 16)
                at = decorFit(s / 2 * sc, s / 2 * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .frame, x: a.x, y: a.y, metal: metal)
                item.size = s

            case .bars:
                let k = rng.int(2, 4)
                let barWidth = Footprint.decorStrokeWidth
                let barSpacing = Footprint.decorStrokeSpacing
                let h = rng.float(8, 13)
                at = decorFit((Float(k - 1) * barSpacing / 2 + barWidth) * sc,
                              h / 2 * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .bars, x: a.x, y: a.y, metal: metal)
                item.barCount = k
                item.barWidth = barWidth
                item.spacing = barSpacing
                item.height = h

            case .tinySquare:
                let s = rng.float(5, 9)
                at = decorFit(s / 2 * sc, s / 2 * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .tinySquare, x: a.x, y: a.y, metal: metal)
                item.size = s

            case .diagonal:
                let bw = rng.float(11, 16), bh = rng.float(4, 6)
                let rad = (bw * bw + bh * bh).squareRoot() / 2
                at = decorFit(rad * sc, rad * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .diagonal, x: a.x, y: a.y, metal: metal)
                item.bw = bw; item.bh = bh; item.rotation = .pi / 4

            case .dip:
                let m = rng.int(3, 5)
                let sp: Float = 4
                let w = Float(m - 1) * sp
                at = decorFit((w / 2 + 3) * sc, 6 * sc)
                guard let a = at else { continue }
                item = DecorItem(kind: .dip, x: a.x, y: a.y, metal: metal)
                item.nx = m; item.spacing = sp
            }

            item.scale = sc
            data.decor.append(item)
            placed += 1
        }
    }
}
