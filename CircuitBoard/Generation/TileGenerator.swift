import CoreGraphics
import Foundation
import simd

/// Generates one tile of board, deterministically, from `(world, index)`.
///
/// One instance is reusable across tiles so the routing grid and the A* arena
/// are allocated once per worker rather than once per tile. Tiles are fully
/// independent — that independence is what lets several generate at the same
/// time, which is where the parallelism lives: the routing itself cannot be
/// parallelised, because every `claim` changes the board the next route sees.
final class TileGenerator {
    /// Mutable so a pooled generator can be reused across parallax layers,
    /// which want the same board at different levels of detail. Only the
    /// *counts* may differ: `geo` sizes the routing grid and is fixed for the
    /// life of the generator, which `retune` enforces.
    private(set) var params: BoardParams
    /// Shorthand for `params.palette` — the generator reaches for it on nearly
    /// every trace and pad it creates.
    var palette: Palette { params.palette }
    let geo: BoardGeometry
    let grid: RoutingGrid
    let router: Router
    private(set) var rng: Rng

    var data = TileData()
    /// Pin rows handed to the fan-out router, as indices into `data.pads`.
    var combs: [[Int]] = []
    var chipSeq = 0
    /// Rasterised label bitmaps, kept for the life of this generator. A
    /// generator is pooled and reused across tiles, so this warms once per
    /// worker and never needs a lock.
    var textBitmaps: [String: TextRaster.Bitmap] = [:]

    /// The cluster makers a `parts` slot can draw, in the original's order —
    /// `pick` indexes this list, so reordering it changes every board.
    enum Maker { case dip, matrix, comb, ring, fingers }
    static let makers: [Maker] = [.dip, .matrix, .comb, .ring, .fingers]

    init(params: BoardParams) {
        self.params = params
        geo = params.geometry
        grid = RoutingGrid(cols: geo.cols, rows: Board.rows)
        router = Router(grid: grid)
        rng = Rng(seed: 1)
    }

    /// Points this generator at the same board with different counts — what the
    /// inner parallax layers ask for. Rejected if the geometry differs, because
    /// the grid buffers are sized from it and allocated once.
    func retune(to new: BoardParams) -> Bool {
        guard new.geometry == geo else { return false }
        params = new
        return true
    }

    // MARK: - Entry point

    func generate(index: Int) -> TileData {
        rng = Rng(seed: Rng.hash(params.world, signed: index + Int(Seam.tileSalt)))
        data = TileData()
        combs.removeAll(keepingCapacity: true)
        chipSeq = 0
        grid.clear()
        router.hugMultiplier = params.hugMultiplier

        let width = geo.width

        // ── 0. Reference designators reserve letter-shaped keepout ─────────
        struct Label {
            var text: String
            var size: CGFloat
            var x: Float
            var y: Float
        }
        var labels: [Label] = []
        // Tile 0 keeps its marker even though the board now runs both ways.
        if index == 0 {
            labels.append(Label(text: Draw.topLabel,
                                size: CGFloat(Draw.topLabelSize),
                                x: rng.float(55, 120), y: rng.float(40, 90)))
        }
        let labelCount = rng.int(1, 2 + TileGenerator.jsRound(geo.widthFactor))
        for _ in 0..<labelCount {
            let text = rng.pick(Draw.labelPrefixes) + String(rng.int(1, 99))
            let size = CGFloat(rng.int(Draw.labelSize))
            labels.append(Label(text: text, size: size,
                                x: rng.float(50, width - 90),
                                y: rng.float(30, Board.tileHeight - 30)))
        }

        var textSites: [SIMD2<Float>] = []
        for label in labels {
            textSites += TextRaster.glyphSites(label.text, size: label.size, bold: true,
                                               x: label.x, y: label.y, tileWidth: width,
                                               cache: &textBitmaps)
        }

        // Each covered lattice point reserves a cell, so copper flows around
        // the letter shapes rather than through them.
        for s in textSites {
            let gx = TileGenerator.jsRound((s.x - geo.originX) / Board.gridSize)
            let gy = TileGenerator.jsRound((s.y - Board.originY) / Board.gridSize)
            grid.markCell(gx, gy, 1)
        }

        // ── 1. Seam ports, decided before anything is placed ────────────────
        // Their inward lanes are reserved up front, otherwise a footprint lands
        // on the boundary column and the trace that should cross the seam never
        // gets routed.
        var seamPorts = edgePorts(index).map { p -> SeamPort in
            var q = p; q.row = 0; return q
        }
        seamPorts += edgePorts(index + 1).map { p -> SeamPort in
            var q = p; q.row = Board.rows - 1; return q
        }

        // One reserved lane PER PORT, released only when that port's turn
        // comes. Releasing them all at once meant the first seam trace could
        // route straight through a later port's lane, so that port had nowhere
        // to go and its column ended up with a trace on one side of the
        // boundary and nothing on the other — a line that stops in mid-air.
        let reserved: [[Int]] = seamPorts.map { p in
            let dir = p.row == 0 ? 1 : -1
            var cells: [Int] = []
            for k in 0..<Seam.reservedDepth {
                let y = p.row + dir * k
                for dx in -1...1 {
                    let x = p.gx + dx
                    guard grid.inBounds(x, y) else { continue }
                    let i = grid.index(x, y)
                    if grid.reserve(i) {
                        cells.append(i)
                    }
                }
            }
            return cells
        }

        // ── 2. Footprints first — they define where the copper has to go ────
        // The chip goes before everything else: it needs the largest clear
        // square on the board, so it has to ask while the board is still empty.
        if rng.unit() < Double(Rates.chip) {
            if let c = clusterChip() { data.chips.append(c) }
        }
        if rng.unit() < Double(Rates.dipChip) {
            if let c = clusterDIPChip() { data.chips.append(c) }
        }
        for _ in 0..<params.partCount {
            switch rng.pick(TileGenerator.makers) {
            case .dip: clusterDIP()
            case .matrix: clusterMatrix()
            case .comb: clusterComb()
            case .ring: clusterRing()
            case .fingers: clusterFingers()
            }
        }
        // Every trace consumes two pads, so the loose-pad count has to keep up
        // with the routing demand or most of the sliders do nothing.
        makePads(TileGenerator.jsRound(Float(rng.int(Rates.loosePads)) * geo.widthFactor))
        // Rare scatter of decorative silkscreen — reserves keepout before
        // routing, so the router flows around it like any other footprint.
        if rng.unit() < Double(Rates.decor) {
            makeDecor(rng.int(Rates.decorCount))
        }
        grid.rebuildHug()

        var pool = Array(data.pads.indices)

        // ── 3. Seam traces first, each with its own lane opened just for it ─
        // The fallback only needs a via inside that lane, so a seam port
        // essentially always produces a trace and both tiles agree on the
        // column.
        for (i, port) in seamPorts.enumerated() {
            grid.release(reserved[i])
            if let t = edgeTrace(port, pool: &pool) {
                data.traces.append(t)
            } else if let t = seamFallback(port) {
                data.traces.append(t)
            }
        }

        // ── 4. Main roads — thick rails across the open board ───────────────
        for _ in 0..<params.railCount {
            if let t = padTrace(&pool, cls: .main, color: rng.pick(palette.traces)) {
                data.traces.append(t)
            }
        }

        // ── 5. Fan-out combs — routed in order so each escape hugs its
        //       neighbour ────────────────────────────────────────────────────
        for row in combs {
            let color = rng.pick(palette.traces)
            for p in row {
                let w = TraceClass.fine.width
                let r = RoutingGrid.keepout(w)
                // A rail routed earlier may already have landed on this pad.
                if data.pads[p].taken { continue }
                // Take the comb pad out of the pool first — otherwise a later
                // padTrace can pick it again and both traces converge on the
                // same pad centre.
                if let own = pool.firstIndex(of: p) { pool.remove(at: own) }
                let aPorts = padPorts(p, keepout: r)
                guard let a = aPorts.first else { continue }

                var b: SIMD2<Int32>?
                var combPath: [SIMD2<Int32>]?
                var target: Int?
                var k = 0
                while k < Routing.combProbeTries && !pool.isEmpty {
                    k += 1
                    let i = rng.int(0, pool.count - 1)
                    let candidate = pool[i]
                    if data.pads[candidate].taken || row.contains(candidate) { continue }
                    // A chip pin never escapes to another pin of the same chip.
                    if data.pads[p].chip != 0,
                       data.pads[candidate].chip == data.pads[p].chip { continue }
                    guard let hit = routeToPad(from: a, pad: candidate, keepout: r, via: {
                        router.route(from: a, to: $0, keepout: r)
                    }) else { continue }
                    b = hit.port; combPath = hit.path; target = candidate
                    pool.remove(at: i)
                    break
                }
                if b == nil {
                    guard let c = grid.freeCell(rng, keepout: r) else { continue }
                    guard let v = endVia(Int(c.x), Int(c.y)) else { continue }
                    // Through `routeToPad` like every other landing, so the
                    // via's stub is held to the same rule as a pad's.
                    guard let hit = routeToPad(from: a, pad: v, keepout: r, via: {
                        router.route(from: a, to: $0, keepout: r)
                    }) else { continue }
                    b = hit.port; combPath = hit.path; target = v
                }
                guard let end = b,
                      let path = combPath ?? router.route(from: a, to: end, keepout: r)
                else { continue }
                // The escape's own stub is drawn too. If it doubles back over
                // the route, drop this pin rather than draw the wedge.
                if stubDoublesBack(Array(path.reversed()), port: a, pad: p) { continue }
                grid.claim(path, width: w)
                data.pads[p].taken = true
                if let target { data.pads[target].taken = true }
                data.traces.append(Trace(path: path, color: color, width: w, cls: .fine,
                                         padA: p, padB: target))
            }
        }

        // ── 6. Buses — one seeded member, the rest follow it through the hug
        //       field ────────────────────────────────────────────────────────
        for _ in 0..<params.busCount {
            let color = rng.pick(palette.traces)
            let k = rng.int(Rates.busMembers)
            for _ in 0..<k {
                if let t = padTrace(&pool, cls: .bus, color: color) {
                    data.traces.append(t)
                }
            }
        }

        // ── 7. Ordinary signals, then a meander or two in whatever is left ──
        for _ in 0..<params.traceCount {
            let cls: TraceClass = rng.unit() < Double(Rates.fineOverSignal) ? .fine : .signal
            if let t = padTrace(&pool, cls: cls, color: rng.pick(palette.traces)) {
                data.traces.append(t)
            }
        }
        if rng.unit() < Double(Rates.meander) {
            if let t = makeMeander(color: rng.pick(palette.traces)) {
                data.traces.append(t)
            }
        }

        // ── 8. Numbers last, so they only take space nothing else wanted ────
        data.numbers = makeNumbers(params.numberCount)

        return data
    }

    // MARK: - Pad-to-pad routing

    func padTrace(_ pool: inout [Int], cls: TraceClass, color: SIMD3<Float>) -> Trace? {
        guard pool.count >= 2 else { return nil }
        let w = cls.width
        let r = RoutingGrid.keepout(w)
        var attempt = 0
        while attempt < Routing.padTraceTries && pool.count > 1 {
            attempt += 1
            let ia = rng.int(0, pool.count - 1)
            var ib = rng.int(0, pool.count - 1)
            if ia == ib { ib = (ib + 1) % pool.count }
            let ai = pool[ia], bi = pool[ib]
            let a = data.pads[ai], b = data.pads[bi]
            if a.taken || b.taken { continue }
            // Never wire two pins of one chip together.
            if a.chip != 0 && a.chip == b.chip { continue }
            if abs(a.gx - b.gx) + abs(a.gy - b.gy) < 7 { continue }
            // Both ends get the same treatment: a port is only right if the
            // stub it leaves behind carries on the way the route runs.
            var best: [SIMD2<Int32>]?
            for pa in padPorts(ai, keepout: r, toward: SIMD2(Int32(b.gx), Int32(b.gy))) {
                guard let hit = routeToPad(from: pa, pad: bi, keepout: r, via: {
                    router.route(from: pa, to: $0, keepout: r)
                }) else { continue }
                // Both stubs are drawn, so both have to carry on the way the
                // route runs. No port that does, no trace: there are nine more
                // pairs to try.
                if !stubDoublesBack(Array(hit.path.reversed()), port: pa, pad: ai) {
                    best = hit.path
                    break
                }
            }
            guard let path = best else { continue }
            grid.claim(path, width: w)
            data.pads[ai].taken = true
            data.pads[bi].taken = true
            pool.remove(at: max(ia, ib))
            pool.remove(at: min(ia, ib))
            return Trace(path: path, color: color, width: w, cls: cls, padA: ai, padB: bi)
        }
        return nil
    }

    // MARK: - Serpentine length-matching meander

    private func lineCells(_ a: SIMD2<Int>, _ b: SIMD2<Int>, into out: inout [SIMD2<Int32>]) {
        var x = a.x, y = a.y
        let sx = (b.x - x).signum(), sy = (b.y - y).signum()
        let n = max(abs(b.x - x), abs(b.y - y))
        for _ in 0..<n {
            x += sx; y += sy
            out.append(SIMD2(Int32(x), Int32(y)))
        }
    }

    func makeMeander(color: SIMD3<Float>) -> Trace? {
        let w = TraceClass.signal.width
        for _ in 0..<Routing.meanderTries {
            let folds = rng.int(3, 7)
            let amp = rng.int(3, 6)
            let half = rng.int(2, 3)
            let wCells = folds * half * 2
            let gx = rng.int(2, grid.cols - 3 - wCells)
            let gy = rng.int(2 + amp, Board.rows - 3 - amp)
            if gx < 2 || gy < 2 || wCells < 2 { continue }
            guard grid.boxFree(gx - 1, gy - amp - 1, gx + wCells + 1, gy + amp + 1)
            else { continue }

            var pts: [SIMD2<Int32>] = [SIMD2(Int32(gx), Int32(gy))]
            var x = gx
            var up = 1
            for _ in 0..<folds {
                lineCells(SIMD2(x, gy), SIMD2(x, gy + up * amp), into: &pts)
                lineCells(SIMD2(x, gy + up * amp), SIMD2(x + half, gy + up * amp), into: &pts)
                x += half
                lineCells(SIMD2(x, gy + up * amp), SIMD2(x, gy), into: &pts)
                lineCells(SIMD2(x, gy), SIMD2(x + half, gy), into: &pts)
                x += half
                up = -up
            }
            guard let last = pts.last else { continue }
            // A failed second via leaves the first one placed, exactly as the
            // original did — it is still a legal pad, just an unused one.
            guard let vA = endVia(Int(pts[0].x), Int(pts[0].y)),
                  let vB = endVia(Int(last.x), Int(last.y)) else { continue }
            grid.claim(pts, width: w)
            return Trace(path: pts, color: color, width: w, cls: .signal,
                         padA: vA, padB: vB)
        }
        return nil
    }

    // MARK: - Silkscreen numbers

    /// Placement is the whole point: each candidate is measured, turned into a
    /// cell footprint, tested against `occ` — which by the time this runs holds
    /// every trace corridor, pad keepout and label keepout — and then marked.
    /// So a number can never cross copper, and never another number either.
    /// Unplaceable candidates are simply dropped.
    func makeNumbers(_ n: Int) -> [NumberLabel] {
        var out: [NumberLabel] = []
        var tries = 0
        while out.count < n && tries < n * Draw.numberTries {
            tries += 1
            let kind = rng.int(0, 3)
            let text: String
            switch kind {
            case 0:
                text = String(format: "%03d", rng.int(0, 999))
            case 1:
                text = "0x" + String(rng.int(0, 255), radix: 16, uppercase: true)
            case 2:
                let whole = rng.int(0, 99)
                let frac = rng.int(0, 99)
                text = "\(whole)." + String(format: "%02d", frac)
            default:
                let bits = rng.int(4, 7)
                text = (0..<bits).map { _ in String(rng.int(0, 1)) }.joined()
            }
            let size = rng.int(Draw.numberSize)
            let vertical = rng.unit() < Double(Draw.numberVerticalChance)
            let tw = TextRaster.width(text, size: CGFloat(size))
            // Half-extents in px, plus a margin so a trace's halo never grazes
            // a digit.
            let ex = (vertical ? Float(size) : tw) / 2 + Draw.numberMargin
            let ey = (vertical ? tw : Float(size)) / 2 + Draw.numberMargin
            let rx = Int((Double(ex) / Double(Board.gridSize)).rounded(.up))
            let ry = Int((Double(ey) / Double(Board.gridSize)).rounded(.up))
            if rx * 2 + 1 >= grid.cols || ry * 2 + 1 >= Board.rows { continue }
            let gx = rng.int(rx, grid.cols - 1 - rx)
            let gy = rng.int(ry, Board.rows - 1 - ry)
            guard grid.boxFree(gx - rx, gy - ry, gx + rx, gy + ry) else { continue }
            grid.markBox(gx - rx, gy - ry, gx + rx, gy + ry)
            out.append(NumberLabel(x: geo.px(gx), y: geo.py(gy),
                                   text: text, size: size, vertical: vertical,
                                   alpha: rng.float(Draw.numberAlpha)))
        }
        return out
    }
}
