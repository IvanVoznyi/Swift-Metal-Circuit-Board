import CoreGraphics
import Foundation
import simd

private struct Label {
    var text: String
    var size: CGFloat
    var x: Float
    var y: Float
}
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
    private var labelBuffer: [Label] = []
    private var textSiteBuffer: [SIMD2<Float>] = []
    private var poolBuffer: [Int] = []
    var data = TileData()
    /// Pin rows handed to the fan-out router, as indices into `data.pads`.
    var combs: [[Int]] = []
    var chipSeq = 0
    /// Rasterised label bitmaps, kept for the life of this generator. A
    /// generator is pooled and reused across tiles, so this warms once per
    /// worker and never needs a lock.
    var textBitmaps: [TextRaster.CacheKey: TextRaster.Bitmap] = [:]
    
    
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
        labelBuffer.removeAll(keepingCapacity: true)
        
        // Tile 0 keeps its marker even though the board now runs both ways.
        if index == 0 {
            labelBuffer.append(Label(text: Draw.topLabel,
                                     size: CGFloat(Draw.topLabelSize),
                                     x: rng.float(55, 120), y: rng.float(40, 90)))
        }
        let labelCount = rng.int(1, 2 + TileGenerator.jsRound(geo.widthFactor))
        for _ in 0..<labelCount {
            let text = rng.pick(Draw.labelPrefixes) + String(rng.int(1, 99))
            let size = CGFloat(rng.int(Draw.labelSize))
            labelBuffer.append(Label(text: text, size: size,
                                     x: rng.float(50, width - 90),
                                     y: rng.float(30, Board.tileHeight - 30)))
        }
        
        textSiteBuffer.removeAll(keepingCapacity: true)
        for label in labelBuffer {
            // NOTE: this still calls into TextRaster.glyphSites, which today
            // allocates and returns its own [SIMD2<Float>] and then gets
            // concatenated on with +=. That's a second alloc+copy per label on
            // top of textSiteBuffer's own growth. To make this truly zero-alloc,
            // give TextRaster an inout-appending overload:
            //
            //   static func glyphSites(_ text: String, size: CGFloat, bold: Bool,
            //                          x: Float, y: Float, tileWidth: Float,
            //                          cache: inout [...],
            //                          into sites: inout [SIMD2<Float>])
            //
            // and have it append directly into textSiteBuffer instead of
            // returning a fresh array. Once that exists, replace the two lines
            // below with a single call:
            //
            //   TextRaster.glyphSites(label.text, size: label.size, bold: true,
            //                         x: label.x, y: label.y, tileWidth: width,
            //                         cache: &textBitmaps, into: &textSiteBuffer)
            let sites = TextRaster.glyphSites(label.text, size: label.size, bold: true,
                                              x: label.x, y: label.y, tileWidth: width,
                                              cache: &textBitmaps)
            textSiteBuffer.append(contentsOf: sites)
        }
        
        // Each covered lattice point reserves a cell, so copper flows around
        // the letter shapes rather than through them.
        for s in textSiteBuffer {
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
        
        poolBuffer.removeAll(keepingCapacity: true)
        poolBuffer.append(contentsOf: data.pads.indices)
        var pool = poolBuffer // still a value-type copy-on-write handle;
        // see note below on why this line stays
        
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
                guard let a = padPort(p, keepout: r) else { continue }
                
                var b: SIMD2<Int32>?
                var combPath: [SIMD2<Int32>]?
                var target: Int?
                var k = 0
                while k < Routing.combProbeTries && !pool.isEmpty {
                    k += 1
                    let i = rng.int(0, pool.count - 1)
                    let candidate = pool[i]
                    
                    // 1. Lazy cleanup: If we randomly hit a pad that is already taken,
                    // instantly remove it from the pool so we never pick it again.
                    if data.pads[candidate].taken {
                        pool.swapRemove(at: i)
                        k -= 1 // Don't count this as a wasted try
                        continue
                    }
                    
                    // 2. We can't route to pads in the current comb row, but we
                    // leave them in the pool for later.
                    if row.contains(candidate) { continue }
                    
                    // 3. A chip pin never escapes to another pin of the same chip.
                    if data.pads[p].chip != 0,
                       data.pads[candidate].chip == data.pads[p].chip { continue }
                    
                    // 4. Try to route
                    guard let hit = routeToPad(from: a, pad: candidate, keepout: r, via: {
                        router.route(from: a, to: $0, keepout: r)
                    }) else { continue }
                    
                    // 5. Success! Route is found.
                    b = hit.port
                    combPath = hit.path
                    target = candidate
                    
                    // We successfully used this candidate, so remove it from the pool in O(1) time
                    pool.swapRemove(at: i)
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
                if stubDoublesBack(path, port: a, pad: p, fromStart: true) { continue }
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
            
            // 1. Pick first pad and lazy-clean if it's dead
            let ia = rng.int(0, pool.count - 1)
            let ai = pool[ia]
            if data.pads[ai].taken {
                pool.swapRemove(at: ia)
                attempt -= 1 // Don't count cleanup as a wasted routing attempt
                continue
            }
            
            // 2. Pick second pad and lazy-clean if it's dead
            var ib = rng.int(0, pool.count - 1)
            if ia == ib { ib = (ib + 1) % pool.count }
            let bi = pool[ib]
            if data.pads[bi].taken {
                pool.swapRemove(at: ib)
                attempt -= 1 // Don't count cleanup as a wasted routing attempt
                continue
            }
            
            // 3. Both pads are valid and available.
            let a = data.pads[ai]
            let b = data.pads[bi]
            
            // Never wire two pins of one chip together.
            if a.chip != 0 && a.chip == b.chip { continue }
            
            // Check Manhattan distance
            if abs(a.gx - b.gx) + abs(a.gy - b.gy) < 7 { continue }
            
            // 4. Try to route
            var best: [SIMD2<Int32>]?
            forEachPadPort(ai, keepout: r, toward: SIMD2(Int32(b.gx), Int32(b.gy))) { pa in
                guard let hit = routeToPad(from: pa, pad: bi, keepout: r, via: {
                    router.route(from: pa, to: $0, keepout: r)
                }) else { return false }
                
                if !stubDoublesBack(hit.path, port: pa, pad: ai, fromStart: true) {
                    best = hit.path
                    return true
                }
                return false
            }
            
            guard let path = best else { continue }
            
            // 5. Success! Claim the route and mark pads taken.
            grid.claim(path, width: w)
            data.pads[ai].taken = true
            data.pads[bi].taken = true
            
            // O(1) removal. MUST remove max first so the min index doesn't shift!
            pool.swapRemove(at: max(ia, ib))
            pool.swapRemove(at: min(ia, ib))
            
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
        var pts: [SIMD2<Int32>] = []
        // Optional micro-optimization: Pre-allocate enough space for the maximum possible meander size
        // Max folds (7) * 2 * (Max amp (6) + Max half (3)) = ~126 points
        pts.reserveCapacity(130)
        
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
            
            // Clear the array from previous failed attempts without deallocating memory,
            // then add the starting point back.
            pts.removeAll(keepingCapacity: true)
            pts.append(SIMD2(Int32(gx), Int32(gy)))
            
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
    /// Decimal digits of `v`, zero-padded to `digits`, built straight into the
    /// string's own storage.
    ///
    /// `String(format:)` goes through CoreFoundation's formatter, which showed
    /// up in the profile under its own name. Every candidate number is composed
    /// before it can be measured and tested for space, and most candidates are
    /// then rejected, so this runs about two hundred times a tile to keep a
    /// dozen.
    @inline(__always)
    private static func digits(_ v: Int, _ digits: Int) -> String {
        String(unsafeUninitializedCapacity: digits) { buf in
            var rest = v
            var i = digits - 1
            while i >= 0 { buf[i] = UInt8(48 + rest % 10); rest /= 10; i -= 1 }
            return digits
        }
    }
    
    func makeNumbers(_ n: Int) -> [NumberLabel] {
        var out: [NumberLabel] = []
        out.reserveCapacity(n) // Prevent array reallocation overhead
        var tries = 0
        
        let gridSizeFloat = Float(Board.gridSize) // Cache this if it's a static/global constant to avoid repeated lookups
        
        while out.count < n && tries < n * Draw.numberTries {
            tries += 1
            let kind = rng.int(0, 3)
            let text: String
            switch kind {
            case 0:
                text = TileGenerator.digits(rng.int(0, 999), 3)
            case 1:
                // "0x" then one or two uppercase hex digits, unpadded
                let v = rng.int(0, 255)
                text = String(unsafeUninitializedCapacity: 4) { buf in
                    func hex(_ d: Int) -> UInt8 { d < 10 ? UInt8(48 + d) : UInt8(55 + d) }
                    buf[0] = UInt8(ascii: "0"); buf[1] = UInt8(ascii: "x")
                    if v < 16 { buf[2] = hex(v); return 3 }
                    buf[2] = hex(v >> 4); buf[3] = hex(v & 15)
                    return 4
                }
            case 2:
                let whole = rng.int(0, 99)
                let frac = rng.int(0, 99)
                let wide = whole > 9
                text = String(unsafeUninitializedCapacity: wide ? 5 : 4) { buf in
                    var i = 0
                    if wide { buf[0] = UInt8(48 + whole / 10); i = 1 }
                    buf[i] = UInt8(48 + whole % 10); i += 1
                    buf[i] = UInt8(ascii: "."); i += 1
                    buf[i] = UInt8(48 + frac / 10); i += 1
                    buf[i] = UInt8(48 + frac % 10); i += 1
                    return i
                }
            default:
                let bits = rng.int(4, 7)
                text = String(unsafeUninitializedCapacity: bits) { buf in
                    for i in 0..<bits { buf[i] = UInt8(48 + rng.int(0, 1)) }
                    return bits
                }
            }
            let size = rng.int(Draw.numberSize)
            let vertical = rng.unit() < Double(Draw.numberVerticalChance)
            let tw = TextRaster.width(text, size: CGFloat(size))
            
            // Half-extents in px, plus a margin
            let ex = (vertical ? Float(size) : tw) / 2 + Draw.numberMargin
            let ey = (vertical ? tw : Float(size)) / 2 + Draw.numberMargin
            
            // Fast Float ceiling math instead of Double casting and .rounded(.up)
            let rx = Int(ceilf(ex / gridSizeFloat))
            let ry = Int(ceilf(ey / gridSizeFloat))
            
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


extension Array {
    mutating func swapRemove(at index: Int) {
        if index != count - 1 {
            swapAt(index, count - 1)
        }
        removeLast()
    }
}
