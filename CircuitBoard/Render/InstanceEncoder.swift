import Foundation
import simd

/// Everything one tile hands to the GPU. Built once when a tile is baked, then
/// thrown away — the result lives in the tile's texture slice.
struct TileInstances {
    /// Trace centrelines, trace-major, expanded into stroke geometry by
    /// `kernel_expand_traces`.
    var pathPoints: [PCBPathPoint] = []
    var traceInfos: [PCBTraceInfo] = []

    /// Teardrops, package bodies, silkscreen decor and pads — in that order,
    /// which is the order Canvas drew them and therefore the order they must
    /// blend in.
    var solids: [PCBShapeInstance] = []
    /// Index into `solids` where the pads begin. They are encoded last, so a
    /// pass can draw them alone with a buffer offset — which is what lets the
    /// parallax layers show pads without their teardrops, packages and decor.
    var padStart = 0
    var glyphs: [PCBGlyphInstance] = []

    /// Ribbon corners the expansion kernel will write: two per centreline
    /// point, plus a duplicate either side of every trace to stitch them all
    /// into one strip. Counted here because this is where trace boundaries are
    /// known — the kernel is one thread per point and cannot see them.
    var traceVertexCount = 0
}

/// Turns a generated tile into GPU instances. This is pure repackaging — the
/// per-primitive expansion of a stroked polyline, which is the part that
/// actually scales with board complexity, happens in the Metal kernel.
struct InstanceEncoder {
    let geo: BoardGeometry
    let scale: Float
    /// When false, copper meets a pad as a plain straight stroke.
    let teardrops: Bool
    /// When false, pad metal is left at its own brightness instead of being
    /// pushed above 1 — so there is nothing for the bloom to find even if it
    /// were switched back on.
    let glow: Bool
    /// The scheme the tile was generated under. Its trace colours are already
    /// baked into `TileData`; this supplies everything the generator did not
    /// choose — backings, holes, package bodies, silkscreen ink.
    let palette: Palette
    /// One per `TileData.number`, in order, already rasterized into this tile's
    /// own atlas.
    let glyphEntries: [GlyphAtlas.Entry?]

    func encode(_ data: TileData) -> TileInstances {
        var out = TileInstances()
        // Polylines are needed twice — once as kernel input, once for the
        // teardrops — so compute them once.
        var polylines: [[SIMD2<Float>]] = []
        polylines.reserveCapacity(data.traces.count)
        for t in data.traces { polylines.append(data.polyline(of: t, geo)) }

        // Reserve before appending: these arrays start empty and take thousands
        // of appends each, so otherwise they realloc and copy their way up
        // through every power of two.
        //
        // Kept because it is right, not because it is fast — measured, it moved
        // a tile from 253 ms to 250 ms in Debug and not at all in Release, both
        // inside the noise. Written down so the next person does not spend an
        // afternoon here: the tile cost is A\* routing, about 85 ms of a 130 ms
        // tile at full width, and neither reserving nor `memcpy` touches that.
        // The copies into Metal memory are already single `copyMemory` calls,
        // which is `memcpy`.
        var pointTotal = 0
        for pts in polylines where pts.count >= 2 { pointTotal += pts.count }
        out.pathPoints.reserveCapacity(pointTotal)
        out.traceInfos.reserveCapacity(data.traces.count)
        out.glyphs.reserveCapacity(data.numbers.count)
        // Teardrops are two per trace, a pad is a backing, a ring and a hole,
        // and a package body is a die, a rim and two inner outlines. Over rather
        // than under: a reservation that falls short has bought nothing.
        out.solids.reserveCapacity(data.traces.count * 2 + data.pads.count * 3
                                   + data.chips.count * 6 + data.decor.count * 4)

        encodeTraces(data, polylines, into: &out)
        if teardrops { encodeTeardrops(data, polylines, into: &out) }
        for c in data.chips { encodeChip(c, into: &out) }
        for d in data.decor { encodeDecor(d, into: &out) }
        out.padStart = out.solids.count
        for p in data.pads { encodePad(p, into: &out) }
        encodeNumbers(data, into: &out)
        return out
    }

    // MARK: - Traces

    /// A stable, effectively random seed for one trace's pulse clock.
    ///
    /// Taken from the trace's own geometry rather than from a counter, because
    /// a counter would give the n-th trace of every tile the same clock and the
    /// board would beat in bands as tiles march past. The endpoints already come
    /// out of the generator's seeded stream, so hashing them is hashing the
    /// seed — and it is stable for the life of the tile, which matters: a clock
    /// that changed on a re-encode would make the light jump.
    private static func pulseSeed(_ pts: [SIMD2<Float>], _ ordinal: Int) -> UInt32 {
        let a = pts[0], b = pts[pts.count - 1]
        let bits = a.x.bitPattern &+ (a.y.bitPattern &* 2_654_435_761)
            &+ (b.x.bitPattern &* 40_503) &+ (b.y.bitPattern &* 73_856_093)
        return Rng.hash(bits, UInt32(truncatingIfNeeded: ordinal &* 2_246_822_519))
    }

    private func encodeTraces(_ data: TileData, _ polylines: [[SIMD2<Float>]],
                              into out: inout TileInstances) {

        // Where this trace's block of ribbon vertices starts. Each block is
        // `2 * points + 2`: the two extras are duplicates of the first and last
        // corner, and the triangles spanning them are degenerate, which is what
        // lets every trace in the tile go out as one strip.
        var vertexCursor = 0
        for (i, pts) in polylines.enumerated() where pts.count >= 2 {
            let t = data.traces[i]
            let index = UInt32(out.traceInfos.count)
            let blockStart = vertexCursor
            vertexCursor += pts.count * 2 + 2
            // Arc length along the trace, normalised, so the pulse travels at a
            // steady speed rather than jumping between long and short segments.
            var total: Float = 0
            for j in 1..<pts.count { total += distance(pts[j - 1], pts[j]) }
            let inverse = total > 1e-5 ? 1 / total : 0

            // Half of a seam crossing: take the clock from what the two tiles
            // agreed rather than from this half's own geometry, or the line
            // gets one runner per half at unrelated speeds. An upstream half
            // also measures its arc backwards from the boundary — a sign, not a
            // reflection, so both halves parameterise the same wave. It costs
            // nothing: the ribbon, and the teardrops keyed to `padA` and
            // `padB`, are untouched.
            let upstream = t.crossing?.towardSeam ?? false
            let clock = t.crossing.map { Pulse.clock($0, length: total) }
                ?? Pulse.clock(from: Rng(seed: Self.pulseSeed(pts, i)))
            // Non-zero puts this trace on the shared wave, and converts the
            // board-unit runner constants into this trace's own parameter.
            let unit: Float = t.crossing != nil && total > 1e-5 ? 1 / total : 0
            out.traceInfos.append(PCBTraceInfo(halfWidth: t.width / 2,
                                               cr: t.color.x, cg: t.color.y, cb: t.color.z,
                                               ca: 1,
                                               pulsePhase: clock.phase,
                                               pulsePeriod: clock.period,
                                               pulseSpeed: clock.speed,
                                               pulseUnit: unit))

            var travelled: Float = 0
            for (j, p) in pts.enumerated() {
                if j > 0 { travelled += distance(pts[j - 1], p) }
                var flags: UInt32 = 0
                if j == 0 { flags |= PCB_POINT_FIRST }
                if j == pts.count - 1 { flags |= PCB_POINT_LAST }
                // +1 leaves room for the leading duplicate.
                out.pathPoints.append(PCBPathPoint(x: p.x, y: p.y,
                                                   traceIndex: index, flags: flags,
                                                   // signed: an upstream half
                                                   // counts back from the seam
                                                   s: upstream ? -travelled * inverse
                                                               : travelled * inverse,
                                                   vertexBase: Float(blockStart + 1 + j * 2),
                                                   _pad1: 0, _pad2: 0))
            }
        }
        out.traceVertexCount = vertexCursor
    }

    /// The flare seen where copper lands on a pad in every board photograph.
    private func encodeTeardrops(_ data: TileData, _ polylines: [[SIMD2<Float>]],
                                 into out: inout TileInstances) {
        for (i, pts) in polylines.enumerated() where pts.count > 1 {
            let t = data.traces[i]
            let color = SIMD4(t.color, 1)

            func flare(_ padIndex: Int?, _ a: SIMD2<Float>, _ b: SIMD2<Float>) {
                guard let padIndex else { return }
                let pad = data.pads[padIndex]
                let rad = pad.type == .smd ? min(pad.w, pad.h) / 2 : pad.pr
                let w1 = t.width / 2
                let w2 = min(rad * Draw.teardropRatio, t.width / 2 + Draw.teardropExtra)
                if let s = Shape.teardrop(from: a, to: b, w1: w1, w2: w2, color: color) {
                    out.solids.append(s)
                }
            }
            flare(t.padA, pts[1], pts[0])
            flare(t.padB, pts[pts.count - 2], pts[pts.count - 1])
        }
    }

    // MARK: - Package bodies

    /// A dark die (or a hollow body) with a metal rim and two fainter inner
    /// outlines, so it reads as a component rather than a stray rectangle.
    private func encodeChip(_ c: ChipBody, into out: inout TileInstances) {
        let x0 = geo.px(c.x0), y0 = geo.py(c.y0)
        let x1 = geo.px(c.x1), y1 = geo.py(c.y1)
        let mx = (x0 + x1) / 2, my = (y0 + y1) / 2
        let w = x1 - x0, h = y1 - y0
        let m = min(w, h)
        let metal = SIMD4(c.metal, 1)

        out.solids.append(Shape.roundRect(mx, my, w, h, radius: Draw.Chip.bodyRadius,
                                          color: c.solid ? palette.chipSolidBody
                                                         : palette.chipOutlineBody))
        out.solids.append(Shape.roundRect(mx, my, w, h, radius: Draw.Chip.bodyRadius,
                                          strokeWidth: Draw.Chip.bodyStroke, color: metal))

        let solidRing = SIMD4(c.metal, Draw.Chip.solidRingAlpha)
        out.solids.append(Shape.roundRect(mx, my,
                                          w - 2 * c.insetSolid * m,
                                          h - 2 * c.insetSolid * m,
                                          radius: Draw.Chip.innerRadius,
                                          strokeWidth: Draw.Chip.solidRingWidth,
                                          color: solidRing))

        // The dashed ring becomes explicit round-capped dashes: a dash pattern
        // has no closed form in a distance field, and the perimeter walk is
        // cheap at four corners.
        let dashRing = SIMD4(c.metal, Draw.Chip.dashRingAlpha)
        let path = Shape.roundRectPath(mx, my,
                                       w - 2 * c.insetDash * m,
                                       h - 2 * c.insetDash * m,
                                       radius: Draw.Chip.innerRadius)
        out.solids.append(contentsOf: Shape.dashed(path,
                                                   on: Draw.Chip.dashOn,
                                                   off: Draw.Chip.dashOff,
                                                   width: Draw.Chip.dashRingWidth,
                                                   color: dashRing))
    }

    // MARK: - Silkscreen decor

    private func encodeDecor(_ d: DecorItem, into out: inout TileInstances) {
        // `vertical` and `rot` are never both set in the generator, so one
        // angle covers the `rotate` calls the original made in sequence.
        let angle: Float = d.vertical ? .pi / 2 : d.rotation
        let f = DecorFrame(origin: SIMD2(d.x, d.y), rotation: angle, scale: d.scale)
        let color = SIMD4(d.metal, Draw.Decor.alpha)

        switch d.kind {
        case .soic:
            let step = d.bw / Float(d.pinCount)
            for i in 0..<d.pinCount {
                let bx = -d.bw / 2 + step * (Float(i) + 0.5)
                if let l = f.line(SIMD2(bx, -d.bh / 2), SIMD2(bx, -d.bh / 2 - d.leg),
                                  width: Draw.Decor.soicLegWidth, color: color) {
                    out.solids.append(l)
                }
                if let l = f.line(SIMD2(bx, d.bh / 2), SIMD2(bx, d.bh / 2 + d.leg),
                                  width: Draw.Decor.soicLegWidth, color: color) {
                    out.solids.append(l)
                }
            }
            out.solids.append(f.rect(0, 0, d.bw, d.bh,
                                     radius: Draw.Decor.bodyRadius,
                                     strokeWidth: d.solid ? 0 : Draw.Decor.soicLegWidth,
                                     color: color))

        case .block:
            let yo = d.paired ? d.bh / 2 + 2 : 0
            out.solids.append(f.rect(0, -yo, d.bw, d.bh,
                                     radius: Draw.Decor.bodyRadius, color: color))
            if d.paired {
                out.solids.append(f.rect(0, yo, d.bw, d.bh,
                                         radius: Draw.Decor.bodyRadius, color: color))
            }

        case .dots:
            for j in 0..<d.ny {
                for i in 0..<d.nx {
                    out.solids.append(f.dot(-Float(d.nx - 1) * d.spacing / 2 + Float(i) * d.spacing,
                                            -Float(d.ny - 1) * d.spacing / 2 + Float(j) * d.spacing,
                                            radius: Draw.Decor.dotRadius, color: color))
                }
            }

        case .frame:
            out.solids.append(f.strokeRect(-d.size / 2, -d.size / 2, d.size, d.size,
                                           width: Draw.Decor.frameOuterWidth, color: color))
            let inset = Draw.Decor.frameInset
            out.solids.append(f.strokeRect(-d.size / 2 + inset, -d.size / 2 + inset,
                                           d.size - inset * 2, d.size - inset * 2,
                                           width: Draw.Decor.frameInnerWidth, color: color))

        case .bars:
            for i in 0..<d.barCount {
                let bx = -Float(d.barCount - 1) * d.spacing / 2 + Float(i) * d.spacing
                out.solids.append(f.fillRect(bx - d.barWidth / 2, -d.height / 2,
                                             d.barWidth, d.height, color: color))
            }

        case .tinySquare:
            out.solids.append(f.fillRect(-d.size / 2, -d.size / 2, d.size, d.size, color: color))

        case .diagonal:
            out.solids.append(f.fillRect(-d.bw / 2, -d.bh / 2, d.bw, d.bh, color: color))

        case .dip:
            let w = Float(d.nx - 1) * d.spacing
            let pad = Draw.Decor.dipPadding
            let hh = Draw.Decor.dipHalfHeight
            out.solids.append(f.strokeRect(-w / 2 - pad, -hh, w + pad * 2, hh * 2,
                                           width: Draw.Decor.dipOutlineWidth, color: color))
            for i in 0..<d.nx {
                let bx = -w / 2 + Float(i) * d.spacing
                out.solids.append(f.dot(bx, -Draw.Decor.dipPinOffset,
                                        radius: Draw.Decor.dipPinRadius, color: color))
                out.solids.append(f.dot(bx, Draw.Decor.dipPinOffset,
                                        radius: Draw.Decor.dipPinRadius, color: color))
            }
        }
    }

    // MARK: - Pads

    /// Each pad is its stack of layers in drawing order: dark backing, metal,
    /// then the hole. Submission order inside the one instanced draw gives the
    /// same result the successive `fill()` calls did.
    private func encodePad(_ p: Pad, into out: inout TileInstances) {
        let cx = geo.px(p.gx), cy = geo.py(p.gy)
        // Above 1 on purpose: the target is HDR, so this is what the bloom
        // prefilter picks out as a light source.
        let metal = SIMD4(p.metal * (glow ? Palette.padGlow : 1), 1)
        let hole = SIMD4(palette.padHole, 1)
        let backing = palette.padBacking

        switch p.type {
        case .smd:
            let grow = Draw.Pad.smdBackingGrow
            out.solids.append(Shape.roundRect(cx, cy, p.w + grow, p.h + grow,
                                              radius: Draw.Pad.smdBackingRadius,
                                              color: backing))
            out.solids.append(Shape.roundRect(cx, cy, p.w, p.h,
                                              radius: Draw.Pad.smdRadius, color: metal))
            out.solids.append(Shape.roundRect(cx, cy, p.w, p.h,
                                              radius: Draw.Pad.smdRadius,
                                              strokeWidth: Draw.Pad.smdOutlineWidth,
                                              color: palette.smdOutline))

        case .square:
            let side = p.pr * 2
            out.solids.append(Shape.roundRect(cx, cy,
                                              side + Draw.Pad.squareBackingGrow,
                                              side + Draw.Pad.squareBackingGrow,
                                              radius: Draw.Pad.squareBackingRadius,
                                              color: backing))
            out.solids.append(Shape.roundRect(cx, cy, side, side,
                                              radius: Draw.Pad.squareRadius, color: metal))
            out.solids.append(Shape.circle(cx, cy, radius: p.pr * Draw.Pad.squareHoleRatio, color: hole))

        case .ring:
            out.solids.append(Shape.circle(cx, cy, radius: p.pr + Draw.Pad.ringBackingGrow,
                                           color: backing))
            out.solids.append(Shape.circle(cx, cy, radius: p.pr, color: metal))
            out.solids.append(Shape.circle(cx, cy, radius: p.pr * Draw.Pad.ringHoleRatio, color: hole))
            out.solids.append(Shape.circle(cx, cy, radius: p.pr * Draw.Pad.ringInnerRatio,
                                           strokeWidth: Draw.Pad.ringInnerWidth, color: metal))

        case .via, .throughHole:
            out.solids.append(Shape.circle(cx, cy, radius: p.pr + Draw.Pad.roundBackingGrow,
                                           color: backing))
            out.solids.append(Shape.circle(cx, cy, radius: p.pr, color: metal))
            let ratio = p.type == .via ? Draw.Pad.viaHoleRatio : Draw.Pad.throughHoleRatio
            out.solids.append(Shape.circle(cx, cy, radius: p.pr * ratio, color: hole))
        }
    }

    // MARK: - Numbers

    /// Faint blue silkscreen on the substrate.
    private func encodeNumbers(_ data: TileData, into out: inout TileInstances) {
        for (i, n) in data.numbers.enumerated() {
            guard i < glyphEntries.count, let e = glyphEntries[i] else { continue }
            let color = SIMD4(palette.numberInk, n.alpha)

            let angle: Float = n.vertical ? -.pi / 2 : 0
            let c = cos(angle), s = sin(angle)
            // The atlas box is in text-local pixels; bring it to board units and
            // place it through the same rotation Canvas applied.
            let localCX = (e.minX + e.maxX) / 2 / scale
            let localCY = (e.minY + e.maxY) / 2 / scale
            let cx = n.x + c * localCX - s * localCY
            let cy = n.y + s * localCX + c * localCY

            out.glyphs.append(PCBGlyphInstance(
                cx: cx, cy: cy,
                hw: (e.maxX - e.minX) / 2 / scale,
                hh: (e.maxY - e.minY) / 2 / scale,
                cosR: c, sinR: s,
                u0: e.u0, v0: e.v0, u1: e.u1, v1: e.v1,
                cr: color.x, cg: color.y, cb: color.z, ca: color.w))
        }
    }
}
