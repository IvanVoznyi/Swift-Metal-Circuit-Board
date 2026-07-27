import Foundation
import simd

/// Everything one tile of board is made of, in board units, ready to be turned
/// into GPU instances. The JS version passed object references around; here
/// pads are addressed by index into `pads` so the structs stay values.

enum PadType {
    case via, throughHole, smd, square, ring
}

struct Pad {
    var gx: Int
    var gy: Int
    var type: PadType
    /// Radius for round pads; unused for `smd`.
    var pr: Float = 0
    /// Full size for `smd`.
    var w: Float = 0
    var h: Float = 0
    /// Keepout half-extents in cells.
    var rx: Int = 0
    var ry: Int = 0
    var metal: SIMD3<Float> = .zero
    /// Non-zero when this pad is a pin of that package — two pins of one chip
    /// are never wired to each other.
    var chip: Int = 0
    var taken = false
}

struct Trace {
    /// Centreline in grid cells.
    var path: [SIMD2<Int32>]
    var color: SIMD3<Float>
    var width: Float
    var cls: TraceClass
    var padA: Int? = nil
    var padB: Int? = nil
    /// This end is a seam port: the polyline runs off the tile edge so the
    /// neighbouring tile's matching trace continues the line.
    var edgeA = false
}

struct ChipBody {
    var x0: Int, y0: Int, x1: Int, y1: Int
    var metal: SIMD3<Float>
    /// `solid` = dark die, otherwise a hollow body showing the substrate.
    var solid: Bool
    var insetSolid: Float
    var insetDash: Float
}

struct DecorItem {
    enum Kind { case soic, block, dots, frame, bars, tinySquare, diagonal, dip }

    var kind: Kind
    var x: Float
    var y: Float
    var metal: SIMD3<Float>
    var scale: Float = 1
    var rotation: Float = 0
    var vertical = false

    // Per-kind fields, mirroring the JS object shapes.
    var bw: Float = 0
    var bh: Float = 0
    var pinCount: Int = 0
    var leg: Float = 0
    var solid = false
    var paired = false
    var nx: Int = 0
    var ny: Int = 0
    var spacing: Float = 0
    var size: Float = 0
    var barCount: Int = 0
    var barWidth: Float = 0
    var height: Float = 0
}

struct NumberLabel {
    var x: Float
    var y: Float
    var text: String
    var size: Int
    var vertical: Bool
    var alpha: Float
}

struct TileData {
    var traces: [Trace] = []
    var pads: [Pad] = []
    var numbers: [NumberLabel] = []
    var chips: [ChipBody] = []
    var decor: [DecorItem] = []

    /// `polyOf` — the drawn polyline in board units, with the collinear
    /// interior points dropped. Douglas-Peucker was rejected upstream because
    /// any non-zero tolerance moves the line off the corridor `claim`
    /// reserved; exact collinear removal deviates by nothing.
    func polyline(of trace: Trace, _ geo: BoardGeometry) -> [SIMD2<Float>] {
        var pts: [SIMD2<Float>] = []
        pts.reserveCapacity(trace.path.count + 2)

        if let a = trace.padA {
            pts.append(SIMD2(geo.px(pads[a].gx), geo.py(pads[a].gy)))
        } else if trace.edgeA, let first = trace.path.first {
            // Run the line clear off the tile edge so it meets its twin.
            pts.append(SIMD2(geo.px(Int(first.x)),
                             first.y == 0 ? -5 : Board.tileHeight + 5))
        }
        for c in trace.path {
            pts.append(SIMD2(geo.px(Int(c.x)), geo.py(Int(c.y))))
        }
        if let b = trace.padB {
            pts.append(SIMD2(geo.px(pads[b].gx), geo.py(pads[b].gy)))
        }
        return TileData.dedupe(pts)
    }

    /// Drops interior points exactly collinear with their neighbours.
    static func dedupe(_ pts: [SIMD2<Float>]) -> [SIMD2<Float>] {
        guard pts.count >= 3 else { return pts }
        var out: [SIMD2<Float>] = [pts[0]]
        out.reserveCapacity(pts.count)
        for i in 1..<(pts.count - 1) {
            let a = out[out.count - 1], b = pts[i], c = pts[i + 1]
            let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
            if abs(cross) > 1e-6 { out.append(b) }
        }
        out.append(pts[pts.count - 1])
        return out
    }
}
