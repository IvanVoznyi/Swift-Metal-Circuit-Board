import Foundation
import simd

/// Constructors for the shared vector instance. Every Canvas2D primitive the
/// board uses reduces to one of four shapes, so a tile is three draw calls of
/// one pipeline instead of thousands of context state changes.
enum Shape {

    static func instance(kind: PCBShapeKind,
                         cx: Float, cy: Float,
                         hw: Float, hh: Float,
                         cosR: Float = 1, sinR: Float = 0,
                         radius: Float = 0,
                         strokeWidth: Float = 0,
                         expandable: Bool = false,
                         color: SIMD4<Float>) -> PCBShapeInstance {
        PCBShapeInstance(cx: cx, cy: cy, hw: hw, hh: hh,
                         cosR: cosR, sinR: sinR,
                         radius: radius, strokeWidth: strokeWidth,
                         kind: Float(kind.rawValue),
                         expandable: expandable ? 1 : 0,
                         cr: color.x, cg: color.y, cb: color.z, ca: color.w)
    }

    /// `roundRect(g, x, y, w, h, r)` — centred, with the radius clamped the way
    /// `arcTo` clamps it.
    static func roundRect(_ cx: Float, _ cy: Float, _ w: Float, _ h: Float,
                          radius: Float, rotation: Float = 0,
                          strokeWidth: Float = 0,
                           color: SIMD4<Float>) -> PCBShapeInstance {
        let hw = max(w, 0) / 2, hh = max(h, 0) / 2
        return instance(kind: PCBShapeRoundRect, cx: cx, cy: cy, hw: hw, hh: hh,
                        cosR: cos(rotation), sinR: sin(rotation),
                        radius: min(radius, min(hw, hh)),
                        strokeWidth: strokeWidth, color: color)
    }

    static func circle(_ cx: Float, _ cy: Float, radius r: Float,
                       strokeWidth: Float = 0,
                       color: SIMD4<Float>) -> PCBShapeInstance {
        instance(kind: PCBShapeRoundRect, cx: cx, cy: cy, hw: r, hh: r,
                 radius: r, strokeWidth: strokeWidth, color: color)
    }

    /// A round-capped line, i.e. what Canvas draws with `lineCap = 'round'`.
    /// Expressed as a stadium so it stays one instance.
    static func capsule(from a: SIMD2<Float>, to b: SIMD2<Float>,
                        width: Float, color: SIMD4<Float>) -> PCBShapeInstance? {
        let v = b - a
        let len = (v.x * v.x + v.y * v.y).squareRoot()
        let r = width / 2
        guard r > 0 else { return nil }
        let dir = len > 1e-6 ? v / len : SIMD2<Float>(1, 0)
        let mid = (a + b) / 2
        return instance(kind: PCBShapeRoundRect, cx: mid.x, cy: mid.y,
                        hw: len / 2 + r, hh: r,
                        cosR: dir.x, sinR: dir.y,
                        radius: r, color: color)
    }

    /// The teardrop fillet where a trace meets its pad: half-width `w1` at `a`,
    /// flaring to `w2` at `b`.
    static func teardrop(from a: SIMD2<Float>, to b: SIMD2<Float>,
                         w1: Float, w2: Float, color: SIMD4<Float>) -> PCBShapeInstance? {
        let v = b - a
        let len = (v.x * v.x + v.y * v.y).squareRoot()
        guard len >= 1 else { return nil }   // the original's `if (L < 1) return`
        let dir = v / len
        return instance(kind: PCBShapeTrapezoid, cx: a.x, cy: a.y,
                        hw: len, hh: w1,
                        cosR: dir.x, sinR: dir.y,
                        radius: w2, color: color)
    }

    /// Perimeter of a centred rounded rectangle, clockwise from the point
    /// `roundRect`'s `moveTo` starts at, for walking dashes along.
    static func roundRectPath(_ cx: Float, _ cy: Float, _ w: Float, _ h: Float,
                              radius: Float, cornerSteps: Int = 6) -> [SIMD2<Float>] {
        let hw = max(w, 0) / 2, hh = max(h, 0) / 2
        let r = min(radius, min(hw, hh))
        var pts: [SIMD2<Float>] = []
        // Corner centres, clockwise from top-left.
        let corners: [(SIMD2<Float>, Float)] = [
            (SIMD2(cx + hw - r, cy - hh + r), -.pi / 2),  // top-right
            (SIMD2(cx + hw - r, cy + hh - r), 0),         // bottom-right
            (SIMD2(cx - hw + r, cy + hh - r), .pi / 2),   // bottom-left
            (SIMD2(cx - hw + r, cy - hh + r), .pi),       // top-left
        ]
        pts.append(SIMD2(cx - hw + r, cy - hh))
        for (centre, startAngle) in corners {
            for s in 0...cornerSteps {
                let a = startAngle + .pi / 2 * Float(s) / Float(cornerSteps)
                pts.append(centre + SIMD2(cos(a) * r, sin(a) * r))
            }
        }
        pts.append(pts[0])
        return pts
    }

    /// Splits a polyline into `on`/`off` runs and returns one round-capped
    /// instance per run — Canvas' `setLineDash` with the round cap it inherits.
    static func dashed(_ path: [SIMD2<Float>], on: Float, off: Float,
                       width: Float, color: SIMD4<Float>) -> [PCBShapeInstance] {
        guard path.count >= 2, on > 0, off > 0 else { return [] }
        var out: [PCBShapeInstance] = []
        var remaining = on
        var drawing = true
        var runStart = path[0]

        for i in 1..<path.count {
            var a = path[i - 1]
            let b = path[i]
            var segLen = simd_length(b - a)
            while segLen > 0 {
                let step = min(segLen, remaining)
                let dir = simd_length(b - a) > 1e-6 ? simd_normalize(b - a) : SIMD2<Float>(1, 0)
                let next = a + dir * step
                if drawing && step > 0, let c = capsule(from: runStart, to: next,
                                                        width: width, color: color) {
                    out.append(c)
                }
                remaining -= step
                segLen -= step
                a = next
                if remaining <= 1e-5 {
                    drawing.toggle()
                    remaining = drawing ? on : off
                    runStart = a
                }
            }
        }
        return out
    }
}

/// The local frame a decor component draws in: `translate(x, y)`, optional
/// rotation, then `scale(sc)` — so its stroke widths and radii scale too, which
/// is what keeps a scaled-up part inside the keepout it reserved.
struct DecorFrame {
    let origin: SIMD2<Float>
    let rotation: Float
    let scale: Float

    private var c: Float { cos(rotation) }
    private var s: Float { sin(rotation) }

    func point(_ p: SIMD2<Float>) -> SIMD2<Float> {
        let q = p * scale
        return origin + SIMD2(c * q.x - s * q.y, s * q.x + c * q.y)
    }

    func length(_ v: Float) -> Float { v * scale }

    /// A rectangle given the way Canvas gives one: centred at a local point.
    func rect(_ lx: Float, _ ly: Float, _ w: Float, _ h: Float,
              radius: Float = 0, strokeWidth: Float = 0,
              color: SIMD4<Float>) -> PCBShapeInstance {
        let centre = point(SIMD2(lx, ly))
        return Shape.roundRect(centre.x, centre.y, length(w), length(h),
                               radius: length(radius), rotation: rotation,
                               strokeWidth: length(strokeWidth), color: color)
    }

    /// `fillRect(x, y, w, h)` — Canvas' corner-anchored form.
    func fillRect(_ lx: Float, _ ly: Float, _ w: Float, _ h: Float,
                  color: SIMD4<Float>) -> PCBShapeInstance {
        rect(lx + w / 2, ly + h / 2, w, h, color: color)
    }

    func strokeRect(_ lx: Float, _ ly: Float, _ w: Float, _ h: Float,
                    width: Float, color: SIMD4<Float>) -> PCBShapeInstance {
        rect(lx + w / 2, ly + h / 2, w, h, strokeWidth: width, color: color)
    }

    func dot(_ lx: Float, _ ly: Float, radius: Float,
             color: SIMD4<Float>) -> PCBShapeInstance {
        let centre = point(SIMD2(lx, ly))
        return Shape.circle(centre.x, centre.y, radius: length(radius), color: color)
    }

    func line(_ a: SIMD2<Float>, _ b: SIMD2<Float>, width: Float,
              color: SIMD4<Float>) -> PCBShapeInstance? {
        Shape.capsule(from: point(a), to: point(b),
                      width: length(width), color: color)
    }
}
