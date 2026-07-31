import CoreGraphics
import CoreText
import Foundation
import simd

enum TextRaster {
    /// `ui-monospace, Menlo, Consolas, monospace` — Menlo is the first entry of
    /// that stack present on both macOS and iOS.
    static func font(size: CGFloat, bold: Bool) -> CTFont {
        let name = bold ? "Menlo-Bold" : "Menlo"
        return CTFontCreateWithName(name as CFString, size, nil)
    }

    /// Advance width of a string, the equivalent of `measureText(str).width`.
    static func width(_ text: String, size: CGFloat, bold: Bool = false) -> Float {
        width(text, font: font(size: size, bold: bold))
    }

    /// The same, for a caller that already holds the font. Building a `CTFont`
    /// is not free, and the rasteriser needs one anyway.
    static func width(_ text: String, font f: CTFont) -> Float {
        var glyphs = [CGGlyph](repeating: 0, count: text.utf16.count)
        let chars = Array(text.utf16)
        guard CTFontGetGlyphsForCharacters(f, chars, &glyphs, chars.count) else {
            // Fall back to the nominal monospace advance. `CTFontGetSize`
            // returns exactly the size the font was created with.
            return Float(CTFontGetSize(f)) * 0.6 * Float(text.count)
        }
        var advances = [CGSize](repeating: .zero, count: glyphs.count)
        CTFontGetAdvancesForGlyphs(f, .horizontal, glyphs, &advances, glyphs.count)
        return Float(advances.reduce(0) { $0 + $1.width })
    }

    /// Canvas' `textBaseline = 'middle'` puts the em box's centre on the given
    /// y, so the baseline sits this far below it.
    static func middleBaselineOffset(size: CGFloat, bold: Bool) -> Float {
        let f = font(size: size, bold: bold)
        return Float((CTFontGetAscent(f) - CTFontGetDescent(f)) / 2)
    }

    /// 8-bit coverage of `text`, plus where pixel (0,0) of the bitmap sits in
    /// tile coordinates. `x` is the left edge of the text, `y` its middle.
    struct Coverage {
        let width: Int
        let height: Int
        let originX: Float
        let originY: Float
        let alpha: [UInt8]

        @inline(__always)
        func value(atTileX tx: Float, tileY ty: Float) -> UInt8 {
            let px = Int((tx - originX).rounded(.down))
            let py = Int((ty - originY).rounded(.down))
            guard px >= 0, py >= 0, px < width, py < height else { return 0 }
            return alpha[py * width + px]
        }
    }

    /// A string turned into pixels, with the font metrics that produced them.
    ///
    /// The one place in the project that draws text into a bitmap. Both users —
    /// the label lattice here and the silkscreen atlas in `GlyphAtlas` — need
    /// the identical Core Text sequence and differ only in the margin they want
    /// and in where they consider the box's origin to be, so `pad` is a
    /// parameter and the metrics come back for the caller to place with.
    struct Raster {
        let width: Int
        let height: Int
        let advance: CGFloat
        let ascent: CGFloat
        let descent: CGFloat
        let pad: CGFloat
        let alpha: [UInt8]

        /// Canvas' `textBaseline = 'middle'`, from metrics already in hand.
        var middleBaselineOffset: Float { Float((ascent - descent) / 2) }
    }

    /// Generous vertical margin: Menlo's ascent plus a pixel of AA fringe.
    private static let labelPad: CGFloat = 3

    static func raster(_ text: String, size: CGFloat, bold: Bool, pad: CGFloat) -> Raster? {
        let f = font(size: size, bold: bold)
        let advance = CGFloat(width(text, font: f))
        guard advance > 0 else { return nil }

        let ascent = CTFontGetAscent(f), descent = CTFontGetDescent(f)
        let w = Int((advance + pad * 2).rounded(.up))
        let h = Int((ascent + descent + pad * 2).rounded(.up))
        guard w > 0, h > 0,
              let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }

        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(w), height: CGFloat(h)))
        ctx.setFillColor(gray: 1, alpha: 1)

        // Core Text keys rather than AppKit/UIKit ones, so this file stays
        // free of a platform UI framework. The glyph colour comes from the
        // context's fill colour, already set to white above.
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): f,
            NSAttributedString.Key(kCTForegroundColorFromContextAttributeName as String): true,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attrs))
        // Core Graphics is y-up; both callers are y-down. Draw at a baseline
        // that leaves `pad` of descent below, then flip when mapping back.
        ctx.textPosition = CGPoint(x: pad, y: descent + pad)
        CTLineDraw(line, ctx)

        guard let data = ctx.data else { return nil }
        let raw = data.bindMemory(to: UInt8.self, capacity: w * h)
        // Core Graphics' bitmap rows already run top-down even though its user
        // space is y-up, so these land in tile order as they are.
        return Raster(width: w, height: h, advance: advance,
                      ascent: ascent, descent: descent, pad: pad,
                      alpha: [UInt8](UnsafeBufferPointer(start: raw, count: w * h)))
    }

    /// The rasterised bitmap on its own. It depends on the string, the size and
    /// the weight — never on where the label sits — so it is worth keeping.
    /// It carries the metrics `place` needs, so a cache hit costs no font.
    struct Bitmap {
        let width: Int
        let height: Int
        let ascent: Float
        let middleOffset: Float
        let pad: Float
        let alpha: [UInt8]
    }

    /// The expensive half of placing a label, which caches.
    static func bitmap(_ text: String, size: CGFloat, bold: Bool) -> Bitmap? {
        guard let r = raster(text, size: size, bold: bold, pad: labelPad) else { return nil }
        return Bitmap(width: r.width, height: r.height,
                      ascent: Float(r.ascent), middleOffset: r.middleBaselineOffset,
                      pad: Float(r.pad), alpha: r.alpha)
    }

    /// …and the cheap half, which just places it.
    static func place(_ b: Bitmap, x: Float, y: Float) -> Coverage {
        let baselineY = y + b.middleOffset
        return Coverage(width: b.width, height: b.height,
                        originX: x - b.pad,
                        originY: baselineY - (b.ascent + b.pad),
                        alpha: b.alpha)
    }

    /// (`glyphSites`) — the lattice points a label covers. Each one reserves a
    /// grid cell, so copper routes around the letter shapes.
    /// Lattice points a label covers. `cache` is the caller's own — one per
    /// worker, so no lock — and holds the rasterised bitmaps, which recur
    /// constantly: a handful of prefixes, two digits and a few sizes.
    static func glyphSites(_ text: String, size: CGFloat, bold: Bool,
                           x: Float, y: Float, tileWidth: Float,
                           cache: inout [String: Bitmap]) -> [SIMD2<Float>] {
        let key = "\(text)|\(size)|\(bold)"
        let bm: Bitmap
        if let hit = cache[key] { bm = hit }
        else if let made = bitmap(text, size: size, bold: bold) { bm = made; cache[key] = made }
        else { return [] }
        let cov = place(bm, x: x, y: y)
        let step = Float(Draw.glyphLattice)
        var out: [SIMD2<Float>] = []

        // Only the lattice points inside the label's box can be covered.
        var ly = max(0, (cov.originY / step).rounded(.down) * step)
        let maxY = min(Board.tileHeight, cov.originY + Float(cov.height))
        let maxX = min(tileWidth, cov.originX + Float(cov.width))
        while ly < maxY {
            var lx = max(0, (cov.originX / step).rounded(.down) * step)
            while lx < maxX {
                if cov.value(atTileX: lx, tileY: ly) > Draw.glyphAlphaCutoff {
                    out.append(SIMD2(lx, ly))
                }
                lx += step
            }
            ly += step
        }
        return out
    }
}
