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
        let f = font(size: size, bold: bold)
        var glyphs = [CGGlyph](repeating: 0, count: text.utf16.count)
        let chars = Array(text.utf16)
        guard CTFontGetGlyphsForCharacters(f, chars, &glyphs, chars.count) else {
            // Fall back to the nominal monospace advance.
            return Float(size) * 0.6 * Float(text.count)
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

    /// The rasterised bitmap on its own. It depends on the string, the size and
    /// the weight — never on where the label sits — so it is worth keeping.
    struct Bitmap {
        let width: Int
        let height: Int
        let ascent: Float
        let pad: Float
        let alpha: [UInt8]
    }

    /// `coverage` split in two: the expensive half, which caches.
    static func bitmap(_ text: String, size: CGFloat, bold: Bool) -> Bitmap? {
        guard let c = coverage(text, size: size, bold: bold, x: 0, y: 0) else { return nil }
        let f = font(size: size, bold: bold)
        return Bitmap(width: c.width, height: c.height,
                      ascent: Float(CTFontGetAscent(f)), pad: 3, alpha: c.alpha)
    }

    /// …and the cheap half, which just places it.
    static func place(_ b: Bitmap, text: String, size: CGFloat, bold: Bool,
                      x: Float, y: Float) -> Coverage {
        let baselineY = y + middleBaselineOffset(size: size, bold: bold)
        return Coverage(width: b.width, height: b.height,
                        originX: x - b.pad,
                        originY: baselineY - (b.ascent + b.pad),
                        alpha: b.alpha)
    }

    static func coverage(_ text: String, size: CGFloat, bold: Bool,
                         x: Float, y: Float) -> Coverage? {
        let f = font(size: size, bold: bold)
        let advance = CGFloat(width(text, size: size, bold: bold))
        guard advance > 0 else { return nil }

        // Generous vertical margin: Menlo's ascent plus a pixel of AA fringe.
        let pad: CGFloat = 3
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

        // Core Graphics is y-up; the tile is y-down. Draw at a baseline that
        // leaves `pad` of descent below, then flip when mapping back.
        let baselineFromBottom = descent + pad
        // Core Text keys rather than AppKit/UIKit ones, so this file stays
        // free of a platform UI framework. The glyph colour comes from the
        // context's fill colour, already set to white above.
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): f,
            NSAttributedString.Key(kCTForegroundColorFromContextAttributeName as String): true,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attrs))
        ctx.textPosition = CGPoint(x: pad, y: baselineFromBottom)
        CTLineDraw(line, ctx)

        guard let data = ctx.data else { return nil }
        let raw = data.bindMemory(to: UInt8.self, capacity: w * h)
        // Core Graphics' bitmap rows already run top-down even though its user
        // space is y-up, so these land in tile order as they are.
        let alpha = [UInt8](UnsafeBufferPointer(start: raw, count: w * h))

        let baselineY = y + middleBaselineOffset(size: size, bold: bold)
        return Coverage(width: w, height: h,
                        originX: x - Float(pad),
                        originY: baselineY - Float(ascent + pad),
                        alpha: alpha)
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
        let cov = place(bm, text: text, size: size, bold: bold, x: x, y: y)
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
