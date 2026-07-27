import CoreGraphics
import CoreText
import Foundation
import Metal

/// The silkscreen numbers of one tile, rasterized and shelf-packed into that
/// slot's own atlas texture.
///
/// One atlas per slot rather than one shared growing atlas: a shared one would
/// have to be mutated from whichever thread is preparing a tile while the GPU
/// samples it for another. Per-slot makes tile preparation completely
/// independent — no lock, no eviction, no cross-thread mutation — and since the
/// textures are allocated once with the ring, it costs nothing per tile.
enum GlyphAtlas {

    struct Entry {
        var u0: Float, v0: Float, u1: Float, v1: Float
        /// The bitmap box in text-local pixels, where the origin is the
        /// string's horizontal centre on its `middle` baseline — matching
        /// Canvas' `textAlign = center; textBaseline = middle`.
        var minX: Float, minY: Float, maxX: Float, maxY: Float
    }

    private struct Bitmap {
        var pixels: [UInt8]
        var width: Int
        var height: Int
        var minX: Float
        var minY: Float
    }

    private static let padding = 2

    /// Rasterizes a tile's numbers into a texture the ring already owns.
    ///
    /// The texture is never created here — it belongs to the slot and has
    /// existed since startup — so this is safe to run on a worker while the
    /// render thread encodes: the slot is not composited until the whole tile
    /// is marked ready, and nothing else touches this texture meanwhile.
    /// A label that will not fit the atlas is dropped rather than growing it.
    static func render(_ labels: [NumberLabel], scale: Float,
                       into texture: MTLTexture, capacity: Int,
                       canvas: inout [UInt8]) -> [Entry?] {
        guard !labels.isEmpty else { return [] }
        let atlasWidth = texture.width, atlasHeight = texture.height
        // The canvas belongs to the worker and is reused tile after tile —
        // a megabyte allocated and zeroed per tile is a megabyte of pointless
        // work on the thread that is already the busiest.
        if canvas.count < atlasWidth * atlasHeight {
            canvas = [UInt8](repeating: 0, count: atlasWidth * atlasHeight)
        }
        var entries: [Entry?] = []
        entries.reserveCapacity(labels.count)

        var shelfX = 0, shelfY = 0, shelfHeight = 0
        var usedHeight = 0
        for (i, label) in labels.enumerated() {
            guard i < capacity,
                  let bitmap = rasterize(label.text, pointSize: label.size, scale: scale),
                  bitmap.width <= atlasWidth
            else { entries.append(nil); continue }

            if shelfX + bitmap.width > atlasWidth {
                shelfX = 0
                shelfY += shelfHeight
                shelfHeight = 0
            }
            guard shelfY + bitmap.height <= atlasHeight else {
                entries.append(nil)
                continue
            }
            // A fresh shelf starts as stale pixels from the previous tile, so
            // clear the strip this glyph is about to occupy plus the gap left
            // of it — never the whole atlas.
            for row in 0..<bitmap.height {
                let dst = (shelfY + row) * atlasWidth
                if shelfX == 0 {
                    for col in 0..<atlasWidth { canvas[dst + col] = 0 }
                }
                let src = row * bitmap.width
                for col in 0..<bitmap.width { canvas[dst + shelfX + col] = bitmap.pixels[src + col] }
            }
            let w = Float(atlasWidth), h = Float(atlasHeight)
            entries.append(Entry(u0: Float(shelfX) / w,
                                 v0: Float(shelfY) / h,
                                 u1: Float(shelfX + bitmap.width) / w,
                                 v1: Float(shelfY + bitmap.height) / h,
                                 minX: bitmap.minX,
                                 minY: bitmap.minY,
                                 maxX: bitmap.minX + Float(bitmap.width),
                                 maxY: bitmap.minY + Float(bitmap.height)))
            shelfX += bitmap.width
            shelfHeight = max(shelfHeight, bitmap.height)
            usedHeight = max(usedHeight, shelfY + bitmap.height)
        }

        // Only the rows actually written are cleared and uploaded. A tile has
        // a few dozen short strings; the rest of a 1024² atlas is untouched.
        guard usedHeight > 0 else { return entries }
        canvas.withUnsafeBytes { buf in
            texture.replace(region: MTLRegionMake2D(0, 0, atlasWidth, usedHeight),
                            mipmapLevel: 0,
                            withBytes: buf.baseAddress!, bytesPerRow: atlasWidth)
        }
        return entries
    }

    private static func rasterize(_ text: String, pointSize: Int, scale: Float) -> Bitmap? {
        let pixelSize = max(1, Int((Float(pointSize) * scale).rounded()))
        let font = TextRaster.font(size: CGFloat(pixelSize), bold: false)
        let advance = CGFloat(TextRaster.width(text, size: CGFloat(pixelSize)))
        guard advance > 0 else { return nil }
        let ascent = CTFontGetAscent(font), descent = CTFontGetDescent(font)
        let pad = CGFloat(padding)
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
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorFromContextAttributeName as String): true,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attrs))
        ctx.textPosition = CGPoint(x: pad, y: descent + pad)
        CTLineDraw(line, ctx)
        guard let data = ctx.data else { return nil }

        // Core Graphics' user space is y-up, but its bitmap rows already run
        // top-down — row 0 is the top of the drawn image — matching the tile.
        let raw = data.bindMemory(to: UInt8.self, capacity: w * h)
        let middleOffset = CGFloat(TextRaster.middleBaselineOffset(
            size: CGFloat(pixelSize), bold: false))
        return Bitmap(pixels: [UInt8](UnsafeBufferPointer(start: raw, count: w * h)),
                      width: w, height: h,
                      minX: Float(-advance / 2 - pad),
                      minY: Float(middleOffset - ascent - pad))
    }
}
