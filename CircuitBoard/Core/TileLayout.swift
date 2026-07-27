import Foundation

/// Where each resident tile's top edge lands on the drawable, in pixels.
///
/// Kept separate from the renderer because this is the other half of the
/// smooth-motion contract: the offsets must stay fractional. Rounding them so
/// tiles composite 1:1 with their texture looks tempting — and is what made
/// the board shudder, since any speed that is not a whole number of pixels per
/// frame then advances unevenly.
enum TileLayout {

    /// Tile index range covering a viewport of `heightPoints` at `scrollY`.
    static func visibleRange(scrollY: Double, heightPoints: Double) -> ClosedRange<Int> {
        let tile = Double(Board.tileHeight)
        let first = Int((scrollY / tile).rounded(.down))
        let last = Int(((scrollY + heightPoints) / tile).rounded(.down))
        return first...max(first, last)
    }

    /// Top edge of tile `index`, in drawable pixels.
    ///
    /// Anchored on `first` and stepped in integers from there, so a board that
    /// has been drifting for hours keeps full precision: only the fractional
    /// remainder ever reaches `Float`.
    static func top(of index: Int, first: Int, scrollY: Double,
                    scale: Float, tilePixelHeight: Int) -> Double {
        let firstTop = Double(first * tilePixelHeight) - scrollY * Double(scale)
        return firstTop + Double((index - first) * tilePixelHeight)
    }
}
