import Foundation

/// A reference type on purpose: the HTML threads one mutable global `S` through
/// every placement and routing helper, and `edgePorts` saves and restores it to
/// derive a seam's ports from a different stream. Passing a class around keeps
/// that structure without `inout` on forty signatures.
final class Rng {
    private var state: Int64

    init(seed: UInt32) {
        state = Int64(seed)
    }

    /// Save / restore point for the seam-port digression.
    var snapshot: Int64 {
        get { state }
        set { state = newValue }
    }

    /// `S = (S * 1664525 + 1013904223) & 0x7fffffff; return S / 0x7fffffff`.
    ///
    /// JS evaluates the product as a double before `&` truncates it. The
    /// largest product is 0x7fffffff * 1664525 ≈ 3.6e15, comfortably inside
    /// 2^53, so the double arithmetic was exact and Int64 reproduces it
    /// exactly. Masking with 0x7fffffff on a value already reduced mod 2^32 is
    /// the same as reducing mod 2^31, which is what this does.
    @inline(__always)
    func unit() -> Double {
        state = (state &* 1664525 &+ 1013904223) & 0x7fff_ffff
        return Double(state) / Double(0x7fff_ffff)
    }

    @inline(__always)
    func float() -> Float { Float(unit()) }

    /// Inclusive integer range. (`ri`)
    @inline(__always)
    func int(_ a: Int, _ b: Int) -> Int {
        a + Int(unit() * Double(b - a + 1))
    }

    @inline(__always)
    func int(_ range: ClosedRange<Int>) -> Int { int(range.lowerBound, range.upperBound) }

    /// Half-open float range. (`rf`)
    @inline(__always)
    func float(_ a: Float, _ b: Float) -> Float {
        a + Float(unit()) * (b - a)
    }

    @inline(__always)
    func float(_ range: ClosedRange<Float>) -> Float { float(range.lowerBound, range.upperBound) }

    /// (`pick`) — `unit()` can return exactly 1.0 once every 2^31 draws, where
    /// JS would hand back `undefined`; clamping keeps that from being a crash.
    @inline(__always)
    /// An even choice. Reads as what it is at the call site, where
    /// `unit() < 0.5` read as an unexplained threshold — and it appeared five
    /// times in the footprint code alone, each time meaning "either way".
    func coinFlip() -> Bool { unit() < 0.5 }

    func pick<T>(_ array: [T]) -> T {
        array[min(array.count - 1, Int(unit() * Double(array.count)))]
    }

    @inline(__always)
    func chance(_ p: Float) -> Bool { unit() < Double(p) }

    /// (`hseed`) — mixes a world and a tile/seam index into a non-zero seed.
    ///
    /// The HTML's first step, `a * 73856093`, overflows 2^53 for large worlds
    /// and silently loses precision in JS. This uses honest 32-bit wrapping
    /// arithmetic instead: statistically equivalent, and since a world is a
    /// fresh random number every launch, nothing observable changes.
    static func hash(_ a: UInt32, _ b: UInt32) -> UInt32 {
        var h = (a &* 73_856_093) ^ (b &* 19_349_663)
        h = (h ^ (h >> 15)) &* 2_654_435_761
        let m = h % 2_147_483_647
        return m == 0 ? 1 : m
    }

    /// Tile and seam indices are unbounded below now that the board grows
    /// upward, so fold a signed index into the hash by bit pattern.
    static func hash(_ a: UInt32, signed b: Int) -> UInt32 {
        hash(a, UInt32(bitPattern: Int32(truncatingIfNeeded: b)))
    }
}
