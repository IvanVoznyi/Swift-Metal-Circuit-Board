import Foundation

/// Where the board is, as a function of time.
///
/// The obvious implementation — `position -= speed * dt` once per frame —
/// inherits every wobble in when `draw` happens to be called, and because each
/// frame builds on the last, the errors accumulate rather than cancel. Solving
/// for the position at a given instant instead means a frame that arrives a
/// little early or late still lands exactly where it belongs.
///
/// The board only ever drifts upward — new tiles arrive at the top — at a
/// constant speed, from an origin that never moves. There is nothing to ease
/// in and nothing to accumulate, so there is no state here at all beyond where
/// the clock started.
struct ScrollClock {
    private let origin: CFTimeInterval

    init(now: CFTimeInterval) {
        origin = now
    }

    /// Board units. Decreasing, unbounded — the board grows upward forever.
    func position(at time: CFTimeInterval) -> Double {
        -Double(Anim.scrollSpeed) * (time - origin)
    }
}
