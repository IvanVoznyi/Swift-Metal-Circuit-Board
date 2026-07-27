import Foundation
import QuartzCore
import os


/// The last time a frame actually reached the screen.
///
/// Metal reports that from `MTLDrawable.addPresentedHandler`, which fires on
/// its own thread, so the value has to cross into the render thread somehow.
/// A lock around a single `Double` is the whole mechanism: the render thread
/// reads it once a frame and never waits on anything that does real work.
/// `Sendable` for real: the one stored property is an immutable `let`, and the
/// value it guards lives *inside* the lock, so there is no way to reach it
/// unguarded.
final class PresentationClock: Sendable {
    private let value = OSAllocatedUnfairLock<CFTimeInterval>(initialState: 0)

    /// `presentedTime` is zero for a drawable that was never shown — a frame
    /// dropped on a resize, say — and passing that on would look like a stall.
    func record(_ presented: CFTimeInterval) {
        guard presented > 0 else { return }
        value.withLock { if presented > $0 { $0 = presented } }
    }

    var latest: CFTimeInterval { value.withLock { $0 } }
}

/// Frame timestamps that follow the display, not the draw call.
///
/// `draw` is called once per vsync, but *when* within that vsync varies — the
/// display link fires, the run loop dispatches, other work intervenes. Feeding
/// `CACurrentMediaTime()` straight into a position function hands all of that
/// wobble to the board.
///
/// The previous version answered that by handing out its own evenly spaced
/// timestamps, which is right only if frames are evenly *displayed*. They are
/// not. Measured off a screen recording of this app, intervals ran 8.3 / 16.7 /
/// 25.0 / 33.3 ms — a 120 Hz panel with the app at 60 — and 23 of 181 frames
/// arrived late. The correlation between elapsed time and how far the board
/// moved on screen was **+0.00**: a frame held up for 25 ms advanced the board
/// exactly as far as one that took 16.7, so its velocity dipped for as long as
/// it was up. That is judder, and evenly spaced timestamps cause it rather than
/// cure it.
///
/// So `Renderer` feeds back `MTLDrawable.presentedTime` and each tick anchors
/// to the frame that actually reached the screen, offset by a learned lead.
/// Motion between two ticks is then exactly the interval those two frames were
/// displayed for. What stays smoothed is the lead — a constant offset on a
/// board that scrolls forever is invisible, so only its wobble matters — and a
/// stall resynchronises hard rather than crawling back.
struct FrameClock {
    private var time: CFTimeInterval = 0
    private var lastSample: CFTimeInterval = 0
    private var interval: CFTimeInterval
    private var started = false

    /// The most recent presentation this clock has been told about, and how far
    /// ahead of it the clock is running.
    ///
    /// The lead itself does not matter — a constant offset in a board that
    /// scrolls forever is invisible — so it is *learned* rather than assumed,
    /// and only its variation is corrected. That is what lets the same code
    /// work with any number of frames in flight.
    private var lastPresented: CFTimeInterval = 0
    /// The presentation this clock is currently anchored to, so a tick with no
    /// new frame on screen can tell that it has nothing to anchor to.
    private var anchor: CFTimeInterval = 0
    private var lead: CFTimeInterval = 0
    private var haveLead = false

    init(refreshRate: Int) {
        interval = 1.0 / Double(max(refreshRate, 1))
    }

    /// Called from the drawable's presented handler, off the render thread's
    /// timeline — `Renderer` funnels it back in before the next `tick`.
    mutating func observePresented(_ presented: CFTimeInterval) {
        guard presented > lastPresented else { return }
        // The real display cadence, which is the thing worth tracking. A frame
        // that missed its deadline shows up here as a longer interval, where
        // sampling the draw call would have hidden it.
        if lastPresented > 0 {
            let sample = presented - lastPresented
            if sample > Anim.minFrameInterval && sample < Anim.maxFrameInterval {
                interval += (sample - interval) * Anim.intervalTracking
            }
        }
        lastPresented = presented
    }

    /// The timestamp this frame should be drawn for.
    mutating func tick(now: CFTimeInterval) -> CFTimeInterval {
        guard started else {
            started = true
            time = now
            lastSample = now
            return now
        }

        // Until a frame has actually been presented there is nothing better to
        // go on, so fall back to the draw call — this is the first frame or two,
        // and the reveal is still holding the board still anyway.
        guard lastPresented > 0 else {
            let sample = now - lastSample
            lastSample = now
            if sample > Anim.minFrameInterval && sample < Anim.maxFrameInterval {
                interval += (sample - interval) * Anim.intervalTracking
            }
            time += interval
            let error = now - time
            if abs(error) > interval * Anim.resyncFrames {
                time = now
            } else {
                time += error * Anim.phaseTracking
            }
            return time
        }
        lastSample = now

        // Nothing new on screen since the last tick — extrapolate, so the board
        // keeps moving rather than freezing for a frame.
        guard lastPresented > anchor else {
            time += interval
            return time
        }

        // Anchor straight to the frame that just landed, offset by a *learned*
        // lead. Two consecutive ticks are then separated by exactly the interval
        // those two frames were displayed for, which is the entire point: a
        // frame held on screen half again as long moves the board half again as
        // far, so its velocity is constant in the only time base the eye has.
        //
        // Easing toward the anchor instead of taking it does not work, and the
        // arithmetic says why: at `phaseTracking` the correction is a few
        // percent a frame, so a late frame is over long before the clock has
        // noticed. Measured, that version modulated by 1.8% where the cadence
        // called for 50%.
        //
        // The lead is what stays smoothed. Its *value* is arbitrary — it is a
        // constant offset on a board that scrolls forever, so it is invisible,
        // and it absorbs however many frames deep the pipeline happens to be.
        // Only its wobble would show, and that is what the low-pass removes.
        let sample = lastPresented - anchor
        if anchor > 0, sample > Anim.minFrameInterval, sample < Anim.maxFrameInterval {
            interval += (sample - interval) * Anim.intervalTracking
        }
        let observed = time - lastPresented
        if haveLead {
            // A stall shows up as a lead nothing like the steady one; take the
            // new reality rather than crawling to it over hundreds of frames.
            if abs(observed - lead) > interval * Anim.resyncFrames {
                lead = interval
            } else {
                lead += (observed - lead) * Anim.intervalTracking
            }
        } else {
            lead = max(observed, interval)
            haveLead = true
        }
        anchor = lastPresented
        time = lastPresented + lead
        return time
    }
}
