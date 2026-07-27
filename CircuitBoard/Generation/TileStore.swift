import Foundation
import Metal

/// Runs tile generation off the render thread and writes the result straight
/// into a ring slot.
///
/// Routing cannot be parallelised inside a tile — every `claim` changes the
/// board the next route sees — but tiles are independent by construction, so
/// the concurrency lives here.
///
/// **Work is pulled, not pushed.** The renderer asks for thirty tiles a frame
/// and only a handful may run at once, so the obvious shape — dispatch all
/// thirty and let a semaphore hold most of them back — puts twenty-odd blocked
/// threads on a concurrent queue. That is thread explosion, and libdispatch's
/// answer to it is to stop widening the pool; measured, the startup set went
/// from finishing in about a second to not finishing inside twelve. Instead the
/// wanted list is kept here and exactly as many tasks are in flight as there are
/// workers, each one pulling the next tile when it finishes.
///
/// **Two lanes**, because launch and cruise want opposite things. Once the board
/// is up a tile is needed every few seconds and there is a minute of slack, so a
/// few `.utility` workers on efficiency cores are right and anything hungrier
/// only takes cores off the render thread. At launch nothing is on screen at
/// all, so politeness buys nothing and costs the user the wait.
///
/// The lock guards nothing but a small set of Ints and the pending list.
/// Generator construction, routing and every allocation happen outside it, so
/// the render thread never waits on a worker doing real work.
/// Safe to hand across threads by construction: every mutable field is guarded
/// by `lock`, and nothing is published outside it.
///
/// This is the one place the lock-owns-the-state pattern `TileRing` uses does
/// not fit, and it is worth saying why rather than leaving it looking like an
/// oversight. `OSAllocatedUnfairLock<State>` requires `State: Sendable`, and the
/// state here includes `pool`, which holds `TileGenerator` and `TileScratch` —
/// mutable classes, deliberately so, because they carry the grid-sized buffers
/// that make reusing a generator worth anything. They are *transferred* between
/// workers rather than shared, which is a sound argument the compiler has no way
/// to express short of `sending`, so moving them into a lock would only push the
/// same `@unchecked` down onto two more types instead of one.
final class TileStore: @unchecked Sendable {
    private var params: BoardParams
    private var filler: TileFiller?
    private var inFlight: Set<Int> = []
    private var pool: [(TileGenerator, TileScratch)] = []

    /// Tiles still wanted, nearest to the viewport first. Rebuilt every frame,
    /// so a tile that has drifted out of view is dropped before it is started
    /// rather than after.
    private var pending: [Int] = []
    /// Tiles the ring may not evict, carried alongside `pending` because a
    /// worker claims its slot when it starts, not when it was asked for.
    private var keep: Set<Int> = []
    private var running = 0

    private let lock = NSLock()

    /// A queue and how many tiles may be generating on it at once.
    private struct Lane {
        let queue: DispatchQueue
        let workers: Int
    }

    /// Steady state: efficiency cores, out of the render thread's way.
    private let cruise = Lane(
        queue: DispatchQueue(label: "com.pcb.tiles",
                             qos: .utility, attributes: .concurrent),
        workers: Layout.generatorConcurrency)

    /// Launch: performance cores, as many as the machine has to spare. This is
    /// the one lever on time-to-first-frame that costs nothing visible, because
    /// it costs nothing while nothing is visible.
    private let burst = Lane(
        queue: DispatchQueue(label: "com.pcb.tiles.burst",
                             qos: .userInitiated, attributes: .concurrent),
        workers: Layout.burstConcurrency)

    private var bursting = false
    private let ring: TileRing

    init(params: BoardParams, ring: TileRing) {
        self.params = params
        self.ring = ring
    }

    /// Route work down the launch lane until the board is first complete.
    func beginBurst() {
        lock.lock(); bursting = true; lock.unlock()
    }

    /// Back to the polite lane. Idempotent — the renderer latches its reveal,
    /// but this must not care how often it is called.
    func endBurst() {
        lock.lock(); bursting = false; lock.unlock()
    }

    /// Any change to the sliders, the world or the board width invalidates
    /// every tile. Returns true if anything was thrown away.
    @discardableResult
    func update(_ new: BoardParams, filler newFiller: TileFiller?) -> Bool {
        lock.lock()
        let sameFiller = filler?.matches(newFiller) ?? (newFiller == nil)
        guard new != params || !sameFiller else { lock.unlock(); return false }
        params = new
        filler = newFiller
        pool.removeAll(keepingCapacity: true)
        inFlight.removeAll(keepingCapacity: true)
        pending.removeAll(keepingCapacity: true)
        lock.unlock()
        // Bumps the ring's epoch, so results already in flight are dropped
        // rather than written into a slot someone else now owns.
        ring.invalidate()
        return true
    }

    /// True until a filler has been handed over — the renderer only knows the
    /// drawable scale once it has drawn a frame.
    var needsFiller: Bool {
        lock.lock(); defer { lock.unlock() }
        return filler == nil
    }

    /// Takes this frame's wanted list, nearest to the viewport first, and gets
    /// as much of it started as there are workers free.
    func request(_ indices: [Int], keeping keep: Set<Int>) {
        lock.lock()
        self.keep = keep
        pending.removeAll(keepingCapacity: true)
        for index in indices where !inFlight.contains(index) { pending.append(index) }
        lock.unlock()
        pump()
    }

    // MARK: - The pump

    /// Starts tiles until the lane is full or there is nothing left to start.
    /// Called from the render thread when the wanted list changes, and from a
    /// worker as it finishes — that second call is what keeps the lane busy.
    private func pump() {
        while true {
            lock.lock()
            guard let currentFiller = filler, running < currentLane.workers,
                  !pending.isEmpty else { lock.unlock(); return }
            let index = pending.removeFirst()
            guard !inFlight.contains(index) else { lock.unlock(); continue }
            let currentParams = params
            let lane = currentLane
            let currentKeep = keep
            inFlight.insert(index)
            running += 1
            lock.unlock()

            // Already there — nothing to do but release the accounting and move
            // on to the next one.
            guard !ring.isResident(index) else { release(index); continue }

            // No evictable slot: the ring is full of tiles that are on screen
            // or already generating. Give up for this frame rather than
            // spinning through the rest of the list to be told the same thing.
            guard let claim = ring.claim(index, keeping: currentKeep) else {
                release(index)
                return
            }

            lane.queue.async { [weak self] in
                guard let self else { return }
                defer { self.release(index); self.pump() }

                // Taken from the pool, or built here — outside the lock, since
                // a generator allocates five grid-sized buffers.
                self.lock.lock()
                let pooled = self.pool.popLast()
                self.lock.unlock()
                // The detail this tile's own layer draws. The inner layers show
                // copper and nothing else, so generating their silkscreen and
                // footprints is work that is thrown away — and it is a large
                // share of the wait before the board first appears.
                let tileParams = currentParams
                    .detailed(for: Parallax.layer(owning: index))
                let scratch = pooled?.1 ?? TileScratch()
                var generator = pooled?.0 ?? TileGenerator(params: tileParams)
                // A pooled generator was built for some other layer's counts.
                // Retuning keeps its grid buffers, which are sized by geometry
                // and are the expensive part; only a geometry change forces a
                // new one, and that cannot happen here because a change to it
                // empties the pool.
                if !generator.retune(to: tileParams) {
                    generator = TileGenerator(params: tileParams)
                }

                let data = generator.generate(index: index)
                let counts = currentFiller.fill(data, slot: claim.slot, ring: self.ring,
                                                scratch: scratch)

                let accepted = self.ring.markFilled(claim.slot, index: index,
                                                    counts: counts, epoch: claim.epoch)
                if !accepted {
                    self.ring.abandon(claim.slot, index: index, epoch: claim.epoch)
                }
                self.lock.lock()
                self.pool.append((generator, scratch))
                self.lock.unlock()
            }
        }
    }

    /// Caller must hold the lock.
    private var currentLane: Lane { bursting ? burst : cruise }

    private func release(_ index: Int) {
        lock.lock()
        inFlight.remove(index)
        running -= 1
        lock.unlock()
    }
}
