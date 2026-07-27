import Foundation
import Metal

/// Turns a generated tile into GPU-resident data by writing straight into a
/// slot the ring already owns.
///
/// Nothing here allocates a Metal resource. The previous version built six
/// buffers and a texture per tile on a worker thread; resource creation takes
/// driver locks and GPU virtual memory, and doing it alongside the render
/// thread's encoding is a good way to lose a frame. A worker now only
/// `memcpy`s into memory that has existed since startup.

/// Reusable per-worker scratch. Pooled alongside the generators so a tile
/// costs no allocation at all on the worker thread.
final class TileScratch {
    var glyphCanvas: [UInt8] = []
}

struct TileFiller {
    let geo: BoardGeometry
    let scale: Float
    let teardrops: Bool
    let palette: Palette
    let glow: Bool

    /// Two fillers are interchangeable when they would produce identical data.
    func matches(_ other: TileFiller?) -> Bool {
        guard let other else { return false }
        return geo == other.geo && scale == other.scale && teardrops == other.teardrops
            && palette == other.palette && glow == other.glow
    }

    /// Encodes `data` into `slot`'s buffers and returns the element counts.
    /// Anything that would overrun a capacity is truncated — losing a few
    /// instances off the busiest board is a far better trade than allocating
    /// on the wrong thread, and in practice the headroom is ~2.5×.
    func fill(_ data: TileData, slot: Int, ring: TileRing,
              scratch: TileScratch) -> TileRing.Counts {
        let entries = GlyphAtlas.render(data.numbers, scale: scale,
                                        into: ring.atlas(slot),
                                        capacity: ring.capacity.glyphs,
                                        canvas: &scratch.glyphCanvas)
        let encoder = InstanceEncoder(geo: geo, scale: scale, teardrops: teardrops,
                                      glow: glow, palette: palette,
                                      glyphEntries: entries)
        let instances = encoder.encode(data)

        var counts = TileRing.Counts()
        counts.pathPoints = copy(instances.pathPoints, into: ring.pathPoints(slot),
                                 capacity: ring.capacity.pathPoints)
        counts.traceInfos = copy(instances.traceInfos, into: ring.traceInfos(slot),
                                 capacity: ring.capacity.traceInfos)
        counts.solids = copy(instances.solids, into: ring.solids(slot),
                             capacity: ring.capacity.solids)
        // Clamped in case the copy truncated before reaching the pads.
        counts.padStart = min(instances.padStart, counts.solids)
        counts.glyphs = copy(instances.glyphs, into: ring.glyphs(slot),
                             capacity: ring.capacity.glyphs)

        // A truncated path-point list would leave the last trace's final point
        // without its `last` flag, so the expansion kernel would emit a segment
        // reaching into the next trace. Trim back to a clean trace boundary.
        if counts.pathPoints < instances.pathPoints.count {
            counts.pathPoints = trimToTraceBoundary(ring.pathPoints(slot),
                                                    count: counts.pathPoints)
        }

        // How far into the ribbon buffer the kept points reach. Derived from the
        // last surviving point rather than from `instances.traceVertexCount`,
        // because the trim above may have dropped whole traces off the end and
        // the strip must not be told to draw vertices nobody wrote.
        //
        // A trace's block is `dup, (left, right) per point, dup`, so the last
        // point's own pair plus its trailing duplicate is `vertexBase + 3`.
        if counts.pathPoints > 0 {
            let points = ring.pathPoints(slot).contents()
                .bindMemory(to: PCBPathPoint.self, capacity: counts.pathPoints)
            let last = points[counts.pathPoints - 1]
            counts.traceVertices = min(Int(last.vertexBase) + 3,
                                       ring.capacity.traceVertices)
        }
        return counts
    }

    private func copy<T>(_ source: [T], into buffer: MTLBuffer, capacity: Int) -> Int {
        let n = min(source.count, capacity)
        guard n > 0 else { return 0 }
        source.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!,
                                         byteCount: n * MemoryLayout<T>.stride)
        }
        return n
    }

    /// Walks back to the last point flagged `PCB_POINT_LAST`, so the buffer
    /// always ends on a complete trace.
    private func trimToTraceBoundary(_ buffer: MTLBuffer, count: Int) -> Int {
        let points = buffer.contents().bindMemory(to: PCBPathPoint.self, capacity: count)
        var n = count
        while n > 0 && (points[n - 1].flags & PCB_POINT_LAST) == 0 { n -= 1 }
        return n
    }
}
