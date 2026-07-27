import Foundation
import Metal
import simd

/// Owns the pipelines and draws the board.
///
/// There is no tile texture any more. A tile's instances are expanded once, on
/// arrival, and then drawn straight to the drawable every frame. That removes
/// ~94 MB of texture array, the chunked bake, the slice ring and the composite
/// pass, and it makes the board resolution-independent: the scroll offset is an
/// exact position rather than a resample of a pre-rendered image.
///
/// The trade is honest — steady-state GPU work is now proportional to visible
/// complexity rather than to six textured quads. It is *constant* frame to
/// frame, which is what smoothness actually requires; the old bake was bursty,
/// which is what it did not.
final class BoardPass {
    let device: MTLDevice

    private let expandPSO: MTLComputePipelineState
    private let shapePSO: MTLRenderPipelineState
    private let tracePSO: MTLRenderPipelineState
    private let glyphPSO: MTLRenderPipelineState

    /// The order Canvas drew in, and therefore the order these must blend in.
    /// Drawn stage-major *across* tiles rather than tile-major: a seam trace
    /// overhangs its tile by a few units, so a neighbour's halo drawn later
    /// would otherwise darken this tile's copper.
    enum Stage: CaseIterable {
        case glyphs, traceHalo, traceCore, solids
    }

    init?(device: MTLDevice, colorFormat: MTLPixelFormat) {
        self.device = device
        // `makeDefaultLibrary()` resolves against Bundle.main, which is the
        // xctest runner when these pipelines are built from a test bundle —
        // fall back to the bundle this class actually ships in.
        let library = device.makeDefaultLibrary()
            ?? (try? device.makeDefaultLibrary(bundle: Bundle(for: BoardPass.self)))
        guard let library else { return nil }

        guard let expandFunction = library.makeFunction(name: "kernel_expand_traces"),
              let expand = try? device.makeComputePipelineState(function: expandFunction)
        else { return nil }
        expandPSO = expand

        func render(_ vfn: String, _ ffn: String) -> MTLRenderPipelineState? {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: vfn)
            d.fragmentFunction = library.makeFunction(name: ffn)
            let a = d.colorAttachments[0]!
            a.pixelFormat = colorFormat
            a.isBlendingEnabled = true
            a.rgbBlendOperation = .add
            a.alphaBlendOperation = .add
            // Premultiplied source-over: every fragment shader here multiplies
            // its colour by coverage, and returns a zero-alpha fragment where
            // it would once have discarded.
            a.sourceRGBBlendFactor = .one
            a.destinationRGBBlendFactor = .oneMinusSourceAlpha
            a.sourceAlphaBlendFactor = .one
            a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            return try? device.makeRenderPipelineState(descriptor: d)
        }
        guard let sp = render("shape_vertex", "shape_fragment"),
              let tp = render("trace_vertex", "trace_fragment"),
              let gp = render("glyph_vertex", "glyph_fragment") else { return nil }
        shapePSO = sp; tracePSO = tp; glyphPSO = gp
    }

    // MARK: - Expansion, once per tile

    /// Turns a tile's centrelines into capsule instances. One small dispatch
    /// per tile, run when the tile arrives — the only GPU work that is not
    /// per-frame.
    func encodeExpansion(slot: Int, ring: TileRing, in command: MTLCommandBuffer) {
        let counts = ring.counts(of: slot)
        guard counts.pathPoints > 0,
              let enc = command.makeComputeCommandEncoder() else { return }
        enc.label = "trace expansion"
        enc.setComputePipelineState(expandPSO)
        enc.setBuffer(ring.pathPoints(slot), offset: 0, index: Int(PCBBufferPoints.rawValue))
        enc.setBuffer(ring.traceInfos(slot), offset: 0, index: Int(PCBBufferTraces.rawValue))
        enc.setBuffer(ring.traceVertices(slot), offset: 0,
                      index: Int(PCBBufferTraceVerts.rawValue))
        var count = UInt32(counts.pathPoints)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.stride,
                     index: Int(PCBBufferPass.rawValue))
        let w = expandPSO.maxTotalThreadsPerThreadgroup
        enc.dispatchThreads(MTLSize(width: counts.pathPoints, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(w, 64), height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Drawing

    /// Binds the pipeline and pass constants for a stage. Called once per
    /// stage, then `draw` once per tile.
    func begin(_ stage: Stage, halo: SIMD4<Float>, _ enc: MTLRenderCommandEncoder) {
        switch stage {
        case .glyphs:
            enc.setRenderPipelineState(glyphPSO)
        case .traceHalo:
            enc.setRenderPipelineState(tracePSO)
            var pass = PCBPassUniforms(expand: Draw.haloHalfExtra, useOverride: 1,
                                       or_: halo.x, og: halo.y,
                                       ob: halo.z, oa: halo.w,
                                       _pad0: 0, _pad1: 0)
            setPass(enc, &pass)
        case .traceCore:
            enc.setRenderPipelineState(tracePSO)
            var pass = PCBPassUniforms(expand: 0, useOverride: 0,
                                       or_: 0, og: 0, ob: 0, oa: 0, _pad0: 0, _pad1: 0)
            setPass(enc, &pass)
        case .solids:
            enc.setRenderPipelineState(shapePSO)
            var pass = PCBPassUniforms(expand: 0, useOverride: 0,
                                       or_: 0, og: 0, ob: 0, oa: 0, _pad0: 0, _pad1: 0)
            setPass(enc, &pass)
        }
    }

    /// Draws one tile's contribution to the current stage.
    func draw(_ stage: Stage, slot: Int, ring: TileRing,
              view: inout PCBViewUniforms, in enc: MTLRenderCommandEncoder) {
        let counts = ring.counts(of: slot)
        let buffer: MTLBuffer
        let instanceCount: Int
        switch stage {
        case .glyphs:
            guard counts.glyphs > 0 else { return }
            buffer = ring.glyphs(slot)
            instanceCount = counts.glyphs
            enc.setFragmentTexture(ring.atlas(slot), index: 0)
        case .traceHalo, .traceCore:
            // Every trace in the tile, stitched into ONE strip by the degenerate
            // vertices the expansion kernel writes at each trace's ends. No
            // instancing, no per-segment quads, and so no overlap for a
            // not-quite-opaque stroke to composite twice.
            guard counts.traceVertices > 0 else { return }
            enc.setVertexBuffer(ring.traceVertices(slot), offset: 0,
                                index: Int(PCBBufferTraceVerts.rawValue))
            // The per-trace record: colour and the light runner's clock, read by
            // index rather than copied onto every corner of the ribbon.
            enc.setVertexBuffer(ring.traceInfos(slot), offset: 0,
                                index: Int(PCBBufferTraces.rawValue))
            enc.setVertexBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                               index: Int(PCBBufferView.rawValue))
            enc.setFragmentBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                                 index: Int(PCBBufferView.rawValue))
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0,
                               vertexCount: counts.traceVertices)
            return
        case .solids:
            guard counts.solids > 0 else { return }
            buffer = ring.solids(slot)
            instanceCount = counts.solids
        }
        enc.setVertexBuffer(buffer, offset: 0, index: Int(PCBBufferInstances.rawValue))
        enc.setFragmentBuffer(buffer, offset: 0, index: Int(PCBBufferInstances.rawValue))
        enc.setVertexBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                           index: Int(PCBBufferView.rawValue))
        enc.setFragmentBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                             index: Int(PCBBufferView.rawValue))
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                           instanceCount: instanceCount)
    }

    /// Draws only the pads of a tile — the suffix of its solids buffer.
    /// Bind with `begin(.solids, …)` first; this shares that state.
    func drawPads(slot: Int, ring: TileRing,
                  view: inout PCBViewUniforms, in enc: MTLRenderCommandEncoder) {
        let counts = ring.counts(of: slot)
        let count = counts.solids - counts.padStart
        guard count > 0 else { return }
        let offset = counts.padStart * MemoryLayout<PCBShapeInstance>.stride
        let buffer = ring.solids(slot)
        enc.setVertexBuffer(buffer, offset: offset, index: Int(PCBBufferInstances.rawValue))
        enc.setFragmentBuffer(buffer, offset: offset, index: Int(PCBBufferInstances.rawValue))
        enc.setVertexBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                           index: Int(PCBBufferView.rawValue))
        enc.setFragmentBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                             index: Int(PCBBufferView.rawValue))
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                           instanceCount: count)
    }

    /// Lays one tile's substrate down as an opaque slab, before any of its
    /// copper.
    ///
    /// Without it a parallax layer is a floating mesh of wires and you see
    /// straight through to the layer beneath, which reads as three boards
    /// sharing one space rather than as a stack. With it each layer occludes
    /// the ones below, exactly as a real inner layer does, and the offsets
    /// between them become visible as edges instead of as clutter.
    ///
    /// The slabs **overlap** rather than abut, by `Draw.substrateOverlap` at each
    /// end.
    ///
    /// Two antialiased quads sharing an edge each cover about half of the pixels
    /// on it, so a quarter of whatever is behind shows through as a hairline —
    /// measured, 64 of 255. Standing still that is a line nobody notices; it
    /// scrolls with the board, and under a wide blur it smears into a moving
    /// band of the layer behind. Overlapping is free to do because a layer's
    /// slabs are all one opaque colour and are all drawn before any of its
    /// copper, so doubling up changes nothing.
    ///
    /// One instance, passed inline: a single quad, so there is nothing worth
    /// keeping in a buffer.
    func drawSubstrate(width: Float, colour: SIMD3<Float>,
                       view: inout PCBViewUniforms, in enc: MTLRenderCommandEncoder) {
        let overlap = Draw.substrateOverlap
        var slab = Shape.roundRect(width / 2, Board.tileHeight / 2,
                                   width, Board.tileHeight + 2 * overlap,
                                   radius: 0, color: SIMD4(colour, 1))
        enc.setVertexBytes(&slab, length: MemoryLayout<PCBShapeInstance>.stride,
                           index: Int(PCBBufferInstances.rawValue))
        enc.setFragmentBytes(&slab, length: MemoryLayout<PCBShapeInstance>.stride,
                             index: Int(PCBBufferInstances.rawValue))
        enc.setVertexBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                           index: Int(PCBBufferView.rawValue))
        enc.setFragmentBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                             index: Int(PCBBufferView.rawValue))
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                           instanceCount: 1)
    }

    private func setPass(_ enc: MTLRenderCommandEncoder, _ pass: inout PCBPassUniforms) {
        enc.setVertexBytes(&pass, length: MemoryLayout<PCBPassUniforms>.stride,
                           index: Int(PCBBufferPass.rawValue))
        enc.setFragmentBytes(&pass, length: MemoryLayout<PCBPassUniforms>.stride,
                             index: Int(PCBBufferPass.rawValue))
    }
}
