import Foundation
import Metal
import simd

/// The HDR chain: a background gradient, then bloom, then an ACES tonemap onto
/// the drawable.
///
/// The board renders into an `rgba16Float` target rather than straight to the screen, so copper can be
/// brighter than white and the bloom has something real to pick up; the
/// composite pass brings it back into display range with exposure, saturation
/// and a filmic roll-off instead of a hard clip.
final class PostPass {
    private let device: MTLDevice
    private let background: MTLRenderPipelineState
    private let prefilter: MTLRenderPipelineState
    private let blurH: MTLRenderPipelineState
    private let blurV: MTLRenderPipelineState
    private let composite: MTLRenderPipelineState
    private let dofH: MTLRenderPipelineState
    private let dofV: MTLRenderPipelineState

    static let hdrFormat: MTLPixelFormat = .rgba16Float

    /// Scene target plus the half-resolution ping-pong pair the blur bounces
    /// between. Rebuilt only on resize.
    private struct Targets {
        let scene: MTLTexture
        /// The scene after the horizontal half of the blur.
        let halfBlurred: MTLTexture
        /// The scene after both halves. Everything downstream — bloom included
        /// — reads this, so the glow blooms the defocused image rather than a
        /// sharp one nobody sees.
        let focused: MTLTexture
        let bloomA: MTLTexture
        let bloomB: MTLTexture
        /// Ping-pong for temporal accumulation: one frame's result is the next
        /// frame's history. Full resolution and HDR, because accumulating has to
        /// happen in linear light and before the blur spreads anything.
        let historyA: MTLTexture
        let historyB: MTLTexture
    }
    private var targets: Targets?
    /// Which history texture holds the last accumulated frame.
    private var historyFlip = false
    /// False until one frame has been accumulated, and after every resize.
    private var historyReady = false
    private let temporalPSO: MTLRenderPipelineState

    private(set) var sceneTexture: MTLTexture?

    init?(device: MTLDevice, drawableFormat: MTLPixelFormat) {
        self.device = device
        let library = device.makeDefaultLibrary()
            ?? (try? device.makeDefaultLibrary(bundle: Bundle(for: PostPass.self)))
        guard let library else { return nil }

        func make(_ vertex: String, _ fragment: String, format: MTLPixelFormat)
            -> MTLRenderPipelineState? {
            let d = MTLRenderPipelineDescriptor()
            d.label = vertex
            d.vertexFunction = library.makeFunction(name: vertex)
            d.fragmentFunction = library.makeFunction(name: fragment)
            d.colorAttachments[0].pixelFormat = format
            return try? device.makeRenderPipelineState(descriptor: d)
        }
        guard let temporal = make("fullscreenVertex", "temporalAccumulate", format: Self.hdrFormat),
              let bg = make("fullscreenVertex", "backgroundFragment", format: Self.hdrFormat),
              let dofHPSO = make("fullscreenVertex", "depthOfFieldHorizontal", format: Self.hdrFormat),
              let dofVPSO = make("fullscreenVertex", "depthOfFieldVertical", format: Self.hdrFormat),
              let pre = make("fullscreenVertex", "bloomPrefilter", format: Self.hdrFormat),
              let bh = make("fullscreenVertex", "bloomBlurHorizontal", format: Self.hdrFormat),
              let bv = make("fullscreenVertex", "bloomBlurVertical", format: Self.hdrFormat),
              let comp = make("fullscreenVertex", "compositeFragment", format: drawableFormat)
        else { return nil }
        self.temporalPSO = temporal
        self.background = bg
        self.dofH = dofHPSO
        self.dofV = dofVPSO
        self.prefilter = pre
        self.blurH = bh
        self.blurV = bv
        self.composite = comp
    }

    // MARK: - Targets

    /// Returns true if the targets were rebuilt.
    @discardableResult
    func resize(width: Int, height: Int) -> Bool {
        guard width > 0, height > 0 else { return false }
        if let t = targets, t.scene.width == width, t.scene.height == height { return false }

        func texture(_ w: Int, _ h: Int, mipmapped: Bool = false) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: Self.hdrFormat, width: max(w, 1), height: max(h, 1),
                mipmapped: mipmapped)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        // No MIP chain: the blur reads the scene at full resolution, which is
        // what keeps it from crawling as the board slides underneath it.
        guard let scene = texture(width, height),
              let half = texture(width, height),
              let focused = texture(width, height),
              let a = texture(width / 2, height / 2),
              let b = texture(width / 2, height / 2),
              let h0 = texture(width, height),
              let h1 = texture(width, height) else { return false }
        targets = Targets(scene: scene, halfBlurred: half, focused: focused,
                          bloomA: a, bloomB: b, historyA: h0, historyB: h1)
        // Nothing to carry forward across a resize.
        historyReady = false
        sceneTexture = scene
        return true
    }

    // MARK: - Passes

    /// Opens the scene pass. The board draws into the returned encoder; the
    /// background gradient is already laid down.
    func beginScene(in command: MTLCommandBuffer) -> MTLRenderCommandEncoder? {
        guard let targets else { return nil }
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = targets.scene
        d.colorAttachments[0].loadAction = .dontCare   // the gradient covers every pixel
        d.colorAttachments[0].storeAction = .store
        guard let enc = command.makeRenderCommandEncoder(descriptor: d) else { return nil }
        enc.label = "scene"
        enc.setRenderPipelineState(background)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        return enc
    }

    /// Bloom, then tonemap onto `drawableTarget`. Call after the scene encoder
    /// has ended.
    func finish(into drawableTarget: MTLRenderPassDescriptor,
                options: PostOptions, temporal: TemporalFrame? = nil,
                in command: MTLCommandBuffer) {
        guard let targets else { return }

        var post = PCBPostUniforms(bloomIntensity: options.bloomIntensity,
                                   exposure: options.exposure,
                                   saturation: options.saturation,
                                   dofStrength: options.dofStrength,
                                   invNear: options.invNear,
                                   invFar: options.invFar,
                                   focusDist: options.focusDistance,
                                   cocScale: Camera.cocScale,
                                   cocMax: Camera.cocMax,
                                   invViewProj: temporal?.invViewProj
                                       ?? matrix_float4x4(),
                                   boardDelta: temporal?.boardDelta ?? 0,
                                   historyBlend: options.historyBlend,
                                   historyValid: 0,
                                   _pad0: 0)

        // ── Temporal accumulation, before anything spreads ──────────────────
        // Ahead of the blur on purpose: the shimmer being averaged out is in the
        // scene pass, and once the depth of field has smeared it across a
        // twenty-six pixel radius it is no longer something a neighbourhood
        // clamp can recognise or reject.
        var accumulated = targets.scene
        if let temporal {
            let history = historyFlip ? targets.historyB : targets.historyA
            let destination = historyFlip ? targets.historyA : targets.historyB
            post.historyValid = historyReady ? 1 : 0
            blit(temporalPSO, from: [targets.scene, history], into: destination, label: "temporal reproject",
                 uniforms: &post, view: temporal.view, in: command)
            accumulated = destination
            historyFlip.toggle()
            historyReady = true
        } else {
            historyReady = false
        }

        // The blur, across then down. Skipped outright when there is no depth of
        // field — the flat view — rather than run with a zero radius.
        let lit = options.dofStrength > 0
        if lit {
            blit(dofH, from: [accumulated], into: targets.halfBlurred, label: "depth of field · horizontal",
                 uniforms: &post, in: command)
            blit(dofV, from: [targets.halfBlurred], into: targets.focused, label: "depth of field · vertical",
                 uniforms: &post, in: command)
        }
        let composited = lit ? targets.focused : accumulated

        blit(prefilter, from: [composited], into: targets.bloomA, label: "bloom prefilter", in: command)
        blit(blurH, from: [targets.bloomA], into: targets.bloomB, label: "bloom blur · horizontal", in: command)
        blit(blurV, from: [targets.bloomB], into: targets.bloomA, label: "bloom blur · vertical", in: command)

        guard let enc = command.makeRenderCommandEncoder(descriptor: drawableTarget) else { return }
        enc.label = "composite"
        enc.setRenderPipelineState(composite)
        enc.setFragmentTexture(composited, index: 0)
        enc.setFragmentTexture(targets.bloomA, index: 1)
        enc.setFragmentBytes(&post, length: MemoryLayout<PCBPostUniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    /// Forget the accumulated frame. Only the tests need this — the app resets
    /// history on resize, which is the one time it is meaningless.
    func resetHistoryForTesting() { historyReady = false }

    /// `label` is not decoration: without it a GPU capture shows this chain as
    /// "Render Command 1…5" and there is no way to tell the bloom prefilter
    /// from the depth-of-field blur. The scene pass costs 295 µs; these five
    /// cost 5.7 ms between them, so they are the ones worth reading.
    private func blit(_ pipeline: MTLRenderPipelineState, from sources: [MTLTexture],
                      into target: MTLTexture, label: String = "post",
                      uniforms: UnsafeMutablePointer<PCBPostUniforms>? = nil,
                      view: PCBViewUniforms? = nil,
                      in command: MTLCommandBuffer) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = target
        d.colorAttachments[0].loadAction = .dontCare
        d.colorAttachments[0].storeAction = .store
        guard let enc = command.makeRenderCommandEncoder(descriptor: d) else { return }
        enc.label = label
        enc.setRenderPipelineState(pipeline)
        for (i, texture) in sources.enumerated() { enc.setFragmentTexture(texture, index: i) }
        if let uniforms {
            enc.setFragmentBytes(uniforms, length: MemoryLayout<PCBPostUniforms>.stride,
                                 index: 0)
        }
        if var view {
            // The camera, for the temporal pass: it un-projects a pixel onto the
            // board plane and projects it again a frame earlier.
            enc.setFragmentBytes(&view, length: MemoryLayout<PCBViewUniforms>.stride,
                                 index: 1)
        }
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }
}

/// Grading knobs for the composite.
struct PostOptions: Equatable {
    /// 0 composites the scene without any bloom at all. The prefilter and blur
    /// still run — they are a fixed few hundred microseconds and skipping them
    /// would mean a second composite pipeline for no measurable gain.
    var bloomIntensity = Float(PCB_BLOOM_INTENSITY)
    var exposure = Float(PCB_EXPOSURE)
    var saturation = Float(PCB_SATURATION)

    /// Depth of field. 0 is off, which is what the flat view and the tests use.
    var dofStrength: Float = 0
    /// Reciprocal slant range at the bottom and top of the frame — the whole of
    /// the screen-row-to-distance mapping, because the board is a plane.
    var invNear: Float = 1
    var invFar: Float = 1
    var focusDistance: Float = 1

    /// How much of each new frame the temporal pass takes. The neighbourhood
    /// clamp is what makes a number this low safe — see `temporalAccumulate`.
    var historyBlend = Float(PCB_HISTORY_BLEND)
}

/// What the temporal pass needs that changes every frame.
///
/// Kept out of `PostOptions` deliberately: those are *settings*, compared for
/// equality to decide whether anything the user chose has changed. A camera and
/// a scroll delta are neither settings nor comparable, and folding them in would
/// mean every frame looked like a settings change.
struct TemporalFrame {
    /// The camera, whole: the pass needs the projection, its inverse and the eye.
    var view: PCBViewUniforms
    var invViewProj: matrix_float4x4
    /// World units the board has slid toward the camera since the last frame.
    var boardDelta: Float
}
