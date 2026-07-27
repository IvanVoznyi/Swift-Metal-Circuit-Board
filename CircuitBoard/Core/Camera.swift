import Foundation
import simd

/// The tilted eye looking down the board.
///
/// The board lies on the XZ plane: board x runs across it, board y runs away
/// into the distance, and the scroll carries the whole plane toward the
/// camera. That keeps the existing behaviour exactly — a tile arriving "at the
/// top" is a tile arriving far away — while turning a flat scroll into a strip
/// of circuitry running to a horizon.
enum Camera {
    static let height = Float(PCB_CAMERA_HEIGHT)
    static let fovDegrees = Float(PCB_CAMERA_FOV)
    static let near = Float(PCB_CAMERA_NEAR)
    static let far = Float(PCB_CAMERA_FAR)
    static let cocScale = Float(PCB_COC_SCALE)
    static let cocMax = Float(PCB_COC_MAX)

    /// Downward pitch. Greater than half the field of view, so the top of the
    /// frame looks *at the board* rather than past it into empty sky — the
    /// board runs to every edge of the picture and dissolves into haze on the
    /// way, instead of stopping at a horizon with black above it.
    static let pitchDegrees: Float = 26

    /// How much of the visible width at the near edge the board should span.
    ///
    /// This is the knob for "how wide, and how far up the frame does it keep
    /// running off the sides". Above 1 the board overflows, and it stays
    /// overflowing out to a ground distance of `widthFill × nearGround` — so
    /// the number reads directly as a depth. At 3.4 roughly the nearest half of
    /// the strip has its edges off-screen and they only come into view once the
    /// board is well into the distance.
    ///
    /// Only meaningful because the drawn depth is measured against the camera
    /// rather than in board widths — see `depthOvershoot`.
    ///
    /// It has to clear the ratio between the top and bottom of the frame,
    /// `tan(pitch + fov/2) / tan(pitch - fov/2)` — about 5.6 at the current
    /// angles — or the board's own left and right edges come into view
    /// somewhere up the strip. `CameraTests` checks that at every aspect ratio
    /// worth caring about.
    static let widthFill: Float = 8

    /// Where the plane of focus sits, as a fraction of the way up the strip
    /// *on screen* — 0 at the bottom of the frame, 1 where the board fades out.
    /// Half puts it in the middle of the picture, so the board is sharp through
    /// the centre and falls off into blur both toward the camera and away into
    /// the distance, the way a long lens sees a table top.
    ///
    /// Screen position, not ground distance: under this much perspective the
    /// two are nothing like each other. Half the ground distance to the fade-out
    /// is already three quarters of the way up the frame, so reading this as a
    /// distance leaves the middle of the picture visibly soft and the far end
    /// sharp — the opposite of the intent.
    static let focusFraction: Float = 0.5

    /// How far past the top row of the frame to keep drawing board.
    ///
    /// The strip's length is not a taste decision — it is whatever reaches the
    /// top of the picture, which the pitch and the field of view already decide.
    /// This is the margin on top of that, so the last row of pixels is board
    /// rather than the edge of it. The tile the haze needs to finish in is added
    /// separately, in `setup`, because it is a depth rather than a fraction.
    static let depthOvershoot: Float = 1.06

    /// Tiles the strip may span. A wide window scales the board up, so each
    /// tile covers more ground and fewer are needed; a narrow one needs more.
    /// The clamp stops either extreme asking for an absurd residency — and
    /// because the fog is derived from the clamped depth, biting here shortens
    /// the visible strip rather than truncating it.
    ///
    /// The ceiling is set by the ring: every parallax layer draws its own
    /// stretch of the board, so `(ceiling + 2) × layers` slots have to be
    /// resident at once. `CameraTests` holds that to `Layout.maxCachedTiles`.
    static let depthTileRange: ClosedRange<Double> = 4...12

    /// Everything the shaders need for one frame's view.
    struct Setup {
        let viewProj: matrix_float4x4
        /// World units per board unit. Solved from the aspect ratio so the
        /// board fills the frame the same way on a phone and on a desktop.
        let worldScale: Float
        let eye: SIMD3<Float>
        let focusDistance: Float
        /// Where copper starts dissolving into the haze, and where it is gone.
        let fogStart: Float
        let fogEnd: Float
        /// How far back along the board, in board units, has to stay resident
        /// to fill that depth.
        let depthBoardUnits: Double
        /// Reciprocal slant range where the board meets the bottom and the top
        /// of the frame.
        ///
        /// The board is a plane, so a pixel's distance from the eye follows from
        /// its screen row alone — and reciprocal distance is *linear* down the
        /// frame, which is what lets the depth-of-field pass reconstruct depth
        /// from two numbers instead of an inverse projection or a second render
        /// target.
        let invNear: Float
        let invFar: Float
    }

    /// Fraction of the drawn depth at which the haze starts taking over.
    static let fogStartFraction: Float = 0.45

    static func setup(aspect: Float, boardWidth: Float) -> Setup {
        let fovY = fovDegrees * .pi / 180
        let pitch = pitchDegrees * .pi / 180

        // Ground distance where the bottom of the frame meets the board.
        let bottomAngle = pitch + fovY * 0.5
        let nearGround = height / max(tan(bottomAngle), 1e-3)

        // Half the width visible there, from the horizontal field of view.
        let hHalf = atan(tan(fovY * 0.5) * max(aspect, 1e-3))
        let visibleHalfWidth = nearGround * tan(hHalf)

        // Solve the world scale so the board spans that width.
        let worldScale = max(2 * visibleHalfWidth * widthFill / max(boardWidth, 1), 1e-4)

        // The eye sits above the board's near edge, looking away down the strip.
        let eye = SIMD3<Float>(0, height, nearGround)
        let target = SIMD3<Float>(0, 0, nearGround - height / max(tan(pitch), 1e-3))

        let view = matrix_float4x4.lookAt(eye: eye, center: target, up: SIMD3(0, 1, 0))
        let proj = matrix_float4x4.perspective(fovY: fovY, aspect: aspect,
                                               near: near, far: far)
        // How far down the strip we would like to see: to the top row of the
        // frame and a little past it. A narrow portrait frame scales the board
        // down, so the same distance swallows far more tiles than a desktop
        // window does — hence the clamp below.
        let topAngle = pitch - fovY * 0.5
        let topGround = height / max(tan(topAngle), 1e-3)
        // Reach just past the top row, plus the one tile the haze has to finish
        // in (see `hazedDepth`). That tile is added rather than folded into the
        // multiplier because its depth depends on `worldScale`, which varies
        // with the window: a fixed multiplier that covered a small window drew
        // a needlessly long strip on a large one, and every extra unit of strip
        // is more compressed board on screen than can be resolved.
        let tileDepth = Float(Board.tileHeight) * worldScale
        let idealDepth = max(topGround - nearGround, nearGround) * depthOvershoot
            + tileDepth
        let rawTiles = Double(idealDepth / max(worldScale, 1e-4)) / Double(Board.tileHeight)
        let tiles = min(max(rawTiles, depthTileRange.lowerBound), depthTileRange.upperBound)

        // What is actually drawn is the shorter of the two, which keeps both
        // ends honest. If the ceiling bit, the haze finishes swallowing the
        // board exactly where the resident tiles run out, so the strip is never
        // seen stopping in mid-air. If the floor bit, the extra tiles are
        // harmless slack and the strip stays inside the width `widthFill`
        // guarantees — draw further and the board's own edges come into frame.
        let boardUnits = tiles * Double(Board.tileHeight)
        let depth = min(Float(boardUnits) * worldScale, idealDepth)

        // The haze has to finish one whole tile before the residency limit.
        //
        // A tile is drawn from the moment its index enters the visible range,
        // and it spans a whole `tileHeight` — so the farthest one has its far
        // edge at the limit and its *near* edge one tile closer. End the fog at
        // the limit itself and that near edge is still 27% opaque, so the tile
        // does not fade in: it appears, every time the strip advances by one.
        // At 46 units a second that is every 8.3 seconds, at the top of the
        // frame, and it is what the board was seen blinking at.
        //
        // `depthOvershoot` pays for the tile given up here, which is why it is
        // larger than the bare margin the haze needs.
        let hazedDepth = max(depth - tileDepth, depth * 0.5)

        // Screen row runs with the *reciprocal* of ground distance, so that is
        // what the focus fraction interpolates in. Straight interpolation
        // between the two distances would land the focus plane far up the
        // frame, however innocent the 0.5 looks.
        let farGround = nearGround + hazedDepth
        let focusGround = 1 / ((1 - focusFraction) / nearGround + focusFraction / farGround)

        // The shaders measure from the eye, and the eye is `height` above the
        // board — so every one of these has to be a slant range, not a distance
        // along the ground. Near the front of the strip the two differ by half
        // again, which is the whole depth of field.
        func fromEye(_ ground: Float) -> Float {
            (ground * ground + height * height).squareRoot()
        }

        return Setup(viewProj: proj * view,
                     worldScale: worldScale,
                     eye: eye,
                     focusDistance: fromEye(focusGround),
                     fogStart: fromEye(nearGround + hazedDepth * fogStartFraction),
                     fogEnd: fromEye(farGround),
                     depthBoardUnits: boardUnits,
                     invNear: 1 / fromEye(nearGround),
                     invFar: 1 / fromEye(topGround))
    }

    /// True when the haze has finished with this tile entirely — nothing it
    /// draws can reach the screen, so it can be left out of the frame.
    ///
    /// The farthest drawn tile sits *wholly* beyond `fogEnd` by construction:
    /// that is what stops a tile popping into view as it arrives, and it is the
    /// invariant `hazedDepth` exists to hold. Drawing it anyway shades every one
    /// of its fragments and multiplies each by a fog of zero, once per parallax
    /// layer, every frame.
    ///
    /// Measured against the tile's NEAREST edge — the one the scroll has carried
    /// closest to the eye — so a tile is only dropped when even its closest
    /// point is past where copper has completely dissolved.
    ///
    /// `place` is the tile's position in the strip with its layer's index offset
    /// already removed, which is what the renderer uses to place it.
    static func isFullyHazed(place: Int, scrollY: Double, setup: Setup) -> Bool {
        let nearestBoardY = Double(place + 1) * Double(Board.tileHeight)
        let worldZ = Float(nearestBoardY - scrollY) * setup.worldScale
        let ground = setup.eye.z - worldZ
        // Behind the eye is not "far away"; leave those to the near-edge clamp.
        guard ground > 0 else { return false }
        let fromEye = (ground * ground + setup.eye.y * setup.eye.y).squareRoot()
        return fromEye >= setup.fogEnd
    }

    /// A flat, screen-aligned projection: board units map one-for-one onto
    /// pixels, y downward. Used by the tests, where exact geometry matters more
    /// than the look, and by the flat-view option.
    static func orthographic(pixelWidth: Float, pixelHeight: Float,
                             scale: Float) -> matrix_float4x4 {
        let sx = 2 * scale / pixelWidth
        let sy = -2 * scale / pixelHeight
        return matrix_float4x4(columns: (
            SIMD4(sx, 0, 0, 0),
            SIMD4(0, 0, 0, 0),          // world Y is ignored by the flat view
            SIMD4(0, sy, 0, 0),
            SIMD4(-1, 1, 0, 1)))
    }
}

extension matrix_float4x4 {
    static func perspective(fovY: Float, aspect: Float,
                            near: Float, far: Float) -> matrix_float4x4 {
        let y = 1 / tan(fovY * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        return matrix_float4x4(columns: (
            SIMD4(x, 0, 0, 0),
            SIMD4(0, y, 0, 0),
            SIMD4(0, 0, z, -1),
            SIMD4(0, 0, z * near, 0)))
    }

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>,
                       up: SIMD3<Float>) -> matrix_float4x4 {
        let f = simd_normalize(center - eye)
        let r = simd_normalize(simd_cross(f, up))
        let u = simd_cross(r, f)
        return matrix_float4x4(columns: (
            SIMD4(r.x, u.x, -f.x, 0),
            SIMD4(r.y, u.y, -f.y, 0),
            SIMD4(r.z, u.z, -f.z, 0),
            SIMD4(-simd_dot(r, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)))
    }
}
