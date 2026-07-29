//
//  PCBBridge.h
//  The single contract shared by Swift and Metal.
//
//  Included from Swift through PCB-Bridging-Header.h and from every
//  .metal source directly, so a struct layout or a magic number can never
//  drift between the two languages. Swift re-exports the numeric macros
//  through the namespace enums in Core/Constants.swift; nothing in Swift or
//  Metal should ever spell one of these values out again.
//
//  Every numeric constant is a plain `#define` with an unsuffixed literal:
//  the Clang importer maps those to Swift `Double`/`Int32`, and MSL treats an
//  unsuffixed floating literal as `float`. A `static const` at file scope
//  would not survive the trip (MSL requires an address-space qualifier).
//

#ifndef PCBBridge_h
#define PCBBridge_h

// Matrices are the one place the two languages spell a type differently.
#ifdef __METAL_VERSION__
#define PCB_FLOAT4X4 metal::float4x4
#else
#include <simd/simd.h>
#define PCB_FLOAT4X4 matrix_float4x4
#endif

// ─── Board metrics ────────────────────────────────────────────────────────
// TILE_H is an exact multiple of GS so row 0 of a tile lines up with the last
// row of its neighbour — that is what lets traces cross the seam.
#define PCB_GRID_SIZE        8.0     // GS
#define PCB_ROWS             48      // ROWS
#define PCB_TILE_HEIGHT      384.0   // ROWS * GS
#define PCB_CLEARANCE        2.2     // copper→copper clearance, px
#define PCB_ORIGIN_Y         4.0     // OFY = GS/2
#define PCB_REFERENCE_WIDTH  800.0   // slider counts scale by W / this

// ─── Trace classes (stroke widths, px) ────────────────────────────────────
#define PCB_W_MAIN           6.6
#define PCB_W_BUS            3.4
#define PCB_W_SIGNAL         2.6
#define PCB_W_FINE           1.9

// ─── Routing ──────────────────────────────────────────────────────────────
#define PCB_HUG_RADIUS       3       // HUGR: graded hug field depth
#define PCB_TURN_45          0.9
#define PCB_TURN_90          2.4
#define PCB_ASTAR_BUDGET     26000
#define PCB_HUG_MIN_MUL      0.55    // 1 - 0.45, the hug slider's floor

// ─── Placement rates ──────────────────────────────────────────────────────
#define PCB_CHIP_RATE        0.14
#define PCB_DIPCHIP_RATE     0.154   // CHIP_RATE * 1.1
#define PCB_DECOR_RATE       0.12
#define PCB_MEANDER_RATE     0.6

// ─── Drawing ──────────────────────────────────────────────────────────────
// Mitre limit, in half-widths. A joint is filled by running each segment past
// it by `r · tan(θ/2)`, which is exactly where the two outer edges meet — but
// that runs away to infinity as a turn approaches a hairpin, so it is capped.
// 2.4 is the true mitre of a 135° turn, the sharpest the octile router emits.
#define PCB_MITRE_LIMIT      2.4
// Half the narrowest a shape may be drawn, in pixels. Anything thinner is
// widened to this and dimmed by exactly the factor it was widened by, so its
// total light stops depending on where it happens to fall on the pixel grid.
// See the energy note in shape_fragment.
// Anything thinner is widened to this and dimmed by the same factor. Kept
// because it is correct, but it is not the lever it looks like: measured, it
// moves the far end's energy error by three points out of ninety. Density, not
// thinness, is what breaks down there — hundreds of strokes to a pixel — and no
// per-shape rule can fix that.
#define PCB_MIN_FEATURE      0.6
// Ceiling on the antialiasing slack, in board units. The slack itself is sized
// in pixels (see `shape_vertex`) because a constant in board units shrinks to
// nothing down the strip — but the board is a plane running to a horizon, so
// board-units-per-pixel goes to infinity there and an unclamped slack goes with
// it. 8 covers the ~6.4 the far end actually needs; past that the fog has taken
// the copper anyway.
// Pixels of ramp outside the drawn stroke edge, for the analytic antialiasing.
// One pixel each side is all a coverage ramp needs; the ribbon carries it in
// screen space, so it is the same softness at every depth.
#define PCB_AA_RAMP_PX       1.0

// How much of each new frame the temporal pass takes. Low is steady; the
// neighbourhood clamp is what keeps low from smearing, because anything that
// genuinely changed cannot survive it. At 0.12 a pixel settles in about eight
// frames, an eighth of a second at 60 Hz.
#define PCB_HISTORY_BLEND    0.12
// How far outside its own neighbourhood a history sample may sit before it is
// rejected, as a multiple of that neighbourhood's half-range. 1.0 is the strict
// min/max clamp, which is safest against ghosting and also throws away most of
// the averaging: measured at 1.0 the far band went 5.06% -> 3.73%, against the
// ~4x an 0.12 blend should give if history survived.
#define PCB_HISTORY_CLAMP    2.0

#define PCB_HALO_EXTRA       3.4     // halo lineWidth = trace width + this
#define PCB_TEARDROP_EXTRA   3.2     // w2 = min(rad*0.85, w/2 + this)
#define PCB_TEARDROP_RATIO   0.85

// ─── Animation ────────────────────────────────────────────────────────────
#define PCB_SCROLL_SPEED     46.0    // board units/second the board drifts down

// ─── Camera ───────────────────────────────────────────────────────────────
// The board lies on the XZ plane and recedes into the distance. Board x maps
// to world X, board y to world Z, and the scroll carries the whole plane
// toward the camera — so a tile arriving "at the top" is a tile arriving far
// away, which is the same behaviour seen from a tilted eye.
#define PCB_CAMERA_HEIGHT    170.0   // eye height above the board plane
#define PCB_CAMERA_BACK      300.0   // how far behind the focus point the eye sits
#define PCB_CAMERA_LOOK_AHEAD 260.0  // how far ahead of the eye it looks
// A longish lens: narrow enough that the pitch clears half of it, so the top
// of the frame still looks at the board instead of past it into empty sky.
// Widen this and a black band opens above the strip.
#define PCB_CAMERA_FOV       32.0    // vertical field of view, degrees
#define PCB_CAMERA_NEAR      8.0
#define PCB_CAMERA_FAR       4000.0
#define PCB_FOCUS_DIST       430.0   // world distance held sharp
// Defocus strength. High enough that only a band around the focus plane is
// sharp: the copper rushing past the lens at the bottom of the frame and the
// copper melting into the haze at the top are both well out of focus, which is
// what makes the sharp middle read as depth rather than as a soft filter.
#define PCB_COC_SCALE        3.0
#define PCB_COC_MAX          3.4     // board units of blur, clamped

// How steeply the blur comes on either side of the focus plane, as a fraction
// of the focus distance. The foreground is the tighter of the two: the board is
// rushing past the lens there, so it should be well out of focus by the bottom
// of the frame rather than merely soft.
#define PCB_COC_NEAR_RAMP    0.34
#define PCB_COC_FAR_RAMP     0.90

// ─── Depth of field ───────────────────────────────────────────────────────
// A real screen-space blur, in the post chain.
//
// Two earlier attempts did it per shape — widening each instance's own coverage
// transition, then dilating each into a hard aperture disc — and neither can
// work, for the same reason: a shape only ever knows about itself, so no light
// ever moves between primitives. At the far end, where traces are thinner than
// a pixel and packed against each other, every trace merely faded on its own
// and its neighbours stayed individually resolved. That reads as aliasing, not
// as distance. A blur has to sample the neighbourhood.
//
// Convolved at **full resolution**, in two separable passes.
//
// The first version took its taps from a MIP level matched to the radius, which
// is the standard way to make a wide blur cheap. It shimmered: MIP texels snap
// to a fixed grid in screen space, so a board sliding underneath them samples a
// slightly different average every frame and the blur crawls against its own
// content. Nothing in the scene was moving relative to the blur — the sampling
// was.
//
// Separable is what makes full resolution affordable: an n×n kernel becomes
// n + n. And it is *exactly* right here rather than the usual approximation,
// because the circle of confusion depends only on the screen row — so the
// radius is constant along every horizontal line, which is the axis the first
// pass integrates.
#define PCB_DOF_MAX_RADIUS   26.0    // pixels of blur at full defocus
#define PCB_DOF_TAPS         24      // per side; spacing is radius / this
#define PCB_DOF_SIGMA        0.42    // of the radius, so the kernel dies inside it

// ─── Signal pulse ─────────────────────────────────────────────────────────
// A light runs the length of every trace, pauses at the end, and starts over,
// in that trace's own colour.
//
// Every trace runs on its OWN clock: its own moment to set off, its own time to
// cross, its own wait before going again. All three are drawn per trace from a
// hash of its geometry, so they are stable for the life of a tile and different
// from its neighbours'. Driving them all from one clock made the board blink in
// unison, which reads as a light show rather than as traffic on a circuit.
#define PCB_PULSE_TRAVEL_MIN 1.6     // seconds a head takes to cross its trace
#define PCB_PULSE_TRAVEL_MAX 4.6
#define PCB_PULSE_DELAY_MIN  0.5     // seconds dark before it sets off again
#define PCB_PULSE_DELAY_MAX  4.0
// ─── Path-following light runner ──────────────────────────────────────────
// A lit SEGMENT of the trace that travels along it, not a soft blob drifting
// over it. The track stays visible underneath the whole time — you can see
// where the runner has been and where it is going, which is what makes it read
// as something moving along a wire rather than as the wire changing brightness.
//
// Shaped from the head backwards: crisp at the leading edge, holding full
// brightness for most of its length, dissolving at the tail.
#define PCB_RUNNER_LENGTH    0.30    // lit fraction of the trace's arc length
#define PCB_RUNNER_TAIL      0.62    // of the runner: how much of it fades out
#define PCB_RUNNER_NOSE      0.018   // arc ahead of the head the edge softens over

// A crossing is one line drawn by two tiles that cannot compare notes, so its
// runner is a wave in SPACE rather than one head per trace: heads every
// WAVELENGTH board units, travelling at the speed the seam agreed. Each half
// then needs only its own length to place them — never its neighbour's.
//
// Sizing the head in board units rather than as a fraction of arc is not a
// detail: the two halves of a crossing have different lengths, so the old
// fraction changed the head's physical size as it passed the boundary, by up
// to 919 px on the measured set.
//
// SPAN / WAVELENGTH is how much of the time any one point is lit — 14%, which
// is what an ordinary trace already does. WAVELENGTH against a crossing's own
// length is how often it has a light on it anywhere: 80% at the median 958 px,
// against 19% for the single head this replaced.
#define PCB_RUNNER_WAVELENGTH 1000.0 // board units between one head and the next
#define PCB_RUNNER_SPAN        140.0 // lit length of one head, board units
#define PCB_RUNNER_NOSE_SPAN     8.0 // softened arc ahead of it, board units

#define PCB_PULSE_WIDTH      0.16    // head length, as a fraction of the trace
#define PCB_PULSE_GAIN       5.2     // how far above the trace colour the head burns
// How much of the stroke's half-width the moving light occupies. It is a
// thinner line running *inside* the trace, the way a lit fibre sits inside its
// own glow — not the whole width of the copper coming up together.
#define PCB_PULSE_CORE       0.55
// How many head-lengths past the end of the trace the head has to travel before
// the tail is gone and the trace is properly dark.
#define PCB_PULSE_TAIL       3.0

// How wide a trace has to be ON SCREEN, in pixels, for the light to be run
// along it at all.
//
// A travelling light needs a wire you can actually see it inside of. At the top
// of the frame perspective squeezes a trace to 0.28 px — well under a pixel —
// so a pixel there holds many traces at once and what reaches the eye is the
// sum of a dozen independent pulse clocks. That is not a light running along a
// wire, it is noise, and it is what "blinking like at New Year's" describes.
// Measured off a recording of the running board: pixels up there brightened for
// a median of 0.53 s at a median of 3.62 s apart, exactly PCB_PULSE_WIDTH x
// travel and travel + delay.
//
// Width, not length: length was the first guess and the measurement rejected it
// — at 12% down the frame a typical trace is still some sixty pixels long, so
// gating on length changed the far band by 0.02 points. Below about a pixel of
// width no per-trace animation can read as anything, however long the trace is.
#define PCB_PULSE_MIN_PIXELS   1.2     // trace narrower than this: no pulse
#define PCB_PULSE_FULL_PIXELS  3.0     // wider than this: full strength

// ─── Glow ─────────────────────────────────────────────────────────────────
// Pads are pushed above 1 so the bloom prefilter picks them out. A steady
// brightness, not an animated one: pads that pulsed between a filled disc and a
// hollow ring, and a slow rise and fall across the whole board underneath them,
// both read as the picture changing colour rather than as a circuit working.
// Neither survived contact with the thing they were meant to improve.
#define PCB_PAD_GLOW         2.6

// ─── Post-processing ──────────────────────────────────────────────────────
#define PCB_BLOOM_INTENSITY  0.85
#define PCB_EXPOSURE         1.15
#define PCB_SATURATION       1.25

// ─── Shape kinds understood by the shared vector pipeline ─────────────────
typedef enum {
    PCBShapeRoundRect = 0,  // rotated rounded rectangle, filled or stroked
    PCBShapeTrapezoid = 1,  // teardrop fillet where a trace meets its pad
} PCBShapeKind;

// ─── Buffer binding slots ─────────────────────────────────────────────────
typedef enum {
    PCBBufferInstances = 0,  // PCBShapeInstance / PCBGlyphInstance
    PCBBufferView      = 1,  // PCBViewUniforms
    PCBBufferPass      = 2,  // PCBPassUniforms
    PCBBufferPoints    = 3,  // PCBPathPoint (trace expansion input)
    PCBBufferTraces    = 4,  // PCBTraceInfo
    PCBBufferTraceVerts = 5, // PCBTraceVertex (the ribbon)
} PCBBufferIndex;

// ─── Shared records ───────────────────────────────────────────────────────
// Deliberately plain floats rather than simd vectors: identical layout and
// stride under Swift, C and MSL with no alignment surprises.

/// One instance of the shared vector pipeline. The meaning of the geometry
/// fields depends on `kind`:
///
///   PCBShapeRoundRect  centre (cx,cy), half extents (hw,hh) in the frame
///                      rotated by (cosR,sinR), corner radius `radius`.
///                      `strokeWidth` 0 fills, >0 strokes the outline.
///                      A trace segment is this shape with hh == radius — a
///                      stadium, i.e. a round-capped line.
///   PCBShapeTrapezoid  A = (cx,cy), direction (cosR,sinR), length `hw`,
///                      half-width `hh` at A and `radius` at B.
///
/// `expandable` is 1 for stroke geometry, which the halo pass inflates by
/// PCBPassUniforms.expand, and 0 for pads / bodies / decor, which never move.
typedef struct {
    float cx, cy;
    float hw, hh;
    float cosR, sinR;
    float radius;
    float strokeWidth;
    float kind;
    float expandable;
    float cr, cg, cb, ca;   // straight alpha; the shader premultiplies
} PCBShapeInstance;

/// One textured glyph quad (silkscreen numbers and reference designators).
typedef struct {
    float cx, cy;           // glyph box centre, tile pixels
    float hw, hh;           // half extents
    float cosR, sinR;       // rotation (vertical numbers use -90°)
    float u0, v0, u1, v1;   // atlas rect
    float cr, cg, cb, ca;
} PCBGlyphInstance;

/// One point of one trace centreline, trace-major so the GPU can expand the
/// whole tile's copper in a single dispatch while preserving draw order.
typedef struct {
    float x, y;
    unsigned int traceIndex;
    unsigned int flags;     // bit0 = first point of its trace, bit1 = last
    /// Arc length from the start of this trace, divided by its total length.
    /// Accumulated on the CPU while the polyline is already in hand — the
    /// kernel is one thread per point and cannot run a prefix sum along a
    /// trace without a second pass.
    float s;
    /// Index of this point's LEFT ribbon vertex; the right one follows it. Laid
    /// out on the CPU, where the trace boundaries are already known, so the
    /// expansion kernel needs no prefix sum to find where to write.
    float vertexBase;
    float _pad1, _pad2;
} PCBPathPoint;

#define PCB_POINT_FIRST 1u
#define PCB_POINT_LAST  2u

/// Per-trace constants consumed by the expansion kernel.
typedef struct {
    float halfWidth;
    float cr, cg, cb, ca;
    /// This trace's own pulse clock. `phase` is where in its cycle it stands at
    /// t = 0, `period` is one setting-off to the next, and `speed` is how much
    /// of the trace the head covers per second — so 1/speed is how long the
    /// crossing takes and the rest of the period is the wait.
    float pulsePhase, pulsePeriod, pulseSpeed;
    /// Normalised arc per board unit — 1 / this trace's length. Non-zero puts
    /// the trace on the shared spatial wave (see PCB_RUNNER_WAVELENGTH) and is
    /// what converts the board-unit runner constants into this trace's own
    /// parameter. Zero keeps the one-head-per-trace clock above.
    float pulseUnit;
} PCBTraceInfo;

/// One corner of a trace's ribbon. Two per centreline point, one either side.
///
/// A trace is drawn as a SINGLE triangle strip — every trace in a tile stitched
/// into one, so the whole board's copper is one `drawPrimitives` call. That is
/// not a micro-optimisation: consecutive segments used to be separate
/// overlapping quads, and where a stroke is not opaque (which is everywhere in
/// the distance, once coverage and fog have had their say) the overlap
/// composited the same coverage twice. A strip's quads share an edge exactly, so
/// no pixel is ever painted twice by the same trace — by construction rather
/// than by a test.
typedef struct {
    /// Centreline point, board units. A trace's two ends are already pushed out
    /// by a half-width here, which is the square cap.
    float px, py;
    /// Unit mitre bisector, and how far along it the true stroke edge sits:
    /// `r / cos(theta/2)`, clamped by PCB_MITRE_LIMIT. This is what makes the
    /// joins mitres rather than notches, and it is the same rule the old
    /// per-segment expansion used.
    float nx, ny;
    float mitre;
    /// -1 or +1: which side of the centreline this corner is on.
    float side;
    /// -1 at a trace's first point, +1 at its last, 0 in between: which way to
    /// push the square cap. Applied in the vertex shader rather than baked in
    /// here, because the halo pass draws the same ribbon wider and its cap has
    /// to grow with it — baked at the core width it left a gap at every end.
    float cap;
    /// Normalised arc length along the trace, for the pulse.
    float s;
    /// Half width in board units, before any halo widening.
    float r;
    /// Which `PCBTraceInfo` this corner belongs to.
    ///
    /// The colour and the light runner's clock used to be copied onto every
    /// corner — seven floats duplicated across two vertices per centreline
    /// point, when both already existed once per trace in `PCBTraceInfo`. The
    /// vertex shader reads them by index instead, which takes the record from
    /// 64 bytes to 40 and a slot's ribbon from 288 KB to 180 KB. A strip is
    /// vertex-bound, so that is bandwidth on the path that draws every frame.
    float traceIndex;
} PCBTraceVertex;

/// Where a tile's board origin lands on the drawable, and how far its colours
/// recede. One of these per tile per parallax layer — the board is drawn
/// straight to the screen, so there is no intermediate tile texture and no
/// resolution baked into anything.
typedef struct {
    PCB_FLOAT4X4 viewProj;
    /// Board (x, y) becomes world (x + originX, height, y + originZ). One of
    /// these per tile per layer: `originZ` carries the scroll and the tile's
    /// place in the strip, `height` lifts or drops a stacked inner layer.
    float originX, originZ, height;
    /// World units per board unit — solved from the aspect so the board frames
    /// the same way on a phone and on a desktop.
    float worldScale;
    /// 1 = the live board. Below that the colour mixes toward the substrate,
    /// so a deeper layer recedes rather than merely dimming to black.
    float fade;
    float subR, subG, subB;
    /// Eye position, for the depth-of-field falloff.
    float camX, camY, camZ;
    /// Multiplies every fragment's alpha. Held at 0 until the whole strip is
    /// resident, then ramped to 1 — so the board arrives complete rather than
    /// being watched to assemble itself tile by tile.
    float alpha;
    float focusDist, cocScale, cocMax;
    /// Copper dissolves into the background haze between these two distances,
    /// so the far end of the strip never simply stops.
    float fogStart, fogEnd;
    /// Seconds since the board was revealed, and how far above the trace's own
    /// colour a head burns. The *time* rather than a head position: every trace
    /// runs its own clock, so the position can only be worked out per trace.
    /// Gain 0 switches the pulse off for the whole layer.
    float pulseTime, pulseGain;
    /// Colour of the light runner, and how much of it to use: 0 takes the
    /// trace's own colour — which is what makes the light look like it belongs
    /// to that wire — and 1 takes this one outright.
    float runR, runG, runB, runMix;
    /// Half the drawable, in pixels. The only way a vertex can know how big a
    /// pixel is in board units — which is what the antialiasing slack has to be
    /// measured in, because a constant in board units shrinks to nothing down
    /// the strip. See the note on `aa` in `shape_vertex`.
    float viewHalfW, viewHalfH;
} PCBViewUniforms;

/// Exposure, saturation and bloom mix for the composite pass.
typedef struct {
    float bloomIntensity;
    float exposure;
    float saturation;
    /// 0 switches the depth of field off entirely.
    float dofStrength;
    /// The board is a plane under a known camera, so a pixel's distance from the
    /// eye follows from its screen row alone — and *reciprocal* distance is
    /// linear in that row, which is what makes this two numbers instead of an
    /// inverse projection. `1/slant range` at the bottom and top of the frame;
    /// the blur pass interpolates between them and inverts.
    float invNear, invFar;
    /// The same circle-of-confusion shaping the board pass used to do per shape.
    float focusDist, cocScale, cocMax;

    // ─── Temporal accumulation ────────────────────────────────────────────
    // The far end of the strip is finer than the pixel grid can hold, and no
    // amount of per-shape care fixes that: a trace up there is 0.28 px wide, so
    // its coverage depends on where it falls between sample points and it
    // shimmers as the board drifts. Measured, lines swing 3.4x as much as pads,
    // which are the control because a pad is a single instance.
    //
    // What resolves it is more samples — and the cheapest place to find them is
    // in TIME. Successive frames sample the same geometry at different sub-pixel
    // offsets, so averaging them is supersampling spread across frames instead
    // of across the fill rate. Supersampling 3x measured 3.10% -> 2.44% on the
    // twinkle metric at nine times the fragment cost; this gets the same effect
    // for one extra full-screen pass.
    //
    // It is unusually cheap here because the motion is ANALYTIC. The board is a
    // plane, the camera never moves, and the content translates along z at a
    // known rate — so where a pixel's board point sat last frame is closed form.
    // No velocity buffer, no G-buffer, no motion-vector pass: just the inverse
    // of the view-projection and how far the board slid.
    PCB_FLOAT4X4 invViewProj;
    /// World units the board slid toward the camera since the last frame. The
    /// whole of the motion, because nothing else in the scene moves.
    float boardDelta;
    /// How much of the new frame to take, per frame. Low is steady and slow to
    /// react; the neighbourhood clamp is what makes low safe.
    float historyBlend;
    /// 0 on the first frame after a resize or a reveal, when there is no
    /// history worth trusting and the current frame has to stand alone.
    float historyValid;
    float _pad0;
} PCBPostUniforms;

/// Per-draw modifiers, so halo and core share one instance buffer.
typedef struct {
    float expand;        // added to expandable stroke geometry (halo pass)
    float useOverride;   // 1 = ignore per-instance colour
    float or_, og, ob, oa;
    float _pad0, _pad1;
} PCBPassUniforms;

#endif /* PCBBridge_h */
