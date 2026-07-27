#include <metal_stdlib>
#include "PCBBridge.h"
using namespace metal;

// ── Board vector art ──────────────────────────────────────────────────────
// Every filled or stroked thing on the board — trace bodies, teardrop fillets,
// pads, package bodies, silkscreen decor — is one
// instance of PCBShapeInstance through one pipeline, drawn in submission order
// so blending stacks exactly the way the Canvas2D draw list did.
//
// Everything outputs PREMULTIPLIED alpha, so a pad's three stacked layers and
// a trace's halo composite with a single (one, 1-srcAlpha) blend.

/// Board units to world space. The board lies on the XZ plane: board x runs
/// across it, board y runs away from the camera, and `height` stacks the inner
/// layers below the live one. No intermediate tile texture is involved — the
/// vector art is transformed at the drawable's own resolution every frame.
static float3 board_to_world(float2 boardPosition, constant PCBViewUniforms &viewUniforms)
{
    return float3((boardPosition.x + viewUniforms.originX) * viewUniforms.worldScale,
                  viewUniforms.height * viewUniforms.worldScale,
                  (boardPosition.y + viewUniforms.originZ) * viewUniforms.worldScale);
}

static float4 board_to_clip(float2 boardPosition, constant PCBViewUniforms &viewUniforms)
{
    return viewUniforms.viewProj * float4(board_to_world(boardPosition, viewUniforms), 1.0);
}

/// Atmospheric depth: a receding layer mixes toward the substrate rather than
/// merely darkening, so it reads as distance instead of dirt.
static float3 recede(float3 baseColor, constant PCBViewUniforms &viewUniforms)
{
    return mix(float3(viewUniforms.subR, viewUniforms.subG, viewUniforms.subB), baseColor, viewUniforms.fade);
}

/// Distance haze. Only so much of the strip is drawn, so it has to have faded
/// out before that edge — otherwise the board stops in mid-air at a hard line.
static float fog(float3 worldPosition, constant PCBViewUniforms &viewUniforms)
{
    if (viewUniforms.fogEnd <= viewUniforms.fogStart) return 1.0;
    const float distanceToCamera = length(float3(viewUniforms.camX, viewUniforms.camY, viewUniforms.camZ) - worldPosition);
    return 1.0 - smoothstep(viewUniforms.fogStart, viewUniforms.fogEnd, distanceToCamera);
}

// ── Trace expansion ───────────────────────────────────────────────────────
// One thread per centreline point; one square-capped rectangle per segment.

/// `normalize`, but a zero-length vector falls back rather than producing NaN.
static float2 normalize_or(float2 inputVector, float2 fallbackVector)
{
    const float vectorLength = length(inputVector);
    return (vectorLength > 1e-6) ? inputVector / vectorLength : fallbackVector;
}

kernel void kernel_expand_traces(device const PCBPathPoint *pathPointsBuffer     [[buffer(PCBBufferPoints)]],
                                 device const PCBTraceInfo *traceInfoBuffer       [[buffer(PCBBufferTraces)]],
                                 device PCBTraceVertex     *outputVerticesBuffer [[buffer(PCBBufferTraceVerts)]],
                                 constant uint             &totalPointCount       [[buffer(PCBBufferPass)]],
                                 uint globalThreadID [[thread_position_in_grid]])
{
    if (globalThreadID >= totalPointCount) return;

    const PCBPathPoint currentPoint = pathPointsBuffer[globalThreadID];
    const PCBTraceInfo currentTraceInfo = traceInfoBuffer[currentPoint.traceIndex];
    const float traceHalfWidth = currentTraceInfo.halfWidth;
    const bool isFirstPointInTrace = (currentPoint.flags & PCB_POINT_FIRST) != 0u;
    const bool isLastPointInTrace  = (currentPoint.flags & PCB_POINT_LAST)  != 0u;

    const float2 currentPointPosition = float2(currentPoint.x, currentPoint.y);

    // The directions in and out of this point.
    float2 incomingDirection, outgoingDirection;
    if (isFirstPointInTrace) {
        const float2 nextPointPosition = float2(pathPointsBuffer[globalThreadID + 1].x, pathPointsBuffer[globalThreadID + 1].y);
        outgoingDirection = normalize_or(nextPointPosition - currentPointPosition, float2(1.0, 0.0));
        incomingDirection = outgoingDirection;
    } else if (isLastPointInTrace) {
        const float2 previousPointPosition = float2(pathPointsBuffer[globalThreadID - 1].x, pathPointsBuffer[globalThreadID - 1].y);
        incomingDirection = normalize_or(currentPointPosition - previousPointPosition, float2(1.0, 0.0));
        outgoingDirection = incomingDirection;
    } else {
        const float2 previousPointPosition = float2(pathPointsBuffer[globalThreadID - 1].x, pathPointsBuffer[globalThreadID - 1].y);
        const float2 nextPointPosition     = float2(pathPointsBuffer[globalThreadID + 1].x, pathPointsBuffer[globalThreadID + 1].y);
        incomingDirection = normalize_or(currentPointPosition - previousPointPosition, float2(1.0, 0.0));
        outgoingDirection = normalize_or(nextPointPosition - currentPointPosition, incomingDirection);
    }

    // The mitre.
    const float2 incomingNormal = float2(-incomingDirection.y, incomingDirection.x);
    const float2 outgoingNormal = float2(-outgoingDirection.y, outgoingDirection.x);
    float2 bisectorNormal = incomingNormal + outgoingNormal;
    const float bisectorLength = length(bisectorNormal);
    float computedMiterLength;
    
    if (bisectorLength < 1e-5) {
        bisectorNormal = incomingNormal;
        computedMiterLength = traceHalfWidth;
    } else {
        bisectorNormal /= bisectorLength;
        computedMiterLength = traceHalfWidth / max(dot(bisectorNormal, outgoingNormal), 1.0 / PCB_MITRE_LIMIT);
    }

    PCBTraceVertex generatedVertex;
    generatedVertex.px = currentPointPosition.x;
    generatedVertex.py = currentPointPosition.y;
    generatedVertex.nx = bisectorNormal.x;
    generatedVertex.ny = bisectorNormal.y;
    generatedVertex.mitre = computedMiterLength;
    generatedVertex.cap = isFirstPointInTrace ? -1.0 : (isLastPointInTrace ? 1.0 : 0.0);
    generatedVertex.s = currentPoint.s;
    generatedVertex.r = traceHalfWidth;
    generatedVertex.traceIndex = float(currentPoint.traceIndex);

    const uint baseVertexIndex = uint(currentPoint.vertexBase);
    generatedVertex.side = -1.0; outputVerticesBuffer[baseVertexIndex]     = generatedVertex;
    generatedVertex.side =  1.0; outputVerticesBuffer[baseVertexIndex + 1] = generatedVertex;
    if (isFirstPointInTrace) { generatedVertex.side = -1.0; outputVerticesBuffer[baseVertexIndex - 1] = generatedVertex; }
    if (isLastPointInTrace)  { generatedVertex.side =  1.0; outputVerticesBuffer[baseVertexIndex + 2] = generatedVertex; }
}

// ── Trace ribbon ──────────────────────────────────────────────────────────

struct TraceVertexOut {
    float4 position [[position]];
    /// Across the ribbon: 0 on the centreline, +/-1 exactly on the drawn edge.
    float  normalizedEdgeOffset;
    /// Normalised arc length, for the pulse head.
    float  arcLengthDistance;
    /// How much this fragment's light was dimmed to pay for widening a stroke.
    float  energyPreservationFactor;
    float  fogFactor;
    float4 baseColor;
    float3 pulseParameters; // phase, period, speed
};

vertex TraceVertexOut trace_vertex(uint vertexID [[vertex_id]],
                              device const PCBTraceVertex *traceVerticesBuffer [[buffer(PCBBufferTraceVerts)]],
                              device const PCBTraceInfo   *traceInfoBuffer     [[buffer(PCBBufferTraces)]],
                              constant PCBViewUniforms    &viewUniforms        [[buffer(PCBBufferView)]],
                              constant PCBPassUniforms    &passUniforms        [[buffer(PCBBufferPass)]])
{
    const PCBTraceVertex vertexData = traceVerticesBuffer[vertexID];
    const PCBTraceInfo traceInfo = traceInfoBuffer[uint(vertexData.traceIndex)];
    const float2 vertexNormal = float2(vertexData.nx, vertexData.ny);
    const float2 vertexTangent = float2(vertexNormal.y, -vertexNormal.x);

    const float targetHalfWidth = max(vertexData.r + passUniforms.expand, 1e-5);
    const float scaledMiterLength = vertexData.mitre * (targetHalfWidth / max(vertexData.r, 1e-5));
    const float2 centerPositionBoard = float2(vertexData.px, vertexData.py) + vertexTangent * (vertexData.cap * targetHalfWidth);

    const float3 centerPositionWorld = board_to_world(centerPositionBoard, viewUniforms);

    const float3 unitStepWorld = board_to_world(centerPositionBoard + vertexNormal, viewUniforms) - centerPositionWorld;
    const float4 centerClipPosition = viewUniforms.viewProj * float4(centerPositionWorld, 1.0);
    const float4 stepClipPosition   = viewUniforms.viewProj * float4(centerPositionWorld + unitStepWorld, 1.0);
    const float2 normalizedDeviceOffset = stepClipPosition.xy / max(stepClipPosition.w, 1e-6) - centerClipPosition.xy / max(centerClipPosition.w, 1e-6);
    const float pixelsPerBoardUnit = max(length(normalizedDeviceOffset * float2(viewUniforms.viewHalfW, viewUniforms.viewHalfH)), 1e-6);

    const float trueWidthPixels  = targetHalfWidth * pixelsPerBoardUnit;
    const float drawnWidthPixels = max(trueWidthPixels, PCB_MIN_FEATURE);
    const float energyScalingFactor  = trueWidthPixels / drawnWidthPixels;

    const float outerExtentPixels = drawnWidthPixels + PCB_AA_RAMP_PX;
    const float offsetBoardUnits  = scaledMiterLength * (outerExtentPixels / max(trueWidthPixels, 1e-6));

    TraceVertexOut out;

    const float2 finalCornerPositionBoard = centerPositionBoard + vertexNormal * (vertexData.side * offsetBoardUnits);

    out.position                 = board_to_clip(finalCornerPositionBoard, viewUniforms);
    out.normalizedEdgeOffset     = vertexData.side * (outerExtentPixels / drawnWidthPixels);
    out.arcLengthDistance        = vertexData.s;
    out.energyPreservationFactor = energyScalingFactor;
    out.fogFactor                = fog(board_to_world(finalCornerPositionBoard, viewUniforms), viewUniforms);
    out.baseColor                = (passUniforms.useOverride > 0.5)
        ? float4(passUniforms.or_, passUniforms.og, passUniforms.ob, passUniforms.oa)
        : float4(traceInfo.cr, traceInfo.cg, traceInfo.cb, traceInfo.ca);
    out.pulseParameters          = float3(traceInfo.pulsePhase, traceInfo.pulsePeriod, traceInfo.pulseSpeed);

    return out;
}

fragment float4 trace_fragment(TraceVertexOut in [[stage_in]],
                               constant PCBViewUniforms &viewUniforms [[buffer(PCBBufferView)]],
                               constant PCBPassUniforms &passUniforms [[buffer(PCBBufferPass)]])
{
    const float absoluteNormalizedEdge = fabs(in.normalizedEdgeOffset);

    const float edgeGradientPerPixel = max(length(float2(dfdx(in.normalizedEdgeOffset), dfdy(in.normalizedEdgeOffset))), 1e-6);
    const float pixelCoverage = saturate(0.5 - (absoluteNormalizedEdge - 1.0) / edgeGradientPerPixel);

    float3 colorRGB = recede(in.baseColor.rgb, viewUniforms);
    const float finalAlpha = in.baseColor.a * pixelCoverage * in.energyPreservationFactor * in.fogFactor * viewUniforms.alpha;

    // ── The path-following light runner ─────────────────────────────────────
    if (viewUniforms.pulseGain > 0.0 && passUniforms.useOverride < 0.5) {
        const float pulsePeriod = max(in.pulseParameters.y, 1e-3);
        const float pulseCycleTime = fract((viewUniforms.pulseTime + in.pulseParameters.x) / pulsePeriod) * pulsePeriod;
        const float runnerHeadPosition = pulseCycleTime * in.pulseParameters.z;

        const float distanceBehindHead = runnerHeadPosition - in.arcLengthDistance;
        const float noseSoftnessFactor = 1.0 - saturate(-distanceBehindHead / PCB_RUNNER_NOSE);
        const float runnerNormalizedLength = 1.0 - saturate(distanceBehindHead / PCB_RUNNER_LENGTH);
        const float runnerBodyBrightness = smoothstep(0.0, PCB_RUNNER_TAIL, runnerNormalizedLength);

        const float runnerCoreFactor = 1.0 - smoothstep(PCB_PULSE_CORE * 0.55, PCB_PULSE_CORE, absoluteNormalizedEdge);

        const float motionVisibilityFactor = smoothstep(PCB_PULSE_MIN_PIXELS, PCB_PULSE_FULL_PIXELS, 2.0 / edgeGradientPerPixel);

        const float3 runnerTintRGB = mix(in.baseColor.rgb, float3(viewUniforms.runR, viewUniforms.runG, viewUniforms.runB), viewUniforms.runMix);
        colorRGB += runnerTintRGB * (viewUniforms.fade * viewUniforms.pulseGain * runnerCoreFactor * motionVisibilityFactor * runnerBodyBrightness * noseSoftnessFactor);
    }

    return float4(colorRGB * finalAlpha, finalAlpha);
}

// ── Shape pipeline ────────────────────────────────────────────────────────

struct ShapeVary {
    float4 position [[position]];
    float2 localCoordinates;
    float  fogFactor;      // 1 = clear, 0 = swallowed by the distance
    uint   instanceID [[flat]];
};

constant float2 quadCornerVertices[6] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1),
    float2( 1, -1), float2(1,  1), float2(-1, 1),
};

vertex ShapeVary shape_vertex(uint vertexID [[vertex_id]],
                              uint instanceID [[instance_id]],
                              device const PCBShapeInstance *shapeInstancesBuffer [[buffer(PCBBufferInstances)]],
                              constant PCBViewUniforms      &viewUniforms         [[buffer(PCBBufferView)]],
                              constant PCBPassUniforms      &passUniforms         [[buffer(PCBBufferPass)]])
{
    const PCBShapeInstance shapeInstance = shapeInstancesBuffer[instanceID];
    const float2 cornerVertex = quadCornerVertices[vertexID];
    const float expansionAmount = (shapeInstance.expandable > 0.5) ? passUniforms.expand : 0.0;

    const float3 centerWorldPosition = board_to_world(float2(shapeInstance.cx, shapeInstance.cy), viewUniforms);

    const float2 rotationAxisX = float2(shapeInstance.cosR, shapeInstance.sinR);
    const float2 rotationAxisY = float2(-rotationAxisX.y, rotationAxisX.x);

    const float antiAliasingMargin = 1.0;
    const uint shapeType = uint(shapeInstance.kind);

    float2 computedLocalCoordinates;
    float2 computedWorldCoordinates;

    if (shapeType == uint(PCBShapeTrapezoid)) {
        const float maxHalfHeight = max(shapeInstance.hh, shapeInstance.radius) + antiAliasingMargin;
        computedLocalCoordinates = float2((cornerVertex.x * 0.5 + 0.5) * shapeInstance.hw, cornerVertex.y * maxHalfHeight);
        computedWorldCoordinates = float2(shapeInstance.cx, shapeInstance.cy) + rotationAxisX * computedLocalCoordinates.x + rotationAxisY * computedLocalCoordinates.y;
    } else {
        const float2 boundingExtent = float2(shapeInstance.hw, shapeInstance.hh) + shapeInstance.strokeWidth * 0.5 + expansionAmount + antiAliasingMargin;
        computedLocalCoordinates = cornerVertex * boundingExtent;
        computedWorldCoordinates = float2(shapeInstance.cx, shapeInstance.cy) + rotationAxisX * computedLocalCoordinates.x + rotationAxisY * computedLocalCoordinates.y;
    }

    ShapeVary outputVaryings;
    outputVaryings.position = board_to_clip(computedWorldCoordinates, viewUniforms);
    outputVaryings.localCoordinates = computedLocalCoordinates;
    outputVaryings.fogFactor = fog(centerWorldPosition, viewUniforms);
    outputVaryings.instanceID = instanceID;
    return outputVaryings;
}

fragment float4 shape_fragment(ShapeVary inputVaryings [[stage_in]],
                               device const PCBShapeInstance *shapeInstancesBuffer [[buffer(PCBBufferInstances)]],
                               constant PCBViewUniforms      &viewUniforms         [[buffer(PCBBufferView)]],
                               constant PCBPassUniforms      &passUniforms         [[buffer(PCBBufferPass)]])
{
    const PCBShapeInstance shapeInstance = shapeInstancesBuffer[inputVaryings.instanceID];
    const float expansionAmount = (shapeInstance.expandable > 0.5) ? passUniforms.expand : 0.0;
    const uint shapeType = uint(shapeInstance.kind);

    float signedDistance;
    float halfThickness;

    if (shapeType == uint(PCBShapeTrapezoid)) {
        const float trapezoidInterpolant = clamp(inputVaryings.localCoordinates.x / max(shapeInstance.hw, 1e-5), 0.0, 1.0);
        signedDistance = fabs(inputVaryings.localCoordinates.y) - mix(shapeInstance.hh, shapeInstance.radius, trapezoidInterpolant);
        halfThickness = min(shapeInstance.hh, shapeInstance.radius);
    } else {
        const float2 boxDistanceVector = fabs(inputVaryings.localCoordinates) - (float2(shapeInstance.hw, shapeInstance.hh) - shapeInstance.radius);
        signedDistance = length(max(boxDistanceVector, 0.0)) + min(max(boxDistanceVector.x, boxDistanceVector.y), 0.0) - shapeInstance.radius;

        if (shapeInstance.strokeWidth > 0.0) {
            signedDistance = fabs(signedDistance) - shapeInstance.strokeWidth * 0.5;
            halfThickness = shapeInstance.strokeWidth * 0.5;
        } else {
            signedDistance -= expansionAmount;
            halfThickness = min(shapeInstance.hw, shapeInstance.hh) + expansionAmount;
        }
    }

    const float signedDistanceGradient = max(length(float2(dfdx(signedDistance), dfdy(signedDistance))), 1e-6);

    const float wideningAmount = max(0.0, PCB_MIN_FEATURE * signedDistanceGradient - halfThickness);
    signedDistance -= wideningAmount;
    const float energyConservationFactor = halfThickness / max(halfThickness + wideningAmount, 1e-6);

    const float pixelCoverage = saturate(0.5 - signedDistance / signedDistanceGradient);

    const float4 instanceColor = (passUniforms.useOverride > 0.5)
        ? float4(passUniforms.or_, passUniforms.og, passUniforms.ob, passUniforms.oa)
        : float4(shapeInstance.cr, shapeInstance.cg, shapeInstance.cb, shapeInstance.ca);
        
    const float finalAlpha = instanceColor.a * pixelCoverage * energyConservationFactor * inputVaryings.fogFactor * viewUniforms.alpha;
    const float3 colorRGB = recede(instanceColor.rgb, viewUniforms);

    return float4(colorRGB * finalAlpha, finalAlpha);
}

// ── Silkscreen numbers ────────────────────────────────────────────────────

struct GlyphVertexOut {
    float4 position [[position]];
    float2 textureCoordinates;
    float  fadeFactor;     // defocus and distance combined
    uint   instanceID [[flat]];
};

vertex GlyphVertexOut glyph_vertex(uint vertexID [[vertex_id]],
                              uint instanceID [[instance_id]],
                              device const PCBGlyphInstance *glyphInstancesBuffer [[buffer(PCBBufferInstances)]],
                              constant PCBViewUniforms      &viewUniforms         [[buffer(PCBBufferView)]])
{
    const PCBGlyphInstance glyphInstance = glyphInstancesBuffer[instanceID];
    const float2 cornerVertex = quadCornerVertices[vertexID];
    const float2 rotationAxisX = float2(glyphInstance.cosR, glyphInstance.sinR);
    const float2 rotationAxisY = float2(-rotationAxisX.y, rotationAxisX.x);
    const float2 localOffset = cornerVertex * float2(glyphInstance.hw, glyphInstance.hh);
    const float2 worldBoardPosition = float2(glyphInstance.cx, glyphInstance.cy) + rotationAxisX * localOffset.x + rotationAxisY * localOffset.y;

    const float3 centerWorldPosition = board_to_world(float2(glyphInstance.cx, glyphInstance.cy), viewUniforms);

    GlyphVertexOut out;
    out.position = board_to_clip(worldBoardPosition, viewUniforms);
    out.textureCoordinates = float2(mix(glyphInstance.u0, glyphInstance.u1, cornerVertex.x * 0.5 + 0.5),
                                               mix(glyphInstance.v0, glyphInstance.v1, cornerVertex.y * 0.5 + 0.5));
    out.fadeFactor = fog(centerWorldPosition, viewUniforms);
    out.instanceID = instanceID;
    return out;
}

fragment float4 glyph_fragment(GlyphVertexOut in [[stage_in]],
                               device const PCBGlyphInstance *glyphInstancesBuffer [[buffer(PCBBufferInstances)]],
                               constant PCBViewUniforms      &viewUniforms         [[buffer(PCBBufferView)]],
                               texture2d<float>               fontAtlasTexture     [[texture(0)]])
{
    constexpr sampler atlasSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    const PCBGlyphInstance glyphInstance = glyphInstancesBuffer[in.instanceID];
    const float coverageFromAtlas = fontAtlasTexture.sample(atlasSampler, in.textureCoordinates).r;
    const float finalAlpha = glyphInstance.ca * coverageFromAtlas * in.fadeFactor * viewUniforms.alpha;
    
    return float4(recede(float3(glyphInstance.cr, glyphInstance.cg, glyphInstance.cb), viewUniforms) * finalAlpha, finalAlpha);
}
