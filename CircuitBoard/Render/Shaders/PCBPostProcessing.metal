#include <metal_stdlib>
#include "PCBBridge.h"
using namespace metal;

// ── Background, bloom and tonemap ─────────────────────────────────────────
// pipelines: the board draws its own copper. The scene renders into an HDR
// target, its bright parts are extracted and blurred at half resolution, and
// the two are composited with exposure, saturation and an ACES filmic curve.

namespace Background {
    constant float3 darkColor     = float3(0.010, 0.022, 0.055);
    constant float3 bandColor     = float3(0.035, 0.085, 0.180);
    // Measured *down* from the top of the frame, like every other v in this
    // file. 0.45 is a tenth of the frame above centre, up where the strip runs
    // into the distance — that is the point of the band, so it does not sit at
    // 0.5. It reads 0.45 rather than the 0.55 it had while the background pass
    // owned its own bottom-up vertex shader; same place on screen.
    constant float  bandCenterY   = 0.45;
    constant float  bandSharpness = 2.6;
    constant float  centerSharpness = 1.8;  // horizontal falloff of the band
    constant float  centerFloor   = 0.30;   // band strength at the frame edges
    constant float  centerGain    = 0.70;   // extra band strength at the center
    constant float  bandStrength  = 0.5;
}

namespace Bloom {
    constant float tapOffset      = 0.5;    // half-texel shift for the 4-tap box filter
    constant float boxAverage     = 0.25;
    constant float kneeStart      = 0.18;   // luma soft-knee: below this nothing blooms
    constant float kneeEnd        = 0.75;   // …above this everything blooms
    constant float blurStepTexels = 1.5;    // 1.5-texel steps double the kernel via bilinear
    constant float3 lumaWeights   = float3(0.299, 0.587, 0.114);
    constant float gauss[5]       = { 0.227027, 0.1945946, 0.1216216, 0.0540541, 0.0162162 };
}

namespace Tonemap {  // Narkowicz ACES filmic curve fit
    constant float coefficientA = 2.51;
    constant float coefficientB = 0.03;
    constant float coefficientC = 2.43;
    constant float coefficientD = 0.59;
    constant float coefficientE = 0.14;
}

struct FullscreenVary {
    float4 position [[position]];
    float2 textureCoordinates;
};

/// One oversized triangle covers the screen with no vertex buffer.
///
/// Every fullscreen pass uses this one — background, depth of field, bloom and
/// composite. There used to be a second, identical but for its UV convention:
/// this one puts v = 0 at the *top*, matching how the post chain samples its
/// textures, and the other left v = 0 at the bottom. Nothing sampled a texture
/// through the second one, so it was not a misalignment; it was two conventions
/// in one file waiting to be confused for each other.
vertex FullscreenVary fullscreenVertex(uint vertexID [[vertex_id]])
{
    float2 screenPosition = float2((vertexID == 1) ? 3.0 : -1.0, (vertexID == 2) ? 3.0 : -1.0);
    FullscreenVary outputData;
    outputData.position = float4(screenPosition, 0.0, 1.0);
    outputData.textureCoordinates = float2(screenPosition.x * 0.5 + 0.5, 0.5 - screenPosition.y * 0.5);
    return outputData;
}

// ── Background ────────────────────────────────────────────────────────────

fragment float4 backgroundFragment(FullscreenVary inputData [[stage_in]])
{
    // Hazy glow band across the middle, strongest near the centre — it sits
    // where the board meets the horizon and gives the distance somewhere to
    // dissolve into.
    float horizontalBandGlow = exp(-pow((inputData.textureCoordinates.y - Background::bandCenterY) * Background::bandSharpness, 2.0));
    float centerWeighting = exp(-pow((inputData.textureCoordinates.x - 0.5) * Background::centerSharpness, 2.0));
    
    float3 finalBackgroundColor = Background::darkColor
               + Background::bandColor * horizontalBandGlow
                 * (Background::centerFloor + Background::centerGain * centerWeighting)
                 * Background::bandStrength;
                 
    return float4(finalBackgroundColor, 1.0);
}

// ── Temporal accumulation ─────────────────────────────────────────────────
// Supersampling paid for in time rather than in fill rate.

/// Where the board point under `textureCoordinates` sat last frame, in UV.
struct Reprojected {
    float2 textureCoordinates;
    bool   isValidReprojection;
};

static Reprojected reproject(float2 textureCoordinates, constant PCBPostUniforms &postUniforms,
                             float4x4 viewProjectionMatrix, float3 cameraWorldPosition)
{
    // UV is y-down; NDC is y-up.
    const float2 normalizedDeviceCoordinates = float2(textureCoordinates.x * 2.0 - 1.0, 1.0 - textureCoordinates.y * 2.0);
    const float4 farClipSpacePosition = postUniforms.invViewProj * float4(normalizedDeviceCoordinates, 1.0, 1.0);
    const float3 targetWorldPosition = farClipSpacePosition.xyz / max(farClipSpacePosition.w, 1e-6);
    const float3 viewDirection = normalize(targetWorldPosition - cameraWorldPosition);

    // The board plane is y = 0. A ray that never descends never meets it — the
    // sky above the horizon, which is background and has no history worth
    // fetching.
    if (viewDirection.y > -1e-5) { return Reprojected{ textureCoordinates, false }; }
    const float3 boardIntersectionPoint = cameraWorldPosition + viewDirection * (-cameraWorldPosition.y / viewDirection.y);

    // Where that piece of board was a frame ago: further away by exactly the
    // distance the strip has advanced.
    const float4 previousClipSpacePosition = viewProjectionMatrix * float4(boardIntersectionPoint.x, boardIntersectionPoint.y, boardIntersectionPoint.z + postUniforms.boardDelta, 1.0);
    
    if (previousClipSpacePosition.w <= 1e-6) { return Reprojected{ textureCoordinates, false }; }
    
    const float2 previousNormalizedDeviceCoords = previousClipSpacePosition.xy / previousClipSpacePosition.w;
    const float2 previousTextureCoords = float2(previousNormalizedDeviceCoords.x * 0.5 + 0.5, 0.5 - previousNormalizedDeviceCoords.y * 0.5);
    
    return Reprojected{ previousTextureCoords, all(previousTextureCoords >= 0.0) && all(previousTextureCoords <= 1.0) };
}

fragment float4 temporalAccumulate(FullscreenVary inputData [[stage_in]],
                                   texture2d<float> sceneTexture   [[texture(0)]],
                                   texture2d<float> historyTexture [[texture(1)]],
                                   constant PCBPostUniforms &postUniforms  [[buffer(0)]],
                                   constant PCBViewUniforms &viewUniforms  [[buffer(1)]])
{
    constexpr sampler linearClampSampler(filter::linear, address::clamp_to_edge);
    constexpr sampler pointClampSampler(filter::nearest, address::clamp_to_edge);

    const float2 currentTextureCoordinates = inputData.textureCoordinates;
    const float4 currentFrameColor = sceneTexture.sample(pointClampSampler, currentTextureCoordinates);
    
    if (postUniforms.historyValid < 0.5) return currentFrameColor;

    const float3 cameraWorldPosition = float3(viewUniforms.camX, viewUniforms.camY, viewUniforms.camZ);
    const Reprojected reprojectedData = reproject(currentTextureCoordinates, postUniforms, viewUniforms.viewProj, cameraWorldPosition);
    
    if (!reprojectedData.isValidReprojection) return currentFrameColor;

    float3 neighborhoodMinColor = currentFrameColor.rgb;
    float3 neighborhoodMaxColor = currentFrameColor.rgb;
    
    const float2 texelDimensions = 1.0 / float2(sceneTexture.get_width(), sceneTexture.get_height());
    
    for (int yOffset = -1; yOffset <= 1; ++yOffset) {
        for (int xOffset = -1; xOffset <= 1; ++xOffset) {
            if (xOffset == 0 && yOffset == 0) continue;
            const float3 neighborSampleColor = sceneTexture.sample(pointClampSampler,
                                                   currentTextureCoordinates + texelDimensions * float2(xOffset, yOffset)).rgb;
            neighborhoodMinColor = min(neighborhoodMinColor, neighborSampleColor);
            neighborhoodMaxColor = max(neighborhoodMaxColor, neighborSampleColor);
        }
    }

    const float3 averageNeighborhoodColor = (neighborhoodMinColor + neighborhoodMaxColor) * 0.5;
    const float3 neighborhoodColorExtent = (neighborhoodMaxColor - neighborhoodMinColor) * 0.5 * PCB_HISTORY_CLAMP;
    
    const float3 clampedHistoryColor = clamp(historyTexture.sample(linearClampSampler, reprojectedData.textureCoordinates).rgb,
                                 averageNeighborhoodColor - neighborhoodColorExtent, averageNeighborhoodColor + neighborhoodColorExtent);
                                 
    return float4(mix(clampedHistoryColor, currentFrameColor.rgb, postUniforms.historyBlend), currentFrameColor.a);
}

// ── Depth of field ────────────────────────────────────────────────────────

inline float distanceAtRow(float screenRowTextureCoordinate, constant PCBPostUniforms &postUniforms)
{
    const float reciprocalDistance = mix(postUniforms.invFar, postUniforms.invNear, saturate(screenRowTextureCoordinate));
    return 1.0 / max(reciprocalDistance, 1e-6);
}

inline float softKnee(float interpolationValue)
{
    return interpolationValue * interpolationValue * (3.0 - 2.0 * interpolationValue);
}

static float circleOfConfusion(float screenRowTextureCoordinate, constant PCBPostUniforms &postUniforms)
{
    float planeDistance = distanceAtRow(screenRowTextureCoordinate, postUniforms);
    float circleOfConfusionValue = (planeDistance - postUniforms.focusDist) / planeDistance;
    
    if (circleOfConfusionValue < 0.0)
        circleOfConfusionValue /= PCB_COC_NEAR_RAMP;
    else
        circleOfConfusionValue /= PCB_COC_FAR_RAMP;
        
    circleOfConfusionValue *= postUniforms.cocScale;
    circleOfConfusionValue = clamp(circleOfConfusionValue, -postUniforms.cocMax, postUniforms.cocMax);
    
    float circleOfConfusionSign = sign(circleOfConfusionValue);
    circleOfConfusionValue = abs(circleOfConfusionValue);
    circleOfConfusionValue /= postUniforms.cocMax;
    circleOfConfusionValue = softKnee(circleOfConfusionValue);
    circleOfConfusionValue = sqrt(circleOfConfusionValue);
    
    const float depthOfFieldDeadZone = 0.01;

    if (circleOfConfusionValue < depthOfFieldDeadZone)
        circleOfConfusionValue = 0.0;

    const float quantizationLevels = 128.0;
    circleOfConfusionValue = round(circleOfConfusionValue * quantizationLevels) / quantizationLevels;
    circleOfConfusionValue *= circleOfConfusionSign;

    return circleOfConfusionValue;
}

static float4 blurAlongAxis(texture2d<float> sourceTexture, float2 centerTextureCoordinate, float2 blurAxisDirection,
                       float blurRadiusPixels)
{
    constexpr sampler linearClampSampler(filter::linear, address::clamp_to_edge);
    if (blurRadiusPixels < 0.75) return sourceTexture.sample(linearClampSampler, centerTextureCoordinate);

    const float distanceBetweenTaps = blurRadiusPixels / float(PCB_DOF_TAPS);
    const float gaussianSigmaValue = max(blurRadiusPixels * PCB_DOF_SIGMA, 0.5);
    const float2 textureCoordinateStride = blurAxisDirection * distanceBetweenTaps;

    float4 accumulatedColorValue = sourceTexture.sample(linearClampSampler, centerTextureCoordinate);
    float totalSampleWeight = 1.0;
    
    for (int tapIndex = 1; tapIndex <= PCB_DOF_TAPS; ++tapIndex) {
        const float tapOffsetDistance = float(tapIndex) * distanceBetweenTaps;
        const float gaussianWeightValue = exp(-0.5 * tapOffsetDistance * tapOffsetDistance / (gaussianSigmaValue * gaussianSigmaValue));
        
        for (int kernelSideMultiplier = -1; kernelSideMultiplier <= 1; kernelSideMultiplier += 2) {
            const float2 sampleTextureCoordinate = centerTextureCoordinate + textureCoordinateStride * float(tapIndex * kernelSideMultiplier);
            const float appliedSampleWeight = gaussianWeightValue;
            
            accumulatedColorValue += appliedSampleWeight * sourceTexture.sample(linearClampSampler, sampleTextureCoordinate);
            totalSampleWeight += appliedSampleWeight;
        }
    }
    return accumulatedColorValue / max(totalSampleWeight, 1e-6);
}

fragment float4 depthOfFieldHorizontal(FullscreenVary inputData [[stage_in]],
                              texture2d<float> sourceTexture [[texture(0)]],
                              constant PCBPostUniforms &postUniforms [[buffer(0)]])
{
    const float2 currentTextureCoordinate = inputData.textureCoordinates;
    const float computedBlurRadius = fabs(circleOfConfusion(currentTextureCoordinate.y, postUniforms)) * PCB_DOF_MAX_RADIUS
                       * postUniforms.dofStrength;
                       
    return blurAlongAxis(sourceTexture, currentTextureCoordinate, float2(1.0 / sourceTexture.get_width(), 0.0),
                    computedBlurRadius);
}

fragment float4 depthOfFieldVertical(FullscreenVary inputData [[stage_in]],
                              texture2d<float> sourceTexture [[texture(0)]],
                              constant PCBPostUniforms &postUniforms [[buffer(0)]])
{
    const float2 currentTextureCoordinate = inputData.textureCoordinates;
    const float computedBlurRadius = fabs(circleOfConfusion(currentTextureCoordinate.y, postUniforms)) * PCB_DOF_MAX_RADIUS
                       * postUniforms.dofStrength;
                       
    return blurAlongAxis(sourceTexture, currentTextureCoordinate, float2(0.0, 1.0 / sourceTexture.get_height()),
                    computedBlurRadius);
}

fragment float4 bloomPrefilter(FullscreenVary inputData [[stage_in]],
                               texture2d<float> sourceTexture [[texture(0)]])
{
    constexpr sampler bilinearClampSampler(filter::linear, address::clamp_to_edge);

    float2 texelDimensions = 1.0 / float2(sourceTexture.get_width(), sourceTexture.get_height());
    float2 offsetTopLeft     = texelDimensions * float2(-Bloom::tapOffset, -Bloom::tapOffset);
    float2 offsetTopRight    = texelDimensions * float2( Bloom::tapOffset, -Bloom::tapOffset);
    float2 offsetBottomLeft  = texelDimensions * float2(-Bloom::tapOffset,  Bloom::tapOffset);
    float2 offsetBottomRight = texelDimensions * float2( Bloom::tapOffset,  Bloom::tapOffset);

    float3 accumulatedFilterColor = sourceTexture.sample(bilinearClampSampler, inputData.textureCoordinates + offsetTopLeft).rgb
                            + sourceTexture.sample(bilinearClampSampler, inputData.textureCoordinates + offsetTopRight).rgb
                            + sourceTexture.sample(bilinearClampSampler, inputData.textureCoordinates + offsetBottomLeft).rgb
                            + sourceTexture.sample(bilinearClampSampler, inputData.textureCoordinates + offsetBottomRight).rgb;

    float3 averageColor = accumulatedFilterColor * Bloom::boxAverage;

    float pixelLuminance = dot(averageColor, Bloom::lumaWeights);
    float highlightSelectionMask = smoothstep(Bloom::kneeStart, Bloom::kneeEnd, pixelLuminance);
    
    return float4(averageColor * highlightSelectionMask, 1.0);
}

static float3 gaussianBlur(texture2d<float> sourceTexture,
                           sampler textureSampler,
                           float2 centerTextureCoordinate,
                           float2 blurStepDirection)
{
    float3 accumulatedColor = sourceTexture.sample(textureSampler, centerTextureCoordinate).rgb * Bloom::gauss[0];
    for (int tapIndex = 1; tapIndex < 5; tapIndex++) {
        float2 sampleOffsetDirection = blurStepDirection * float(tapIndex);
        accumulatedColor += sourceTexture.sample(textureSampler, centerTextureCoordinate + sampleOffsetDirection).rgb * Bloom::gauss[tapIndex];
        accumulatedColor += sourceTexture.sample(textureSampler, centerTextureCoordinate - sampleOffsetDirection).rgb * Bloom::gauss[tapIndex];
    }
    return accumulatedColor;
}

fragment float4 bloomBlurHorizontal(FullscreenVary inputData [[stage_in]],
                                    texture2d<float> sourceTexture [[texture(0)]])
{
    constexpr sampler bilinearClampSampler(filter::linear, address::clamp_to_edge);
    float horizontalStepSize = Bloom::blurStepTexels / float(sourceTexture.get_width());
    return float4(gaussianBlur(sourceTexture, bilinearClampSampler,
                               inputData.textureCoordinates, float2(horizontalStepSize, 0.0)), 1.0);
}

fragment float4 bloomBlurVertical(FullscreenVary inputData [[stage_in]],
                                  texture2d<float> sourceTexture [[texture(0)]])
{
    constexpr sampler bilinearClampSampler(filter::linear, address::clamp_to_edge);
    float verticalStepSize = Bloom::blurStepTexels / float(sourceTexture.get_height());
    return float4(gaussianBlur(sourceTexture, bilinearClampSampler,
                               inputData.textureCoordinates, float2(0.0, verticalStepSize)), 1.0);
}

// ── Composite ─────────────────────────────────────────────────────────────

static float3 tonemapACES(float3 highDynamicRangeColor)
{
    float3 tonemapNumerator = highDynamicRangeColor * (Tonemap::coefficientA * highDynamicRangeColor + Tonemap::coefficientB);
    float3 tonemapDenominator = highDynamicRangeColor * (Tonemap::coefficientC * highDynamicRangeColor + Tonemap::coefficientD) + Tonemap::coefficientE;
    return saturate(tonemapNumerator / tonemapDenominator);
}

fragment float4 compositeFragment(FullscreenVary inputData [[stage_in]],
                                  texture2d<float> baseSceneTexture [[texture(0)]],
                                  texture2d<float> bloomTexture [[texture(1)]],
                                  constant PCBPostUniforms &postUniforms [[buffer(0)]])
{
    constexpr sampler bilinearClampSampler(filter::linear, address::clamp_to_edge);

    float3 baseSceneColor = baseSceneTexture.sample(bilinearClampSampler, inputData.textureCoordinates).rgb;
    float3 filteredBloomColor = bloomTexture.sample(bilinearClampSampler, inputData.textureCoordinates).rgb;
    float3 combinedHighDynamicRangeColor = baseSceneColor + filteredBloomColor * postUniforms.bloomIntensity;

    float pixelLuminance = dot(combinedHighDynamicRangeColor, Bloom::lumaWeights);
    float3 saturationAdjustedColor = max(mix(float3(pixelLuminance), combinedHighDynamicRangeColor, postUniforms.saturation), 0.0);

    return float4(tonemapACES(saturationAdjustedColor * postUniforms.exposure), 1.0);
}
