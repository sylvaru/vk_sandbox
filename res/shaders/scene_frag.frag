// scene.frag
#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_GOOGLE_include_directive : enable
#include "include/helpers.glsl"

// Material
layout(set = 2, binding = 0) uniform sampler2D materialSampler;
layout(set = 2, binding = 1) uniform sampler2D normalSampler;

// IBL (new)
layout(set = 3, binding = 0) uniform sampler2D brdfLUT;
layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in vec4 inColor;
layout(location = 5) in vec3 inViewVec;


layout(location = 0) out vec4 outFragColor;

layout(constant_id = 0) const bool  ALPHA_MASK = false;
layout(constant_id = 1) const float ALPHA_MASK_CUTOFF = 0.0;

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor;
    vec4 viewPos;
    PointLight pointLights[10];
    int numLights;
} ubo;

// Constants
const float METALLIC_VALUE  = 1.0;
const float ROUGHNESS_VALUE = 1.0;
const float AO_VALUE        = 0.3;

const float DIRECT_LIGHT_INTENSITY = 0.29;
const float IBL_INTENSITY          = 1.0;
const float EXPOSURE               = 0.5; 



void main() {

    vec4 texColor = texture(materialSampler, inUV) * inColor;

    if (ALPHA_MASK && texColor.a < ALPHA_MASK_CUTOFF) {
        discard;
    }

    // Normal Mapping
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = cross(N, T) * inTangent.w;
    mat3 TBN = mat3(T, B, N);

    vec3 nMap = texture(normalSampler, inUV).xyz * 2.0 - 1.0;
    N = normalize(TBN * nMap);

    vec3 V = normalize(inViewVec);
    float NdotV = max(dot(N, V), 0.0);


    // Direct Light
    vec3 lighting = vec3(0.0);

    for (int i = 0; i < ubo.numLights; ++i) {

        vec3 L = normalize(ubo.pointLights[i].position.xyz - inWorldPos);
        vec3 H = normalize(L + V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        float NdotH = max(dot(N, H), 0.0);

        vec3 lightCol = ubo.pointLights[i].color.rgb * ubo.pointLights[i].color.a;

        // Simple Blinn-Phong for direct light
        float specPower = 32.0;
        float spec = pow(max(dot(H, N), 0.0), specPower);

        vec3 diffuse = texColor.rgb * NdotL * lightCol;
        vec3 specular = spec * lightCol;

        lighting += diffuse + specular;
    }

    lighting *= DIRECT_LIGHT_INTENSITY;


    // Image-Based Lighting
    float metallic  = METALLIC_VALUE;
    float roughness = clamp(ROUGHNESS_VALUE, 0.04, 1.0);
    float ao        = AO_VALUE;
    if (ALPHA_MASK) { ao = 0.02f; }
    vec3 albedo = srgbToLinear(texColor.rgb);

    // Base reflectance
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Diffuse IBL
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuseIBL = irradiance * albedo;

    // Specular IBL
    vec3 R = reflect(-V, N);

    int mipCount = textureQueryLevels(prefilteredMap);
    float maxLod = float(max(0, mipCount - 1));

    vec3 prefiltered = textureLod(prefilteredMap, R, roughness * maxLod).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;

    vec3 F = fresnelSchlick(NdotV, F0);
    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 specIBL = prefiltered * (kS * brdf.x + brdf.y);

    vec3 ambientIBL = (diffuseIBL * kD + specIBL) * ao * IBL_INTENSITY;

    // Final Composite
    vec3 color = lighting + ambientIBL;
    color = ACESFilm(color * EXPOSURE);

    outFragColor = vec4(color, texColor.a);
}
