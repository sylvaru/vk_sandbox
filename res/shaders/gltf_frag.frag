// gltf_pbr.frag
#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_GOOGLE_include_directive : enable
#include "include/helpers.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in vec4 inColor;
layout(location = 5) in vec3 inViewVec;

layout(location = 0) out vec4 outFragColor;

struct PointLight { vec4 position; vec4 color; };

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor;
    vec4 viewPos;
    PointLight pointLights[10];
    int numLights;
} ubo;

// Material textures
layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D normalMap;
layout(set = 2, binding = 2) uniform sampler2D metallicMap;
layout(set = 2, binding = 4) uniform sampler2D aoMap;
layout(set = 2, binding = 5) uniform sampler2D emissiveMap;

// IBL
layout(set = 3, binding = 0) uniform sampler2D brdfLUT;
layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;


// Tweakable multipliers    //TODO: This should be configurable via imgui during runtime
const float ALBEDO_INTENSITY      = 1.0;
const float METALLIC_MULTIPLIER   = 1.0;
const float ROUGHNESS_MULTIPLIER  = 0.6;
const float AO_MULTIPLIER         = 0.04;
const float NORMAL_STRENGTH       = 1.0;

const float IBL_INTENSITY         = 1.0;
const float DIRECT_LIGHT_INTENSITY = 1.0;

const float EMISSIVE_INTENSITY    = 0.2;
const float EXPOSURE              = 1.0;


void main() {
    vec2 uvMaterial = vec2(inUV.x, 1.0 - inUV.y);

    // Albedo
    vec4 albedoSample = texture(albedoMap, uvMaterial);
    vec3 albedo = srgbToLinear(albedoSample.rgb * inColor.rgb) * ALBEDO_INTENSITY;
    float alpha = albedoSample.a * inColor.a;

    // Metallic / Roughness / AO
    vec3 mr = texture(metallicMap, uvMaterial).rgb;
    float roughness = clamp(mr.g * ROUGHNESS_MULTIPLIER, 0.04, 1.0);
    float metallic  = clamp(mr.b * METALLIC_MULTIPLIER, 0.0, 1.0);
    float ao = clamp(texture(aoMap, uvMaterial).r * AO_MULTIPLIER, 0.0, 1.0);


    // Normal Mapping
    vec3 tNormal = texture(normalMap, uvMaterial).xyz * 2.0 - 1.0;
    tNormal = normalize(mix(vec3(0,0,1), tNormal, NORMAL_STRENGTH));

    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = normalize(cross(N, T) * inTangent.w);
    vec3 worldNormal = normalize(mat3(T, B, N) * tNormal);

    // View
    vec3 V = normalize(inViewVec);
    float NdotV = max(dot(worldNormal, V), 0.0);

    // Fresnel base reflectance
    vec3 F0 = mix(vec3(0.04), albedo, metallic);


    // Direct light from point lights
    vec3 directLighting = vec3(0.0);

    for (int i = 0; i < ubo.numLights; i++) {

        vec3 L = normalize(ubo.pointLights[i].position.xyz - inWorldPos);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(worldNormal, L), 0.0);
        if (NdotL <= 0.0) continue;

        float NdotH = max(dot(worldNormal, H), 0.0);

        float D = DistributionGGX(NdotH, roughness);
        float G = GeometrySmith(NdotV, NdotL, roughness);
        vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);

        vec3 numerator = D * G * F;
        float denom = 4.0 * NdotV * NdotL + 0.001;
        vec3 specular = numerator / denom;

        vec3 radiance = ubo.pointLights[i].color.rgb * ubo.pointLights[i].color.a;

        directLighting += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    directLighting *= DIRECT_LIGHT_INTENSITY;


    // Image based lighting
    vec3 R = reflect(-V, worldNormal);

    vec3 irradiance = texture(irradianceMap, worldNormal).rgb;
    vec3 diffuseIBL = irradiance * albedo;

    int mipCount = textureQueryLevels(prefilteredMap);
    float maxLod = float(max(0, mipCount - 1));
    vec3 prefiltered = textureLod(prefilteredMap, R, roughness * maxLod).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;

    vec3 kS = fresnelSchlick(NdotV, F0);
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 specIBL = prefiltered * (kS * brdf.x + brdf.y);

    vec3 ambientIBL = (diffuseIBL * kD + specIBL) * ao * IBL_INTENSITY;

    // Emissive
    vec3 emissive = srgbToLinear(texture(emissiveMap, uvMaterial).rgb) * EMISSIVE_INTENSITY;


    // Final composite
    vec3 pbrColor = ambientIBL + directLighting + emissive;

    vec3 color = ACESFilm(pbrColor * EXPOSURE);
    color = linearToSrgb(color);

    outFragColor = vec4(color, alpha);
}
