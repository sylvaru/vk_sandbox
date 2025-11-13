// gltf_pbr.frag
#version 450
#extension GL_KHR_vulkan_glsl : enable
#include "include/tonemapping.glsl"

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

layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D normalMap;
layout(set = 2, binding = 2) uniform sampler2D metallicMap;
layout(set = 2, binding = 3) uniform sampler2D roughnessMap;
layout(set = 2, binding = 4) uniform sampler2D aoMap;
layout(set = 2, binding = 5) uniform sampler2D emissiveMap;

layout(set = 3, binding = 0) uniform sampler2D brdfLUT;
layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    // Flip Y for material UV
    vec2 uvMaterial = vec2(inUV.x, 1.0 - inUV.y);

    // --- Albedo ---
    vec4 albedoSample = texture(albedoMap, uvMaterial);
    vec3 albedo = albedoSample.rgb * inColor.rgb;
    float alpha = albedoSample.a * inColor.a;

    // --- Metallic / Roughness / AO ---
    float metallic = texture(metallicMap, uvMaterial).r;
    float roughness = clamp(texture(roughnessMap, uvMaterial).r, 0.04, 1.0);
    float ao = texture(aoMap, uvMaterial).r;

    // --- Normal mapping ---
    vec3 tNormal = texture(normalMap, uvMaterial).xyz * 2.0 - 1.0;
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = normalize(cross(N, T) * inTangent.w);
    mat3 TBN = mat3(T, B, N);
    vec3 worldNormal = normalize(TBN * tNormal);

    // --- View vector & reflection ---
    vec3 V = normalize(inViewVec);
    float NdotV = clamp(dot(worldNormal, V), 0.0, 1.0);
    vec3 R = normalize(reflect(-V, worldNormal));

    // --- Fresnel & diffuse/specular factors ---
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 kS = fresnelSchlick(NdotV, F0);
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    // --- IBL ---
    vec3 irradiance = texture(irradianceMap, worldNormal).rgb;
    vec3 diffuseIBL = irradiance * albedo;
    int mipCount = textureQueryLevels(prefilteredMap);
    float maxLod = float(max(0, mipCount - 1));
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * maxLod).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefilteredColor * (kS * brdf.x + brdf.y);

    vec3 emissive = texture(emissiveMap, uvMaterial).rgb;

    // --- Compose final color ---
    vec3 diffuseTerm = diffuseIBL * kD * ao;
    vec3 pbrColor = diffuseTerm + specularIBL + ubo.ambientLightColor.rgb * ubo.ambientLightColor.a + emissive;

    // ACES tone mapping + gamma
    const float EXPOSURE = 0.3;
    vec3 color = ACESFilm(pbrColor * EXPOSURE);
    color = toSRGB(color);

    outFragColor = vec4(color, alpha);
}
