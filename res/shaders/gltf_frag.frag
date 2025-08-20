#version 450
#extension GL_KHR_vulkan_glsl : enable

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
    vec4 ambientLightColor; // .rgb = color, .a = intensity (how you use it is up to you)
    vec4 viewPos;
    PointLight pointLights[10];
    int numLights;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D normalMap;
layout(set = 2, binding = 2) uniform sampler2D metallicMap;
layout(set = 2, binding = 3) uniform sampler2D roughnessMap;
layout(set = 2, binding = 4) uniform sampler2D aoMap;

layout(set = 3, binding = 0) uniform sampler2D brdfLUT;
layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;

layout(constant_id = 0) const bool alphaMask = false;
layout(constant_id = 1) const float cutoff = 0.5;

// DEBUG: set to 0 for normal, 1 = Lo (direct), 2 = diffuseIBL, 3 = specularIBL, 4 = albedo
layout(constant_id = 2) const int DEBUG_OUTPUT = 0;

const float PI = 3.14159265359;
const float EXPOSURE = 1.0; // tweak this if things are too dark/light

// utility
vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(c, vec3(1.0/2.2)); }

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N,H),0.0);
    float denom = NdotH*NdotH*(a2-1.0)+1.0;
    return a2 / (PI * denom*denom);
}
float GeometrySchlickGGX(float NdotV, float roughness) {
    float k = (roughness+1.0)*(roughness+1.0)/8.0;
    return NdotV / (NdotV*(1.0-k)+k);
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N,V),0.0), roughness) *
           GeometrySchlickGGX(max(dot(N,L),0.0), roughness);
}

void main() {
    // --- albedo/alpha ---
    vec4 baseColorTex = texture(albedoMap, inUV);
    vec3 albedo = srgbToLinear(baseColorTex.rgb); // assume your textures are sRGB -> convert to linear if required
    float alpha = baseColorTex.a * inColor.a;
    if(alphaMask && alpha < cutoff) discard;

    // --- material ---
    float metallic = texture(metallicMap, inUV).r;
    float roughness = clamp(texture(roughnessMap, inUV).r, 0.04, 1.0);
    float ao = texture(aoMap, inUV).r;

    // --- normal ---
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = cross(N, T) * inTangent.w;
    mat3 TBN = mat3(T, B, N);
    vec3 sampledNormal = texture(normalMap, inUV).xyz * 2.0 - 1.0;
    N = normalize(TBN * sampledNormal);

    // --- view/reflection ---
    vec3 V = normalize(inViewVec);
    vec3 R = reflect(-V, N);

    // --- F0 ---
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // --- direct lighting (Lo) ---
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < ubo.numLights; ++i) {
        vec3 L = normalize(ubo.pointLights[i].position.xyz - inWorldPos);
        vec3 H = normalize(V + L);
        float distance = length(ubo.pointLights[i].position.xyz - inWorldPos);
        float attenuation = 1.0 / max(0.0001, distance * distance);
        vec3 radiance = ubo.pointLights[i].color.rgb * ubo.pointLights[i].color.a * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.001) * max(dot(N, L), 0.001);
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // --- IBL ---
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuseIBL = irradiance * albedo;

    int mipCount = textureQueryLevels(prefilteredMap);
    float maxLod = max(0.0, float(mipCount - 1));
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * maxLod).rgb;
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specularIBL = prefilteredColor * (F0 * brdf.x + brdf.y);

    vec3 ambientIBL = (diffuseIBL + specularIBL) * ao;

    // --- Compose final (FIXED: do NOT multiply whole color by ambientLightColor) ---
    // ambientLightColor is additive ambient (if you use it); it should not zero Lo.
    vec3 ambientAdd = ubo.ambientLightColor.rgb * ubo.ambientLightColor.a; // small additive ambient
    vec3 color = Lo + ambientIBL + ambientAdd;

    // --- Debug outputs (quick) ---
    if (DEBUG_OUTPUT == 1) { outFragColor = vec4(linearToSrgb(Lo * EXPOSURE), alpha); return; }
    if (DEBUG_OUTPUT == 2) { outFragColor = vec4(linearToSrgb(diffuseIBL * EXPOSURE), alpha); return; }
    if (DEBUG_OUTPUT == 3) { outFragColor = vec4(linearToSrgb(specularIBL * EXPOSURE), alpha); return; }
    if (DEBUG_OUTPUT == 4) { outFragColor = vec4(linearToSrgb(albedo), alpha); return; }

    // --- tone map & gamma ---
    color *= EXPOSURE;
    color = color / (color + vec3(1.0)); // simple Reinhard
    color = linearToSrgb(color);

    outFragColor = vec4(color, alpha);
}
