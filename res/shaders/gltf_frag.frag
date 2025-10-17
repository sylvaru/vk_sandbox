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

layout(set = 3, binding = 0) uniform sampler2D brdfLUT;
layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;

// Debug modes
// 0 = final, 1=albedo,2=irradiance,3=prefilteredColor,4=brdf,5=F0,6=kS,7=kD,8=diffuseTerm,9=specularIBL,
//10=worldNormal,11=roughness,12=metallic,13=ao
layout(constant_id = 0) const int DEBUG_MODE = 0;

// Specialization constants to force values (set via VkSpecializationInfo or leave default)
layout(constant_id = 1) const int FORCE_METALLIC = 0;
layout(constant_id = 2) const float METALLIC_VALUE = 0.0;
layout(constant_id = 3) const int FORCE_ROUGHNESS = 0;
layout(constant_id = 4) const float ROUGHNESS_VALUE = 0.5;
layout(constant_id = 5) const int FORCE_AO = 0;
layout(constant_id = 6) const float AO_VALUE = 1.0;
layout(constant_id = 7) const int FLIP_MATERIAL_UV = 1; // 1 = flip material UVs, 0 = don't
layout(constant_id = 8) const int FLIP_ENV_MAP_Y = 1; // 0 = no flip, 1 = flip Y for env sampling

const float PI = 3.14159265359;

// Utilities
vec3 linearToSrgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }
vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    // flip y
    vec2 uvNoFlip = inUV;
    vec2 uvMaterial = (FLIP_MATERIAL_UV != 0) ? vec2(inUV.x, 1.0 - inUV.y) : inUV;


    // --- Albedo ---
    vec4 albedoSample = texture(albedoMap, uvMaterial);
    vec3 albedo = albedoSample.rgb * inColor.rgb;
    float alpha = albedoSample.a * inColor.a;

    // --- MR/AO sampling ---
    float texMetallic = texture(metallicMap, uvMaterial).r;
    float texRoughness = texture(roughnessMap, uvMaterial).r;
    float texAO = texture(aoMap, uvMaterial).r;

    float metallic = (FORCE_METALLIC != 0) ? METALLIC_VALUE : texMetallic;
    float roughness = (FORCE_ROUGHNESS != 0) ? clamp(ROUGHNESS_VALUE, 0.04, 1.0) : clamp(texRoughness, 0.04, 1.0);
    float ao = (FORCE_AO != 0) ? AO_VALUE : texAO;

    // --- Normal (TBN) ---
    // sample normal map (assumes normal map is in tangent space, stored in [0,1] -> remap to [-1,1])
    vec3 tNormal = texture(normalMap, uvMaterial).xyz * 2.0 - 1.0;

    // Reconstruct TBN from vertex attributes
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = normalize(cross(N, T) * inTangent.w); // inTangent.w stores handedness
    mat3 TBN = mat3(T, B, N);

    // Transform normal from tangent space to world space
    vec3 worldNormal = normalize(TBN * tNormal);

    // view vector & reflection
    vec3 V = normalize(inViewVec);
    float NdotV = clamp(dot(worldNormal, V), 0.0, 1.0);
    vec3 R = normalize(reflect(-V, worldNormal));

    // Optionally flip Y of the reflection direction used for env/cubemap sampling
    vec3 envR = R;
    if (FLIP_ENV_MAP_Y != 0) {
        envR.y = -envR.y;
    }

    // --- F0 & fresnel ---
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 kS = fresnelSchlick(NdotV, F0);
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    // --- IBL DIFFUSE ---
    vec3 irradiance = texture(irradianceMap, worldNormal).rgb;
    vec3 diffuseIBL = irradiance * albedo;

    // --- IBL SPECULAR (split-sum approximation) ---
    int mipCount = textureQueryLevels(prefilteredMap); // number of mip levels
    float maxLod = max(0.0, float(mipCount - 1));
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * maxLod).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefilteredColor * (kS * brdf.x + brdf.y);

    // --- Compose ---
    vec3 diffuseTerm = diffuseIBL * kD * ao;
    vec3 ambientIBL = diffuseTerm + specularIBL;
    vec3 ambientAdd = ubo.ambientLightColor.rgb * ubo.ambientLightColor.a;
    vec3 pbrColor = ambientIBL + ambientAdd;

    // tone mapping + gamma (ACES-lite / simple Reinhard)
    vec3 color = pbrColor / (pbrColor + vec3(1.0));
    color = linearToSrgb(color);

    // Default / final output
    outFragColor = vec4(color, alpha);
}
