#version 450
#extension GL_KHR_vulkan_glsl : enable

// Interpolated inputs (from vertex)
layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in vec4 inColor;
layout(location = 5) in vec3 inViewVec;

layout(location = 0) out vec4 outFragColor;

// Keep your GlobalUbo exactly the same (set 0 binding 0)
struct PointLight { vec4 position; vec4 color; };
layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor;
    vec4 viewPos;
    PointLight pointLights[10];
    int numLights;
} ubo;

// Additional Sascha-style scene params (optional, set 0 binding 1)
// If you prefer, feed exposure/gamma/prefiltered mip levels and scaleIBLAmbient from CPU into this.
layout(set = 0, binding = 1) uniform UBOParams {
    vec4 lightDir;                // .xyz = directional light direction (world), .w unused
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float debugViewInputs;
    float debugViewEquation;
} uboParams;

// PBR material maps (set = 2)
layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D normalMap;
layout(set = 2, binding = 2) uniform sampler2D mrMap; // metallic-roughness packed

// IBL textures live at set = 3
layout(set = 3, binding = 0) uniform sampler2D   brdfLUT;
layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;

// specialization constants (same as your original)
layout(constant_id = 0) const bool alphaMask = false;
layout(constant_id = 1) const float cutoff = 0.5;
layout(constant_id = 2) const int DEBUG_OUTPUT = 0;

const float PI = 3.14159265359;
const float MIN_ROUGHNESS = 0.04;

// Utility: sRGB <-> Linear (keep your conversion)
vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(c, vec3(1.0/2.2)); }

// Fresnel + microfacet helpers (based on your functions + Sascha conventions)
vec3 fresnelSchlick(float cosTheta, vec3 F0) { return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0); }
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

// Sascha-style PBRInfo struct (useful for IBL function)
struct PBRInfo {
    float NdotL;
    float NdotV;
    float NdotH;
    float LdotH;
    float VdotH;
    float perceptualRoughness;
    float metalness;
    vec3 reflectance0;
    vec3 reflectance90;
    float alphaRoughness;
    vec3 diffuseColor;
    vec3 specularColor;
};

// get normal helper: if user provides a normalMap, use it; if tangent available, use tangent; otherwise use derivatives
vec3 getNormalTBN(vec3 N, vec4 tangentAttr, vec2 uv) {
    // tangentAttr.xyz = tangent in world-space, .w = bitangent sign
    // If a normal map is not bound or not used, caller should return N
    vec3 T = normalize(tangentAttr.xyz);
    vec3 B = cross(N, T) * tangentAttr.w;
    mat3 TBN = mat3(T, B, N);
    vec3 sampled = texture(normalMap, uv).xyz * 2.0 - 1.0;
    return normalize(TBN * sampled);
}

vec3 getNormalDeriv(vec3 N, vec2 uv) {
    // fallback if tangent is not reliable: compute tangent using derivatives (Sascha approach)
    vec3 q1 = dFdx(inWorldPos);
    vec3 q2 = dFdy(inWorldPos);
    vec2 st1 = dFdx(uv);
    vec2 st2 = dFdy(uv);
    vec3 T = normalize(q1 * st2.t - q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
    vec3 sampled = texture(normalMap, uv).xyz * 2.0 - 1.0;
    return normalize(TBN * sampled);
}

// IBL contribution (Sascha style)
vec3 getIBLContribution(PBRInfo pbrInputs, vec3 n, vec3 reflection)
{
    float lod = (pbrInputs.perceptualRoughness * uboParams.prefilteredCubeMipLevels);
    // retrieve scale and bias to F0 from BRDF LUT (x = NdotV, y = roughness)
    vec3 brdf = texture(brdfLUT, vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness)).rgb;
    // irradiance & prefiltered maps are assumed to be sRGB stored; convert to linear as you used earlier
    vec3 diffuseLight = srgbToLinear(texture(irradianceMap, n).rgb);
    vec3 specularLight = srgbToLinear(textureLod(prefilteredMap, reflection, lod).rgb);

    vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
    vec3 specular = specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);

    diffuse *= uboParams.scaleIBLAmbient;
    specular *= uboParams.scaleIBLAmbient;

    return diffuse + specular;
}

// Simple Uncharted2 tonemap (copied from your shader)
vec3 Uncharted2Tonemap(vec3 x) {
    float A = 0.15; float B = 0.50; float C = 0.10;
    float D = 0.20; float E = 0.02; float F = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
// --- Push constants (Sascha style) ---
layout(push_constant) uniform PushConstants {
    int meshIndex;
    int materialIndex;
} pushConstants;


void main() {
    // --- albedo/alpha
    vec4 baseColorTex = texture(albedoMap, inUV);
    vec3 albedo = srgbToLinear(baseColorTex.rgb); // keep sRGB -> linear use
    float alpha = baseColorTex.a * inColor.a;
    if (alphaMask && alpha < cutoff) discard;

    // --- material: metallicRoughness map (g = roughness, b = metal)
    vec3 mrSample = texture(mrMap, inUV).rgb;
    float perceptualRoughness = mrSample.g;
    float metallic  = mrSample.b;
    perceptualRoughness = clamp(perceptualRoughness, MIN_ROUGHNESS, 1.0);
    float alphaRoughness = perceptualRoughness * perceptualRoughness;
    float roughness = alphaRoughness;

    // --- normal: prefer normalMap + TBN from vertex tangent; fallback to derivs if tangent invalid
    vec3 N = normalize(inNormal);
    bool useNormalMap = true; // if you may run objects without normal maps, gate by a material flag on CPU
    if (useNormalMap) {
        // if tangent available (non-zero), use TBN built from tangent; else use derivative-based TBN (Sascha)
        if (length(inTangent.xyz) > 0.0) {
            N = getNormalTBN(N, inTangent, inUV);
        } else {
            N = getNormalDeriv(N, inUV);
        }
    }

    // --- view/reflection
    vec3 V = normalize(ubo.viewPos.xyz - inWorldPos); // camera vector (world-space)
    vec3 R = reflect(-V, N);

    // --- F0 base reflectance
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // --- direct lighting (keep your point lights loop, but use PBR terms)
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

    // --- IBL
    vec3 irradiance = srgbToLinear(texture(irradianceMap, N).rgb);
    vec3 diffuseIBL = irradiance * albedo;

    int mipCount = textureQueryLevels(prefilteredMap);
    float maxLod = max(0.0, float(mipCount - 1));
    vec3 prefilteredColor = srgbToLinear(textureLod(prefilteredMap, R, perceptualRoughness * maxLod).rgb);
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), perceptualRoughness)).rg;
    vec3 specularIBL = prefilteredColor * (F0 * brdf.x + brdf.y);

    vec3 ambientIBL = (diffuseIBL + specularIBL) * uboParams.scaleIBLAmbient;
    vec3 ambientAdd = ubo.ambientLightColor.rgb * ubo.ambientLightColor.a;

    vec3 color = Lo + ambientIBL + ambientAdd;

    // --- tonemap & gamma (use uboParams.exposure/gamma if provided)
    float exposure = uboParams.exposure > 0.0 ? uboParams.exposure : 1.0;
    color *= exposure;
    color = Uncharted2Tonemap(color);
    // normalization constant from Uncharted2 (keeps look consistent)
    color = color * (1.0 / Uncharted2Tonemap(vec3(11.2)));

    // sRGB conversion
    color = linearToSrgb(color);

    outFragColor = vec4(color, 1.0);

    // optional debug outputs (if you want them enabled)
    if (DEBUG_OUTPUT != 0 || int(uboParams.debugViewInputs) != 0 || int(uboParams.debugViewEquation) != 0) {
        // you can replicate Sascha's debug switches here if desired.
        // I left this stub so you can wire debug parameters from CPU into uboParams.debugView*
    }
}
