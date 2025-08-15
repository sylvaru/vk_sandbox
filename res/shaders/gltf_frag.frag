#version 450
#extension GL_KHR_vulkan_glsl : enable

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

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor; // vertex color or vertex alpha multiplier
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inViewVec;
layout(location = 4) in vec3 inWorldPos;
layout(location = 5) in vec4 inTangent;

layout(location = 0) out vec4 outFragColor;

struct PointLight {
    vec4 position;  // xyz = world-space position, w = unused
    vec4 color;     // xyz = RGB, w = intensity
};

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    vec4 ambientLightColor;
    vec4 viewPos;
    PointLight pointLights[10];
    int numLights;
} ubo;

// Constants for PBR
const float PI = 3.14159265359;

// Fresnel Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Normal Distribution Function: GGX/Trowbridge-Reitz
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

// Geometry function: Schlick-GGX
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

// Geometry Smith function
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

void main() {

   float alpha = texture(albedoMap, inUV).a * inColor.a;
    
    if (alphaMask && alpha < cutoff) {
        discard;
    }
    

    // Sample textures
    vec3 albedo = pow(texture(albedoMap, inUV).rgb, vec3(2.2)); // gamma to linear
    float metallic = texture(metallicMap, inUV).r;
    float roughness = texture(roughnessMap, inUV).r;
    float ao = texture(aoMap, inUV).r;

    // Normal mapping in tangent space
    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = cross(N, T) * inTangent.w;
    mat3 TBN = mat3(T, B, N);
    vec3 sampledNormal = texture(normalMap, inUV).xyz * 2.0 - 1.0;
    sampledNormal.y = -sampledNormal.y;
    N = normalize(TBN * sampledNormal);

    vec3 V = normalize(inViewVec);

    // Calculate reflectance at normal incidence; if dielectric use 0.04, else use albedo color
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Calculate kS and kD once here
    vec3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    // Direct lighting contribution from point lights
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < ubo.numLights; ++i) {
        vec3 L = normalize(ubo.pointLights[i].position.xyz - inWorldPos);
        vec3 H = normalize(V + L);
        float distance = length(ubo.pointLights[i].position.xyz - inWorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = ubo.pointLights[i].color.rgb * ubo.pointLights[i].color.a * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / denominator;

        float NdotL = max(dot(N, L), 0.0);

        // Add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Ambient lighting (IBL)
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-V, N);
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * MAX_REFLECTION_LOD).rgb;

    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;

    vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;

    vec3 color = Lo + ambient;

    // HDR tonemapping (simple Reinhard)
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    outFragColor = vec4(color, texture(albedoMap, inUV).a * inColor.a);
}
