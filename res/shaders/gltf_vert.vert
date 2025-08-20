#version 450
#extension GL_KHR_vulkan_glsl : enable

// Vertex inputs
layout(location = 0) in vec3  inPos;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inUV;
layout(location = 3) in vec4  inColor;
layout(location = 4) in vec4  inTangent;   // .xyz = tangent, .w = bitangent sign

// Scene UBO (set 0)
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

// Per-object node UBO (set 1)
layout(set = 1, binding = 0) uniform PerNode {
    mat4 modelMatrix;
    mat4 normalMatrix;  // inverse-transpose of model
} perNode;

// Outputs
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec4 outTangent;
layout(location = 4) out vec4 outColor;
layout(location = 5) out vec3 outViewVec;

void main() {
    // Transform to world
    vec4 worldPos = perNode.modelMatrix * vec4(inPos, 1.0);
    outWorldPos = worldPos.xyz;

    // Normal and tangent to world
    outNormal   = normalize(mat3(perNode.normalMatrix) * inNormal);
    vec3 tangentWS = normalize(mat3(perNode.normalMatrix) * inTangent.xyz);
    outTangent = vec4(tangentWS, inTangent.w);

    // Pass-through
    outUV    = inUV;
    outColor = inColor;

    // Camera view vector (world-space)
    outViewVec = ubo.viewPos.xyz - worldPos.xyz;

    // Final clip-space position
    gl_Position = ubo.projection * ubo.view * worldPos;
}
