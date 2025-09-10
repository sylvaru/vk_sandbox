#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3  inPos;
layout(location = 1) in vec3  inNormal;
layout(location = 2) in vec2  inUV;
layout(location = 3) in vec4  inColor;
layout(location = 4) in vec4  inTangent;
layout(location = 5) in uvec4 inJoint0;   // for skinning
layout(location = 6) in vec4  inWeight0;

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

#define MAX_NUM_JOINTS 128

struct MeshShaderData {
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
    uint jointCount;
};

layout(std430, set = 3, binding = 0) readonly buffer MeshSSBO {
    MeshShaderData meshData[];
};

layout(push_constant) uniform PushConstants {
    int meshIndex;
    int materialIndex;
} pushConstants;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec4 outTangent;
layout(location = 4) out vec4 outColor;

void main() {
    MeshShaderData data = meshData[pushConstants.meshIndex];

    mat4 skinMat = mat4(1.0);
    if (data.jointCount > 0) {
        skinMat =
            inWeight0.x * data.jointMatrix[inJoint0.x] +
            inWeight0.y * data.jointMatrix[inJoint0.y] +
            inWeight0.z * data.jointMatrix[inJoint0.z] +
            inWeight0.w * data.jointMatrix[inJoint0.w];
    }

    vec4 worldPos = data.matrix * skinMat * vec4(inPos, 1.0);
    outWorldPos = worldPos.xyz;

    outNormal = normalize(mat3(data.matrix * skinMat) * inNormal);

    vec3 tangentWS = normalize(mat3(data.matrix * skinMat) * inTangent.xyz);
    outTangent = vec4(tangentWS, inTangent.w);

    outUV = inUV;
    outColor = inColor;

    gl_Position = ubo.projection * ubo.view * worldPos;
}
