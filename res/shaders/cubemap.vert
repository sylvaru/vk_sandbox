// cubemap.vert
#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal; // ensure pipeline provides this

layout(push_constant) uniform PushBlock {
    mat4 mvp;
} pushBlock;

// explicit locations
layout(location = 0) out vec3 worldDir;
layout(location = 1) out vec3 vNormal;

void main() {
    worldDir = inPos;      // direction used for sampling
    vNormal  = inNormal;   // used to detect top/bottom faces
    gl_Position = pushBlock.mvp * vec4(inPos, 1.0);
}
