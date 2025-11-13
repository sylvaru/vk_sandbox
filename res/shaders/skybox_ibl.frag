#version 450 core
#extension GL_KHR_vulkan_glsl : enable
#include "include/tonemapping.glsl"

layout(set = 1, binding = 0) uniform samplerCube skybox;

layout(location = 0) in vec3 vDirection;
layout(location = 0) out vec4 outColor;

const float DEFAULT_EXPOSURE = 0.2;

void main() {
    vec3 dir = normalize(vDirection);
    dir.y = -dir.y;

    vec3 hdr = texture(skybox, dir).rgb;

    // Apply exposure
    vec3 color = hdr * DEFAULT_EXPOSURE;

    // ACES tone mapping
    color = ACESFilm(color);

    // gamma
    color = toSRGB(color);

    outColor = vec4(color, 1.0);
}
