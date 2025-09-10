#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 vWorldPos;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D equirectMap;

const float PI = 3.14159265358979323846;

vec2 dirToEquirectUV(vec3 dir) {
    // note: choose the v flip to match your earlier fix (we used v = 0.5 + phi/pi)
    float theta = atan(dir.z, dir.x); // -PI..PI
    float phi = asin(clamp(dir.y, -1.0, 1.0)); // -PI/2..PI/2
    float u = 0.5 + theta / (2.0 * PI);
    float v = 0.5 + phi / PI; // matches the 'v flip' you applied earlier
    return vec2(u, v);
}

void main() {
    vec3 dir = normalize(vWorldPos);
    vec2 uv = dirToEquirectUV(dir);
    vec4 c = texture(equirectMap, uv);
    outColor = c;
}
