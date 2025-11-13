// equirectangular_to_cube.frag
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 worldDir;
layout(location = 1) in vec3 vNormal;

layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D equirectMap;

const float PI = 3.14159265359;

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
         c, 0, -s,
         0, 1,  0,
         s, 0,  c
    );
}

void main()
{
    // debug guard: if either input is NaN/zero you will notice it quickly
    vec3 N = normalize(worldDir);

    // Detect top / bottom faces using the vertex normal (model's face normal)
    // Adjust thresholds if your normals are slightly off axis.

    // spherical mapping
    float phi   = atan(N.z, N.x);
    float theta = asin(clamp(N.y, -1.0, 1.0));
    vec2 uv = vec2(phi / (2.0 * PI) + 0.5, 0.5 - theta / PI);

    vec3 color = texture(equirectMap, uv).rgb;
    outColor = vec4(color, 1.0);
}
