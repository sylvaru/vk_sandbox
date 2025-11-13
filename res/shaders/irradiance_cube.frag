// irradiance_cube.frag
#version 450

// Vertex → Fragment interface
layout(location = 0) in vec3 inPos;
layout(location = 0) out vec4 outColor;

// Your environment map
layout(binding = 0) uniform samplerCube samplerEnv;

// Push‐constant block in std140 packing (offsets handled for you)
layout(push_constant, std140) uniform PushConsts {
    mat4 mvp;         // takes 0..63
    float deltaPhi;   // at offset 64
    float deltaTheta; // at offset 68
} consts;

#define PI 3.14159265358979323846

void main()
{
    vec3 N = normalize(inPos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, N));
    up = cross(N, right);

    const float TWO_PI  = PI * 2.0;
    const float HALF_PI = PI * 0.5;

    vec3 color = vec3(0.0);
    uint sampleCount = 0u;
    for (float phi = 0.0; phi < TWO_PI; phi += consts.deltaPhi) {
        for (float theta = 0.0; theta < HALF_PI; theta += consts.deltaTheta) {
            vec3 tangent   = cos(phi) * right + sin(phi) * up;
            vec3 sampleVec = cos(theta) * N   + sin(theta) * tangent;
            color += texture(samplerEnv, sampleVec).rgb * cos(theta) * sin(theta);
            sampleCount++;
        }
    }
    outColor = vec4(PI * color / float(sampleCount), 1.0);
}
