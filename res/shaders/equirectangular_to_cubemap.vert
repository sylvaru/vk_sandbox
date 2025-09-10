#version 450

layout(location = 0) in vec3 inPosition; // cube vertex positions (direction)

layout(push_constant) uniform PC {
    mat4 proj;
    mat4 view;
} pc;

layout(location = 0) out vec3 vWorldPos;

void main() {
    vWorldPos = inPosition;
    gl_Position = pc.proj * pc.view * vec4(inPosition, 1.0);
}
