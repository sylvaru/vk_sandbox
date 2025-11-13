// skybox_ibl.vert
#version 450 core

layout(location = 0) in vec3 position;

//— Scene UBO (set 0)
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


layout(location = 0) out vec3 vDirection;

void main() {
    // remove translation:
    mat4 rotView = mat4(mat3(ubo.view));
    
    vec4 clipPos = ubo.projection * rotView * vec4(position, 1.0);
    gl_Position = clipPos.xyww; // Depth = 1.0 (far plane)
    
    vDirection = position; 
}
