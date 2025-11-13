// render_data.h
#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>

namespace vkglTF {
    class Model;
}

using RenderableID = uint32_t;
static constexpr RenderableID INVALID_RENDERABLE = ~RenderableID(0);

struct TransformData {
    glm::mat4 model;
    glm::mat4 normalMat;
};

enum class RenderableType {
    None,
    Gltf,
    Scene,
    Obj,
    Skybox,
    Light,
    Ghost
};

struct DrawIndirect {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t  vertexOffset;
    uint32_t firstInstance; // index into transform array (PerObject)
};


struct MeshInstance {
    RenderableID id;
    uint32_t meshIndex;
    uint32_t materialIndex;
    TransformData transform;
    float boundingSphereRadius;
    glm::vec3 boundingSphereCenterModelSpace; // object-space center
    glm::vec3 aabbMinWorld;
	glm::vec3 aabbMaxWorld;
    RenderableType type = RenderableType::None;
    vkglTF::Model* model = nullptr;
    glm::vec3 emissiveColor{ 1.0f };
    float intensity = 1.0f;
};