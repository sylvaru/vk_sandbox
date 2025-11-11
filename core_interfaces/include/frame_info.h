#pragma once
// frame_info.h
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include <memory>
#include "interfaces/camera_i.h"
#include "interfaces/scene_i.h"
#include "interfaces/game_object_i.h"


class RenderableRegistry;

#define MAX_LIGHTS 10
struct IGameObject;
struct IScene;

struct PointLight {
    glm::vec4 position{};
    glm::vec4 color{};
};

struct GlobalUbo {
    glm::mat4 projection{ 1.f };
    glm::mat4 view{ 1.f };
    glm::vec4 ambientLightColor{ 0.0f,1.0f,0.0f,0.0f };
    glm::vec4 viewPos{ 0.f };
    PointLight pointLights[MAX_LIGHTS];
    alignas(4) int numLights = 0;
    // pad up to 16‑byte multiple:
    alignas(16) int _pad[3]{};
};

struct FrameInfo {
    int frameIndex;
    float frameTime;
    VkCommandBuffer commandBuffer;
    const ICamera& camera;
    VkDescriptorSet globalDescriptorSet;
    std::unordered_map<unsigned int, std::shared_ptr<IGameObject>>& gameObjects;
    IScene& scene;
    const RenderableRegistry* renderRegistry = nullptr;
};

