// sandbox_renderer/include/IRenderSystem.h
#pragma once
#include <vulkan/vulkan.h>
#include "renderer_i.h"
#include "global_common/frame_info.h"

class VkSandboxDevice;
class VkSandboxDescriptorPool;

struct IRenderSystem {
public:
    virtual ~IRenderSystem() = default;


    virtual void init(
        VkSandboxDevice& device,
        VkRenderPass            renderPass,
        VkDescriptorSetLayout   globalSetLayout,
        VkSandboxDescriptorPool& descriptorPool)
    {};   

    virtual void update(FrameInfo& frame, GlobalUbo& ubo) {
  
    }

    virtual void render(FrameInfo& frame) {}


};