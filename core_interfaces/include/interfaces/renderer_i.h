#pragma once
#include <memory>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vector>
#include <glm/glm.hpp>
#include "frame_info.h"

class ISandboxRenderer {
public:
	struct FrameContext {
		std::vector<VkCommandBuffer> graphicsCommandBuffers;
		VkCommandBuffer primaryGraphicsCommandBuffer = VK_NULL_HANDLE;
		VkFence frameFence = VK_NULL_HANDLE;
		uint32_t frameIndex = 0;
		uint32_t imageIndex = 0;
		bool isValid() const { return primaryGraphicsCommandBuffer != VK_NULL_HANDLE; }
	};

	
	virtual ~ISandboxRenderer() = default;

	virtual void renderSystems(FrameInfo& info) {};
	virtual void renderSystems(FrameInfo& info, FrameContext& frame) {};
	virtual void updateSystems(FrameInfo& frame, GlobalUbo& ubo, float deltaTime) {};
	virtual FrameContext beginFrame() = 0;
	virtual void beginSwapChainRenderPass(FrameContext& frame) = 0;
	virtual void endSwapChainRenderPass(FrameContext& frame) = 0;
	virtual void endFrame(FrameContext& frame) = 0;

	virtual void waitDeviceIdle() = 0;
	
};