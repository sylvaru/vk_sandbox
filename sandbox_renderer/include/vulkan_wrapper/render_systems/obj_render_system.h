#pragma once
#include "interfaces/render_system_i.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"
#include <vulkan/vulkan.h>

// STD
#include <memory>
#include <vector>


class ObjRenderSystem : public IRenderSystem {
public:
	ObjRenderSystem(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout);
	~ObjRenderSystem();

	ObjRenderSystem(const ObjRenderSystem&) = delete;
	ObjRenderSystem& operator=(const ObjRenderSystem&) = delete;

	void init(
		VkSandboxDevice& device,
		VkRenderPass            renderPass,
		VkDescriptorSetLayout   globalSetLayout);

	void render(FrameInfo& frame) override;
private:
	void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
	void createPipeline(VkRenderPass renderPass);

	VkSandboxDevice& m_device;

	VkDescriptorSetLayout m_globalSetLayout;

	std::unique_ptr<VkSandboxPipeline> m_pipeline;
	VkPipelineLayout m_pipelineLayout;
};

