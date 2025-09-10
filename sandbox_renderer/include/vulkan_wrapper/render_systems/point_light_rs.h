#pragma once
#include "interfaces/render_system_i.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"
#include <vulkan/vulkan.h>

// STD
#include <memory>
#include <vector>

struct PointLightComponent;

class PointLightRS : public IRenderSystem {
public:

	PointLightRS(const PointLightRS&) = delete;
	PointLightRS& operator=(const PointLightRS&) = delete;

	PointLightRS(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout);
	~PointLightRS();

	void init(
		VkSandboxDevice& device,
		VkRenderPass            renderPass,
		VkDescriptorSetLayout   globalSetLayout);

	void update(FrameInfo& frame, GlobalUbo& ubo) override;
	void render(FrameInfo& frame) override;
private:
	void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
	void createPipeline(VkRenderPass renderPass);

	VkSandboxDevice& m_device;

	VkDescriptorSetLayout m_globalSetLayout;

	std::unique_ptr<VkSandboxPipeline> m_pipeline;
	VkPipelineLayout m_pipelineLayout;

	float m_rotationSpeed = 0.2f;
};

