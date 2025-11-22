#pragma once
#include "interfaces/render_system_i.h"
#include "interfaces/asset_provider_i.h"
#include "interfaces/game_object_i.h"
#include "interfaces/model_i.h"

#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"
#include "vulkan_wrapper/vulkan_descriptor.h"
#include "vulkan_wrapper/vulkan_gltf.h"

#include "vulkan_wrapper/core/renderable_registry.h"
#include "vulkan_wrapper/core/render_graph.h"


#include <memory>
#include <vector>
#include <vulkan/vulkan.h>


class SceneRenderSystem : public IRenderSystem {
public:
	SceneRenderSystem(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout, IAssetProvider& assets);
	~SceneRenderSystem();

	SceneRenderSystem(const SceneRenderSystem&) = delete;
	SceneRenderSystem& operator=(const SceneRenderSystem&) = delete;

	void init(
		VkSandboxDevice& device,
		VkRenderPass            renderPass,
		VkDescriptorSetLayout   globalSetLayout);

	void render(FrameInfo& frame) override;
	void record(const RGContext& rgctx, FrameInfo& frame);

private:
	void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
	void createPipeline(VkRenderPass renderPass);

	VkSandboxDevice& m_device;

	VkDescriptorSetLayout m_globalSetLayout;

	std::unique_ptr<VkSandboxPipeline> m_opaquePipeline;
	std::unique_ptr<VkSandboxPipeline> m_maskPipeline;
	std::unique_ptr<VkSandboxPipeline> m_blendPipeline;
	VkPipelineLayout m_pipelineLayout;

	IAssetProvider& m_assets;
};

