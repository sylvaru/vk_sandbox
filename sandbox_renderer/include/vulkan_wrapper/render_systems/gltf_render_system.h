#pragma once
#include "interfaces/render_system_i.h"
#include "interfaces/asset_provider_i.h"
#include "interfaces/game_object_i.h"
#include "interfaces/model_i.h"

#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"
#include "vulkan_wrapper/vulkan_descriptor.h"

#include "vulkan_wrapper/vulkan_gltf.h"
#include "vulkan_wrapper/core/render_graph.h"
#include "vulkan_wrapper/core/renderable_registry.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>






class GltfRenderSystem : public IRenderSystem {
public:
	GltfRenderSystem(
		VkSandboxDevice& device,
		VkRenderPass renderPass,
		VkDescriptorSetLayout globalSetLayout,
		VkDescriptorSetLayout iblSetLayout,
		const std::vector<VkDescriptorSet>& iblDescriptorSets
	);

	void init(
		VkSandboxDevice& device,
		VkRenderPass            renderPass,
		VkDescriptorSetLayout   globalSetLayout
		);


	~GltfRenderSystem();

	GltfRenderSystem(const GltfRenderSystem&) = delete;
	GltfRenderSystem& operator=(const GltfRenderSystem&) = delete;

	

	void render(FrameInfo& frame) override;
	void record(const RGContext& rgctx, FrameInfo& frame);

private:
	void createPipelineLayout(
		VkDescriptorSetLayout globalSetLayout,
		VkDescriptorSetLayout iblSetLayout
	);
	void createPipeline(VkRenderPass renderPass);

	VkSandboxDevice&							  m_device;

	VkDescriptorSetLayout					      m_globalSetLayout;
	VkDescriptorSetLayout						  m_iblSetLayout;
	std::vector<VkDescriptorSet>				  m_iblDescriptorSets;


	std::unique_ptr<VkSandboxPipeline>		      m_opaquePipeline;
	std::unique_ptr<VkSandboxPipeline>			  m_maskPipeline;
	std::unique_ptr<VkSandboxPipeline>			  m_blendPipeline;
	VkPipelineLayout							  m_pipelineLayout;
};

