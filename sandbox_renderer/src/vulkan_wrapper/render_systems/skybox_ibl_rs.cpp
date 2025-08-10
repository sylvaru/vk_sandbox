#include "vulkan_wrapper/render_systems/skybox_ibl_rs.h"
#include "interfaces/game_object_i.h"


SkyboxIBLrenderSystem::SkyboxIBLrenderSystem(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout)
	: m_device{ device }, m_pipelineLayout{ VK_NULL_HANDLE }, m_skyboxDescriptorSet(VK_NULL_HANDLE), m_bHasCubemap(false)
{
}



SkyboxIBLrenderSystem::~SkyboxIBLrenderSystem() {
	// destroy the pipeline layout you created
	vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
	// (the VkSandboxPipeline unique_ptr will destroy the VkPipeline)
}
void SkyboxIBLrenderSystem::init(
	VkSandboxDevice& device,
	VkRenderPass renderPass,
	VkDescriptorSetLayout globalSetLayout,
	VkSandboxDescriptorPool& descriptorPool,
	size_t frameCount)
{
	assert(&device == &m_device);
	m_descriptorPool = &descriptorPool;

	createSkyboxDescriptorSetLayout();

	if (m_bHasCubemap) {
		allocateAndWriteSkyboxDescriptorSet();
	}

	createPipelineLayout(globalSetLayout);
	createPipeline(renderPass);
}

void SkyboxIBLrenderSystem::createSkyboxDescriptorSetLayout() {
	m_skyboxSetLayout = VkSandboxDescriptorSetLayout::Builder(m_device)
		.addBinding(
			0,
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			VK_SHADER_STAGE_FRAGMENT_BIT,
			1,
			0) // no binding flags needed here
		.build();
}


void SkyboxIBLrenderSystem::allocateAndWriteSkyboxDescriptorSet() {
	assert(m_descriptorPool && "Descriptor pool must be set before allocating descriptors");
	assert(m_skyboxSetLayout && "Descriptor set layout must be created before allocating");

	VkSandboxDescriptorWriter writer(*m_skyboxSetLayout, *m_descriptorPool);
	writer.writeImage(0, &m_skyboxImageInfo);
	bool success = writer.build(m_skyboxDescriptorSet);
	assert(success && "Failed to build skybox descriptor set");
}

void SkyboxIBLrenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {

	VkDescriptorSetLayout skyboxLayoutHandle =
		m_skyboxSetLayout->getDescriptorSetLayout();

	std::array<VkDescriptorSetLayout, 2> layouts = {
		globalSetLayout,
		skyboxLayoutHandle // from createSkyboxDescriptorSetLayout()
	};

	VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layoutInfo.setLayoutCount = (uint32_t)layouts.size();
	layoutInfo.pSetLayouts = layouts.data();
	layoutInfo.pushConstantRangeCount = 0;
	layoutInfo.pPushConstantRanges = nullptr;

	if (vkCreatePipelineLayout(m_device.device(), &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create skybox pipeline layout");
	}
}

void SkyboxIBLrenderSystem::render(FrameInfo& frameInfo) {

	if (!m_bHasCubemap) return;

	//if (!m_skyboxModel) return;
	auto skyOpt = frameInfo.scene.getSkyboxObject();
	if (!skyOpt.has_value()) {
		return; // nothing to draw
	}
	IGameObject& skyObj = skyOpt->get();

	assert(m_skyboxDescriptorSet != VK_NULL_HANDLE && "Skybox descriptor set is not allocated!");

	m_pipeline->bind(frameInfo.commandBuffer);
	// Bind two descriptor sets: 0=global UBO, 1=skybox cubemap
	std::array<VkDescriptorSet, 2> sets = {
		frameInfo.globalDescriptorSet,
		m_skyboxDescriptorSet
	};
	vkCmdBindDescriptorSets(
		frameInfo.commandBuffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		m_pipelineLayout,
		0, // firstSet = 0 (binds set 0 and 1 in this call)
		static_cast<uint32_t>(sets.size()),
		sets.data(),
		0,
		nullptr
	);



	auto model = skyObj.getModel();
	if (model) {
		model->bind(frameInfo.commandBuffer);
		model->gltfDraw(frameInfo.commandBuffer);
	}
}


void SkyboxIBLrenderSystem::createPipeline(VkRenderPass renderPass) {
	assert(m_pipelineLayout != VK_NULL_HANDLE && "Pipeline layout must be created before pipeline");

	PipelineConfigInfo config{};
	VkSandboxPipeline::defaultPipelineConfigInfo(config);

	std::vector<VkVertexInputBindingDescription>   bindings = {
		vkinit::vertexInputBindingDescription(
			0,
			sizeof(vkglTF::Vertex),
			VK_VERTEX_INPUT_RATE_VERTEX)
	};
	std::vector<VkVertexInputAttributeDescription> attributes = {
		vkinit::vertexInputAttributeDescription(
			/*binding=*/0,
			/*location=*/0,
			/*format=*/VK_FORMAT_R32G32B32_SFLOAT,
			/*offset=*/offsetof(vkglTF::Vertex, pos))
	};

	config.bindingDescriptions = bindings;
	config.attributeDescriptions = attributes;
	config.renderPass = renderPass;
	config.pipelineLayout = m_pipelineLayout;
	config.depthStencilInfo.depthTestEnable = VK_TRUE;
	config.depthStencilInfo.depthWriteEnable = VK_FALSE;
	config.depthStencilInfo.depthWriteEnable = VK_COMPARE_OP_LESS_OR_EQUAL;


	std::string vertPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/skybox_ibl.vert.spv";
	std::string fragPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/skybox_ibl.frag.spv";

	m_pipeline = std::make_unique<VkSandboxPipeline>(
		m_device,
		vertPath.c_str(),
		fragPath.c_str(),
		config
	);
}
