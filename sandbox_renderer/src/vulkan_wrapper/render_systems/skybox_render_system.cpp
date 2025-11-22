#include "vulkan_wrapper/render_systems/skybox_render_system.h"
#include "interfaces/game_object_i.h"


SkyboxRenderSystem::SkyboxRenderSystem(
	VkSandboxDevice& device,
	VkRenderPass renderPass,
	VkDescriptorSetLayout globalSetLayout,
	VkSandboxDescriptorPool& pool)
	: m_device{ device }, m_pipelineLayout{ VK_NULL_HANDLE }, m_skyboxDescriptorSet(VK_NULL_HANDLE), m_bHasCubemap(false)
{
}

SkyboxRenderSystem::~SkyboxRenderSystem() {
	// destroy the pipeline layout you created
	vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
	// (the VkSandboxPipeline unique_ptr will destroy the VkPipeline)
}
void SkyboxRenderSystem::init(
	VkSandboxDevice& device,
	VkRenderPass renderPass,
	VkDescriptorSetLayout globalSetLayout,
	VkSandboxDescriptorPool& descriptorPool)
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
void SkyboxRenderSystem::render(FrameInfo& frame) {
	if (!m_bHasCubemap || !frame.renderRegistry) return;

	const auto& registry = *frame.renderRegistry;
	const auto& skyboxes = registry.getInstancesByType(RenderableType::Skybox);
	if (skyboxes.empty()) return;

	m_pipeline->bind(frame.commandBuffer);

	std::array<VkDescriptorSet, 2> sets = {
		frame.globalDescriptorSet,
		m_skyboxDescriptorSet
	};
	vkCmdBindDescriptorSets(
		frame.commandBuffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		m_pipelineLayout,
		0,
		static_cast<uint32_t>(sets.size()),
		sets.data(),
		0, nullptr
	);

	for (auto* inst : skyboxes) {
		if (!inst->model) continue;

		inst->model->bind(frame.commandBuffer);
		inst->model->gltfDraw(frame.commandBuffer);
	}
}


void SkyboxRenderSystem::record(const RGContext& rgctx, FrameInfo& frame) {
	frame.commandBuffer = rgctx.cmd;
	frame.frameIndex = rgctx.frameIndex;
	frame.globalDescriptorSet = rgctx.globalSet;

	this->render(frame);
}

void SkyboxRenderSystem::createSkyboxDescriptorSetLayout() {
	m_skyboxSetLayout = VkSandboxDescriptorSetLayout::Builder(m_device)
		.addBinding(
			0,
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			VK_SHADER_STAGE_FRAGMENT_BIT,
			1,
			0)
		.build();
}

void SkyboxRenderSystem::allocateAndWriteSkyboxDescriptorSet() {
	assert(m_descriptorPool && "Descriptor pool must be set before allocating descriptors");
	assert(m_skyboxSetLayout && "Descriptor set layout must be created before allocating");

	VkSandboxDescriptorWriter writer(*m_skyboxSetLayout, *m_descriptorPool);
	writer.writeImage(0, &m_skyboxImageInfo);
	bool success = writer.build(m_skyboxDescriptorSet);
	assert(success && "Failed to build skybox descriptor set");
}

void SkyboxRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {

	VkDescriptorSetLayout skyboxLayoutHandle =
		m_skyboxSetLayout->getDescriptorSetLayout();

	std::array<VkDescriptorSetLayout, 2> layouts = {
		globalSetLayout,
		skyboxLayoutHandle
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



void SkyboxRenderSystem::createPipeline(VkRenderPass renderPass) {
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
	config.depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;


	std::string vertPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/skybox_ibl.vert.spv";
	std::string fragPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/skybox_ibl.frag.spv";

	m_pipeline = std::make_unique<VkSandboxPipeline>(
		m_device,
		vertPath.c_str(),
		fragPath.c_str(),
		config
	);
}
