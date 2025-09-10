// obj_rs.cpp
#include "vulkan_wrapper/render_systems/obj_render_system.h"

// External
#define GLM_FORCE_RADIANS	
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// STD
#include <string>
#include <array>
#include <cassert>
#include <stdexcept>

// TODO: Add wireframe pipeline that player input will toggle 


struct PushConstantData {
	glm::mat4 modelMatrix{ 1.f };
	glm::mat4 normalMatrix{ 1.f };
	//int textureIndex;
};

ObjRenderSystem::ObjRenderSystem(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout)
	: m_device(device), m_globalSetLayout(globalSetLayout)
{
	init(device, renderPass, globalSetLayout);
}

void ObjRenderSystem::init(
	VkSandboxDevice& device,
	VkRenderPass renderPass,
	VkDescriptorSetLayout globalSetLayout)
{
	assert(&device == &m_device);

	createPipelineLayout(globalSetLayout);
	createPipeline(renderPass);
}

ObjRenderSystem::~ObjRenderSystem()
{
	vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
}



void ObjRenderSystem::render(FrameInfo& frame)
{
	m_pipeline->bind(frame.commandBuffer);
	std::array<VkDescriptorSet, 1> descriptorSets = {
		frame.globalDescriptorSet
	};

	vkCmdBindDescriptorSets(
		frame.commandBuffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		m_pipelineLayout,
		0,
		static_cast<uint32_t>(descriptorSets.size()),
		descriptorSets.data(),
		0,
		nullptr
	);

	for (auto& kv : frame.gameObjects) {

		auto& obj = kv.second;
		if (obj->getPreferredRenderTag() != RenderTag::Obj) {
			continue; // not mine, skip
		}

		TransformComponent& tc = obj->getTransform();
		PushConstantData push{};
		push.modelMatrix = obj->getTransform().mat4();
		push.normalMatrix = obj->getTransform().normalMatrix();

		vkCmdPushConstants(
			frame.commandBuffer,
			m_pipelineLayout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0,
			sizeof(PushConstantData),
			&push);

		if (auto model = obj->getModel()) {
			model->bind(frame.commandBuffer);
			model->draw(frame.commandBuffer);
		}

	}

}


void ObjRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(PushConstantData);

	std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
		globalSetLayout
	};

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
	pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;



	if (vkCreatePipelineLayout(m_device.device(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create pipeline layout");
	}
}

void ObjRenderSystem::createPipeline(VkRenderPass renderPass)
{
	assert(m_pipelineLayout != nullptr && "Cannot create pipeline before pipeline layout");

	PipelineConfigInfo pipelineConfig{};
	VkSandboxPipeline::defaultPipelineConfigInfo(pipelineConfig);

	pipelineConfig.renderPass = renderPass;
	pipelineConfig.pipelineLayout = m_pipelineLayout;

	std::string vertShaderPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/vert.vert.spv";
	std::string fragShaderPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/frag.frag.spv";

	m_pipeline = std::make_unique<VkSandboxPipeline>(
		m_device,
		vertShaderPath.c_str(),
		fragShaderPath.c_str(),
		pipelineConfig
	);
}