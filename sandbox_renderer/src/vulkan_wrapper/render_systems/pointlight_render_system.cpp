#include "common/renderer_pch.h"
#include "vulkan_wrapper/render_systems/pointlight_render_system.h"
#include "global_common/frame_info.h"
#include "interfaces/game_object_i.h"



// Define light constants
struct PointLightPushConstants {
    glm::vec4 position{};
    glm::vec4 color{};
    float radius;
};

PointLightRenderSystem::PointLightRenderSystem(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout)
    : m_device(device), m_globalSetLayout(globalSetLayout)
{
    init(device, renderPass, globalSetLayout);
}

PointLightRenderSystem::~PointLightRenderSystem() {
    vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
}

void PointLightRenderSystem::init(
    VkSandboxDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout)
{
    assert(&device == &m_device);

    createPipelineLayout(globalSetLayout);
    createPipeline(renderPass);
}



void PointLightRenderSystem::render(FrameInfo& frame) {
    if (!frame.renderRegistry)
        return;

    auto& cmd = frame.commandBuffer;
    const auto& instances = frame.renderRegistry->getInstancesByType(RenderableType::Light);

    // Sort by distance from camera (optional)
    std::map<float, const MeshInstance*> sorted;
    const glm::vec3 camPos = frame.camera->getPosition();
    for (const MeshInstance* inst : instances) {
        glm::vec3 pos = glm::vec3(inst->transform.model[3]);
        float dist2 = glm::dot(camPos - pos, camPos - pos);
        sorted[dist2] = inst;
    }

    m_pipeline->bind(cmd);

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout,
        0, 1,
        &frame.globalDescriptorSet,
        0, nullptr
    );

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        const MeshInstance* inst = it->second;
        PointLightPushConstants push{};

        glm::vec3 position = glm::vec3(inst->transform.model[3]);
        push.position = glm::vec4(position, 1.0f);
        push.color = glm::vec4(inst->emissiveColor, inst->intensity);
        push.radius = inst->boundingSphereRadius;

        vkCmdPushConstants(
            cmd,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(PointLightPushConstants),
            &push
        );

        vkCmdDraw(cmd, 6, 1, 0, 0);
    }
}


void PointLightRenderSystem::record(const RGContext& rgctx, FrameInfo& frame) {
    frame.commandBuffer = rgctx.cmd;
    frame.frameIndex = rgctx.frameIndex;
    frame.globalDescriptorSet = rgctx.globalSet;

    this->render(frame);
}


void PointLightRenderSystem::update(FrameInfo& frame, GlobalUbo& ubo) {
    auto rotateLight = glm::rotate(glm::mat4(1.f), m_rotationSpeed * frame.frameTime, glm::vec3(0.f, -1.f, 0.f));

    int lightIndex = 0;

    auto& registry = *frame.renderRegistry;
    auto& lights = registry.getInstancesByType(RenderableType::Light);

    for (MeshInstance* inst : lights) {
        if (!inst || lightIndex >= MAX_LIGHTS)
            break;

        // Extract original position from model matrix 
        glm::vec3 pos = glm::vec3(inst->transform.model[3]);

        // Rotate position 
        glm::vec3 newPos = glm::vec3(rotateLight * glm::vec4(pos, 1.f));

        // Write back to transform matrices 
        inst->transform.model[3] = glm::vec4(newPos, 1.0f);
        inst->transform.normalMat = glm::transpose(glm::inverse(inst->transform.model));

        //  Update UBO 
        ubo.pointLights[lightIndex].position = glm::vec4(newPos, 1.f);
        ubo.pointLights[lightIndex].color = glm::vec4(inst->emissiveColor, inst->intensity);

        ++lightIndex;
    }

    ubo.numLights = lightIndex;
}






void PointLightRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PointLightPushConstants);

    std::vector<VkDescriptorSetLayout> setLayouts = { globalSetLayout };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(m_device.device(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create PointLightRS pipeline layout!");
    }
}

void PointLightRenderSystem::createPipeline(VkRenderPass renderPass) {
    assert(m_pipelineLayout != VK_NULL_HANDLE && "Cannot create pipeline before pipeline layout");

    PipelineConfigInfo pipelineConfig{};
    VkSandboxPipeline::defaultPipelineConfigInfo(pipelineConfig);
    pipelineConfig.bindingDescriptions.clear();
    pipelineConfig.attributeDescriptions.clear();
    pipelineConfig.renderPass = renderPass;
    pipelineConfig.pipelineLayout = m_pipelineLayout;

    std::string vertShaderPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/point_light.vert.spv";
    std::string fragShaderPath = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/point_light.frag.spv";

    m_pipeline = std::make_unique<VkSandboxPipeline>(
        m_device,
        vertShaderPath.c_str(),
        fragShaderPath.c_str(),
        pipelineConfig
    );
}

