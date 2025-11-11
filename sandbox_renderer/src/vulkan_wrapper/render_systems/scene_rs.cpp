#include"vulkan_wrapper/render_systems/scene_rs.h"
#include <spdlog/spdlog.h>


SceneRenderSystem::SceneRenderSystem(
    VkSandboxDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout,
    IAssetProvider& assets
) :
    m_device(device),
    m_globalSetLayout(globalSetLayout),
    m_assets(assets)
{
    init(device, renderPass, globalSetLayout);
}

SceneRenderSystem::~SceneRenderSystem() {
    vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
}


void SceneRenderSystem::init(
    VkSandboxDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout
) {
    m_globalSetLayout = globalSetLayout;


    createPipelineLayout(globalSetLayout);
    createPipeline(renderPass);

}

void SceneRenderSystem::render(FrameInfo& frame) {
    const RenderableRegistry* registry = frame.renderRegistry;
    if (!registry) return;

    // Get all scene-type instances
    const auto& instances = registry->getInstancePool();
    std::vector<const MeshInstance*> sceneInstances;
    sceneInstances.reserve(instances.size());

    for (const auto& inst : instances) {
        if (inst.type == RenderableType::Gltf || inst.type == RenderableType::Scene) {
            if (inst.model) sceneInstances.push_back(&inst);
        }
    }

    if (sceneInstances.empty()) return;

    // Record commands for each scene instance
    for (const MeshInstance* inst : sceneInstances) {
        vkglTF::Model* model = inst->model;
        if (!model) continue;

        model->bind(frame.commandBuffer);

        // Update transforms (could later be GPU-buffer-driven)
        const glm::mat4& world = inst->transform.model;
        const glm::mat4& normalMat = inst->transform.normalMat;

        for (auto* node : model->m_linearNodes) {
            if (!node->mesh) continue;

            // Write updated per-object transforms
            glm::mat4 nodeWorld = world * node->getMatrix();
            glm::mat4 nodeNormalMat = glm::transpose(glm::inverse(nodeWorld));

            // Write updated per-object transforms (per-node)
            memcpy(node->mesh->uniformBuffer.mapped, &nodeWorld, sizeof(nodeWorld));
            memcpy(
                reinterpret_cast<char*>(node->mesh->uniformBuffer.mapped) + sizeof(nodeWorld),
                &nodeNormalMat,
                sizeof(nodeNormalMat)
            );

            for (auto* primitive : node->mesh->primitives) {
                // Bind descriptor sets
                std::array<VkDescriptorSet, 2> sets = {
                    frame.globalDescriptorSet,
                    node->mesh->uniformBuffer.descriptorSet
                };

                vkCmdBindDescriptorSets(
                    frame.commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    static_cast<uint32_t>(sets.size()),
                    sets.data(),
                    0,
                    nullptr
                );

                // Pick pipeline by alpha mode
                switch (primitive->material.alphaMode) {
                case vkglTF::Material::ALPHAMODE_OPAQUE:
                    m_opaquePipeline->bind(frame.commandBuffer);
                    break;
                case vkglTF::Material::ALPHAMODE_MASK:
                    m_maskPipeline->bind(frame.commandBuffer);
                    break;
                case vkglTF::Material::ALPHAMODE_BLEND:
                default:
                    m_blendPipeline->bind(frame.commandBuffer);
                    break;
                }

                // Draw node
                model->drawNode(node, frame.commandBuffer, vkglTF::RenderFlags::BindImages, m_pipelineLayout, 2);
            }
        }
    }
}

void SceneRenderSystem::record(const RGContext& rgctx, FrameInfo& frame) {
    frame.commandBuffer = rgctx.cmd;
    frame.frameIndex = rgctx.frameIndex;
    frame.globalDescriptorSet = rgctx.globalSet;

    this->render(frame);
}


void SceneRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
    const std::vector<VkDescriptorSetLayout> layouts = {
        globalSetLayout,
        vkglTF::descriptorSetLayoutUbo,
        vkglTF::descriptorSetLayoutImage
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
    pipelineLayoutInfo.pSetLayouts = layouts.data();

    if (vkCreatePipelineLayout(m_device.device(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create GLTF pipeline layout");
    }
}

void SceneRenderSystem::createPipeline(VkRenderPass renderPass) {
    assert(m_pipelineLayout != VK_NULL_HANDLE);

    auto vertSpv = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/scene_vert.vert.spv";
    auto fragSpv = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/scene_frag.frag.spv";

    std::vector<VkVertexInputBindingDescription> bindings = {
        vkinit::vertexInputBindingDescription(0, sizeof(vkglTF::Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
    };

    std::vector<VkVertexInputAttributeDescription> attributes = {
        vkinit::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, pos)),
        vkinit::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, normal)),
        vkinit::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(vkglTF::Vertex, uv)),
        vkinit::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vkglTF::Vertex, color)),
        vkinit::vertexInputAttributeDescription(0, 4, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(vkglTF::Vertex, tangent))
    };

    // OPAQUE
    PipelineConfigInfo opaqueConfig{};
    VkSandboxPipeline::defaultPipelineConfigInfo(opaqueConfig);
    opaqueConfig.pipelineLayout = m_pipelineLayout;
    opaqueConfig.renderPass = renderPass;
    opaqueConfig.bindingDescriptions = bindings;
    opaqueConfig.attributeDescriptions = attributes;

    m_opaquePipeline = std::make_unique<VkSandboxPipeline>(
        m_device, vertSpv, fragSpv, opaqueConfig);

    // MASK
    PipelineConfigInfo maskConfig{};
    VkSandboxPipeline::defaultPipelineConfigInfo(maskConfig);
    maskConfig.pipelineLayout = m_pipelineLayout;
    maskConfig.renderPass = renderPass;
    maskConfig.bindingDescriptions = bindings;
    maskConfig.attributeDescriptions = attributes;
    maskConfig.colorBlendAttachment.blendEnable = VK_FALSE;

    struct SpecData { VkBool32 alphaMask; float cutoff; };
    static SpecData specData{ VK_TRUE, 0.5f };
    static VkSpecializationMapEntry mapEntries[2] = {
        { 0, offsetof(SpecData, alphaMask), sizeof(VkBool32) },
        { 1, offsetof(SpecData, cutoff),    sizeof(float) }
    };
    static VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 2;
    specInfo.pMapEntries = mapEntries;
    specInfo.dataSize = sizeof(specData);
    specInfo.pData = &specData;

    maskConfig.fragSpecInfo = &specInfo;

    m_maskPipeline = std::make_unique<VkSandboxPipeline>(
        m_device, vertSpv, fragSpv, maskConfig);

    // BLEND
    PipelineConfigInfo blendConfig{};
    VkSandboxPipeline::defaultPipelineConfigInfo(blendConfig);
    blendConfig.pipelineLayout = m_pipelineLayout;
    blendConfig.renderPass = renderPass;
    blendConfig.bindingDescriptions = bindings;
    blendConfig.attributeDescriptions = attributes;

    blendConfig.colorBlendAttachment.blendEnable = VK_TRUE;
    blendConfig.colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendConfig.colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendConfig.colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendConfig.colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendConfig.colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendConfig.colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    blendConfig.colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    m_blendPipeline = std::make_unique<VkSandboxPipeline>(
        m_device, vertSpv, fragSpv, blendConfig);
}
