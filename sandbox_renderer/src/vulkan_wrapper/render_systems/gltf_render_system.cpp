#include "common/renderer_pch.h"
#include"vulkan_wrapper/render_systems/gltf_render_system.h"




GltfRenderSystem::GltfRenderSystem(
    VkSandboxDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout,
    VkDescriptorSetLayout iblSetLayout,
    const std::vector<VkDescriptorSet>& iblDescriptorSets
)
    : m_device(device)
    , m_globalSetLayout(globalSetLayout)
    , m_iblSetLayout(iblSetLayout)
    , m_iblDescriptorSets(iblDescriptorSets)
{
    createPipelineLayout(globalSetLayout, iblSetLayout);
    createPipeline(renderPass);
}


GltfRenderSystem::~GltfRenderSystem() {
    vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
}

void GltfRenderSystem::render(FrameInfo& frame) {
    if (!frame.renderRegistry)
        return;

    VkCommandBuffer cmd = frame.commandBuffer;
    const auto& instances = frame.renderRegistry->getInstancesByType(RenderableType::Gltf);

    for (const MeshInstance* inst : instances) {
        vkglTF::Model* model = inst->model;
        if (!model) continue;

        model->bind(cmd);

        const glm::mat4 instModel = inst->transform.model; // instance/world transform
        // iterate nodes and write *per-node* world = instModel * node->getMatrix()
        for (auto* node : model->m_linearNodes) {
            if (!node->mesh) continue;

            // Compose final world matrix: instance/world transform * node local (hierarchical) matrix
            glm::mat4 nodeWorld = instModel * node->getMatrix();

            glm::mat4 nodeNormalMat = glm::transpose(glm::inverse(nodeWorld));

            // Copy into this node's uniform buffer
            memcpy(node->mesh->uniformBuffer.mapped, &nodeWorld, sizeof(nodeWorld));
            memcpy(reinterpret_cast<char*>(node->mesh->uniformBuffer.mapped) + sizeof(nodeWorld),
                &nodeNormalMat,
                sizeof(nodeNormalMat));


            for (auto* primitive : node->mesh->primitives) {
                std::array<VkDescriptorSet, 2> sets = {
                    frame.globalDescriptorSet,
                    node->mesh->uniformBuffer.descriptorSet
                };
                vkCmdBindDescriptorSets(
                    cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    static_cast<uint32_t>(sets.size()),
                    sets.data(),
                    0, nullptr
                );

                VkDescriptorSet iblSet = m_iblDescriptorSets[frame.frameIndex];
                if (iblSet != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(
                        cmd,
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        m_pipelineLayout,
                        3, 1, &iblSet,
                        0, nullptr
                    );
                }

                switch (primitive->material.alphaMode) {
                case vkglTF::Material::ALPHAMODE_OPAQUE: m_opaquePipeline->bind(cmd); break;
                case vkglTF::Material::ALPHAMODE_MASK:   m_maskPipeline->bind(cmd);   break;
                case vkglTF::Material::ALPHAMODE_BLEND:
                default:                                  m_blendPipeline->bind(cmd);  break;
                }

                model->drawNode(node, frame.commandBuffer, vkglTF::RenderFlags::BindImages, m_pipelineLayout, 2);
            }
        }
    }
}


void GltfRenderSystem::record(const RGContext& rgctx, FrameInfo& frame) {
    frame.commandBuffer = rgctx.cmd;
    frame.frameIndex = rgctx.frameIndex;
    frame.globalDescriptorSet = rgctx.globalSet;

    this->render(frame);
}


void GltfRenderSystem::createPipelineLayout(
    VkDescriptorSetLayout globalSetLayout,
    VkDescriptorSetLayout iblSetLayout
) {
    const std::vector<VkDescriptorSetLayout> layouts = {
        globalSetLayout,                 // set = 0
        vkglTF::descriptorSetLayoutUbo,  // set = 1
        vkglTF::descriptorSetLayoutImage,// set = 2
        iblSetLayout                     // set = 3 
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = (uint32_t)layouts.size();
    pipelineLayoutInfo.pSetLayouts = layouts.data();

    if (vkCreatePipelineLayout(m_device.device(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create GLTF pipeline layout");
    }
}


void GltfRenderSystem::createPipeline(VkRenderPass renderPass) {
    assert(m_pipelineLayout != VK_NULL_HANDLE);

    auto vertSpv = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/gltf_vert.vert.spv";
    auto fragSpv = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/gltf_frag.frag.spv";

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

    struct SpecData { 
        VkBool32 alphaMask; 
        float cutoff;
        int flipMaterialUV;
        int flipEnvMapY;
    };
    static SpecData specData{ VK_TRUE, 0.5f, 1 };
    static VkSpecializationMapEntry mapEntries[4] = {
        { 0, offsetof(SpecData, alphaMask), sizeof(VkBool32) },
        { 1, offsetof(SpecData, cutoff),    sizeof(float) },
        { 7, offsetof(SpecData, flipMaterialUV), sizeof(int)},
        { 8, offsetof(SpecData, flipEnvMapY),  sizeof(int) }
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

