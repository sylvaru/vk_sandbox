#include"vulkan_wrapper/render_systems/gltf_render_system.h"
#include <spdlog/spdlog.h>



GltfRenderSystem::GltfRenderSystem(
    VkSandboxDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout,
    VkSandboxDescriptorPool& descriptorPool,
    IAssetProvider& assets,
    size_t frameCount
) :
    m_device(device),
    m_globalSetLayout(globalSetLayout),
    m_assets(assets)
{
    init(device,renderPass,globalSetLayout, descriptorPool, frameCount);
}

GltfRenderSystem::~GltfRenderSystem() {
    vkDestroyPipelineLayout(m_device.device(), m_pipelineLayout, nullptr);
}


void GltfRenderSystem::init(
    VkSandboxDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout,
    VkSandboxDescriptorPool& descriptorPool,
    size_t frameCount
) {
    m_globalSetLayout = globalSetLayout;

    m_iblLayout = VkSandboxDescriptorSetLayout::Builder{ device }
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .build();

    createPipelineLayout(globalSetLayout);
    createPipeline(renderPass);

    m_iblDescriptorSets.resize(frameCount);
    for (uint32_t i = 0; i < frameCount; i++) {
        VkDescriptorSet set;
        descriptorPool.allocateDescriptor(
            m_iblLayout->getDescriptorSetLayout(),
            set,
            /*setIndex=*/0
        );

        auto brdfInfo = m_assets.getBRDFLUTDescriptor();
        auto irradianceInfo = m_assets.getIrradianceDescriptor();
        auto prefilterInfo = m_assets.getPrefilteredDescriptor();

        VkSandboxDescriptorWriter(*m_iblLayout, descriptorPool)
            .writeImage(0, &brdfInfo)
            .writeImage(1, &irradianceInfo)
            .writeImage(2, &prefilterInfo)
            .build(set);

        m_iblDescriptorSets[i] = set;
    }
}

void GltfRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
    const std::vector<VkDescriptorSetLayout> layouts = {
       globalSetLayout,                         // set 0: global UBO
       vkglTF::descriptorSetLayoutUbo,          // set 1: per-node UBO
       vkglTF::descriptorSetLayoutImage,        // set 2: material image samplers (albedo/normal/metal/rough/ao)
       m_iblLayout->getDescriptorSetLayout()    // set 3: IBL (brdf/irradiance/prefilter)
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
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

void GltfRenderSystem::render(FrameInfo& frame) {
    static bool warnedThisFrame = false;

    for (auto& [id, go] : frame.gameObjects) {

        if (go->getPreferredRenderTag() != RenderTag::Gltf) {
            continue; // not mine, skip
        }
        auto baseModel = go->getModel();
        if (!baseModel) continue;

        auto model = std::dynamic_pointer_cast<vkglTF::Model>(baseModel);
        if (!model) continue;
      
        model->bind(frame.commandBuffer);

        for (auto* node : model->m_linearNodes) {
            if (!node->mesh) continue;

            glm::mat4 world = go->getTransform().mat4() * node->getMatrix();
            glm::mat4 normalMat = glm::transpose(glm::inverse(world));

            memcpy(node->mesh->uniformBuffer.mapped, &world, sizeof(world));
            memcpy((char*)node->mesh->uniformBuffer.mapped + sizeof(world), &normalMat, sizeof(normalMat));

            for (auto* primitive : node->mesh->primitives) {
                // --- Bind sets 0 & 1 (global + node UBO) ---
                std::array<VkDescriptorSet, 2> sets = {
                    frame.globalDescriptorSet,                 // set 0
                    node->mesh->uniformBuffer.descriptorSet    // set 1
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

                // --- Bind IBL set ---
                VkDescriptorSet iblSet = m_iblDescriptorSets[frame.frameIndex];
                if (iblSet == VK_NULL_HANDLE) { continue; }
                vkCmdBindDescriptorSets(
                    frame.commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    3, 1, &iblSet,
                    0, nullptr
                );

                // Pick pipeline by alpha mode
                switch (primitive->material.alphaMode) {
                case vkglTF::Material::ALPHAMODE_OPAQUE: m_opaquePipeline->bind(frame.commandBuffer); break;
                case vkglTF::Material::ALPHAMODE_MASK:   m_maskPipeline->bind(frame.commandBuffer);   break;
                case vkglTF::Material::ALPHAMODE_BLEND:
                default:                                  m_blendPipeline->bind(frame.commandBuffer);  break;
                }

                model->gltfDraw(frame.commandBuffer, vkglTF::RenderFlags::BindImages, m_pipelineLayout, 2);
                warnedThisFrame = false;
            }
        }
    }
}
