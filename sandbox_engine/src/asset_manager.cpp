// asset_manager.cpp
#include "asset_manager.h"

#include <json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>
#include <glm/glm.hpp>

using json = nlohmann::json;

AssetManager::AssetManager(VkSandboxDevice& device) : m_device(device), m_transferQueue(m_device.graphicsQueue()) {

}

AssetManager::~AssetManager()
{

}
// AssetManager.cpp

void AssetManager::preloadGlobalAssets() {
    // Read model & cubemap list JSON
    std::ifstream in(PROJECT_ROOT_DIR "/sandbox_game/res/scene_assets/default_scene_assets.json");
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open model list JSON.");
    }

    json modelJson;
    in >> modelJson;

    // 1) Load models first (unchanged from your flow)
    for (const auto& entry : modelJson["models"]) {
        const std::string name = entry["name"];
        const std::string type = entry.value("type", "obj");
        const std::string path = std::string(PROJECT_ROOT_DIR) + "/res/models/" + entry["path"].get<std::string>();

        try {
            if (type == "obj") {
                auto model = loadObjModel(name, path, false);
                spdlog::info("[AssetManager] Successfully loaded OBJ model '{}' from '{}'", name, path);
            }
            else if (type == "gltf") {
                uint32_t flags = entry.value("flags", 0);  // Optional flags
                float scale = entry.value("scale", 1.0f);  // Optional scale
                auto model = loadGLTFmodel(name, path, flags, scale);
                if (entry.value("usage", "") == "skybox" || name == "cube") {
                    m_skyboxModel = model;
                }
                spdlog::info("[AssetManager] Successfully loaded glTF model '{}' from '{}'", name, path);
            }
            else {
                spdlog::warn("[AssetManager] Unknown model type '{}' for asset '{}'", type, name);
            }
        }
        catch (const std::exception& e) {
            spdlog::error("[AssetManager] Failed to load model '{}': {}", name, e.what());
        }
    }

    // 2) Load cubemaps *before* generating BRDF / irradiance / prefiltered maps
    // Keep track of whether we loaded the environment cubemap used for IBL
    std::shared_ptr<VkSandboxTexture> loadedEnvironmentCubemap = nullptr;

    for (const auto& entry : modelJson["cubemaps"]) {
        const std::string name = entry["name"];
        const std::string path = std::string(PROJECT_ROOT_DIR) + "/res/textures/" + entry["path"].get<std::string>();

        // Map format string (if present) to VkFormat; default to R32G32B32A32_SFLOAT
        std::string fmtStr = entry.value("format", "VK_FORMAT_R32G32B32A32_SFLOAT");
        VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (fmtStr == "VK_FORMAT_R16G16B16A16_SFLOAT") format = VK_FORMAT_R16G16B16A16_SFLOAT;
        else if (fmtStr == "VK_FORMAT_R32G32B32A32_SFLOAT") format = VK_FORMAT_R32G32B32A32_SFLOAT;
        // Add more mappings here if you expect other formats

        try {
            // loadCubemap() must return std::shared_ptr<VkSandboxTexture>
            auto cubemap = loadCubemap(
                name,
                path,
                format,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );

            if (!cubemap) {
                spdlog::error("[AssetManager] loadCubemap returned nullptr for '{}'", name);
                continue;
            }

            // Register into your caches (optional helper)
            registerTextureIfNeeded(name, cubemap, m_textures, m_textureIndexMap, m_textureList);

            spdlog::info("[AssetManager] Successfully loaded cubemap '{}' from '{}'", name, path);

            // If this cubemap is the environment (skybox_hdr per your JSON), remember it.
            // Use the name you expect in your code / JSON. I see "skybox_hdr" in your example JSON.
            if (name == "skybox_hdr" || entry.value("environment", false)) {
                // prefer explicit "environment" : true in JSON or name match
                loadedEnvironmentCubemap = cubemap;
            }
        }
        catch (const std::exception& e) {
            spdlog::error("[AssetManager] Failed to load cubemap '{}': {}", name, e.what());
        }
    }

    // 3) Create/generate BRDF LUT and IBL assets — ensure environmentCube points to the loaded cubemap
    // If your JSON didn't mark which cubemap is the environment, we fallback to looking up "skybox_hdr"
    if (!loadedEnvironmentCubemap) {
        auto it = m_textures.find("skybox_hdr");
        if (it != m_textures.end()) {
            loadedEnvironmentCubemap = it->second;
        }
    }

    if (!loadedEnvironmentCubemap) {
        spdlog::warn("[AssetManager] No environment cubemap found (expected 'skybox_hdr' or 'environment':true). Using placeholder/empty environmentCube.");
        // Optionally: throw or create a debug 1x1 texture so validation doesn't fail.
        // For now we will not create an invalid shared_ptr (keeps previous behavior safer).
    }
    else {
        environmentCube = loadedEnvironmentCubemap;
    }

    // Create BRDF LUT, irradianceCube, prefilteredCube structures (these should allocate their own images)
    // Note: remove any line that reassigns environmentCube to an empty VkSandboxTexture (that was the bug)
    lutBrdf = std::make_shared<VkSandboxTexture>(&m_device);
    irradianceCube = std::make_shared<VkSandboxTexture>(&m_device);
    prefilteredCube = std::make_shared<VkSandboxTexture>(&m_device);

    // Generate BRDF LUT first (your existing function)
    generateBRDFlut();

    // Now generate irradiance and prefiltered maps using environmentCube (must be valid)
    if (!environmentCube) {
        spdlog::error("[AssetManager] environmentCube is null - aborting IBL generation to avoid descriptor errors.");
    }
    else {
        try {
            generateIrradianceMap();
            generatePrefilteredEnvMap();
            spdlog::info("[AssetManager] IBL assets generated successfully.");
        }
        catch (const std::exception& e) {
            spdlog::error("[AssetManager] IBL generation failed: {}", e.what());
        }
    }

    spdlog::info("Assets loaded");
}


void AssetManager::generateIrradianceMap() {
    auto tStart = std::chrono::high_resolution_clock::now();

    const VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    const int32_t dim = 64;
    const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

    // create irradiance cubemap (same as before)
    VkImageCreateInfo imageCI = vkinit::imageCreateInfo();
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = format;
    imageCI.extent.width = dim;
    imageCI.extent.height = dim;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = numMips;
    imageCI.arrayLayers = 6;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    VK_CHECK_RESULT(vkCreateImage(m_device.device(), &imageCI, nullptr, &irradianceCube->m_image));

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(m_device.device(), irradianceCube->m_image, &memReqs);
    VkMemoryAllocateInfo memAlloc = vkinit::memoryAllocateInfo();
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = m_device.getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device.device(), &memAlloc, nullptr, &irradianceCube->m_deviceMemory));
    VK_CHECK_RESULT(vkBindImageMemory(m_device.device(), irradianceCube->m_image, irradianceCube->m_deviceMemory, 0));

    // view & sampler
    VkImageViewCreateInfo viewCI = vkinit::imageViewCreateInfo();
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewCI.format = format;
    viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.baseMipLevel = 0;
    viewCI.subresourceRange.levelCount = numMips;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount = 6;
    viewCI.image = irradianceCube->m_image;
    VK_CHECK_RESULT(vkCreateImageView(m_device.device(), &viewCI, nullptr, &irradianceCube->m_view));

    VkSamplerCreateInfo samplerCI = vkinit::samplerCreateInfo();
    samplerCI.magFilter = VK_FILTER_LINEAR;
    samplerCI.minFilter = VK_FILTER_LINEAR;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = static_cast<float>(numMips);
    samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(m_device.device(), &samplerCI, nullptr, &irradianceCube->m_sampler));

    irradianceCube->m_descriptor.imageView = irradianceCube->m_view;
    irradianceCube->m_descriptor.sampler = irradianceCube->m_sampler;
    irradianceCube->m_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    irradianceCube->m_pDevice = &m_device;

    // --- create offscreen renderpass/framebuffer (unchanged) ---
    VkAttachmentDescription attDesc = {};
    attDesc.format = format;
    attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
    attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

    std::array<VkSubpassDependency, 2> dependencies;
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassCI = vkinit::renderPassCreateInfo();
    renderPassCI.attachmentCount = 1;
    renderPassCI.pAttachments = &attDesc;
    renderPassCI.subpassCount = 1;
    renderPassCI.pSubpasses = &subpassDescription;
    renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassCI.pDependencies = dependencies.data();
    VkRenderPass renderpass;
    VK_CHECK_RESULT(vkCreateRenderPass(m_device.device(), &renderPassCI, nullptr, &renderpass));

    // offscreen color image (1 mip, reused for all mips/faces)
    struct {
        VkImage image;
        VkImageView view;
        VkDeviceMemory memory;
        VkFramebuffer framebuffer;
    } offscreen;

    {
        VkImageCreateInfo imageCreateInfo = vkinit::imageCreateInfo();
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent.width = dim;
        imageCreateInfo.extent.height = dim;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK_RESULT(vkCreateImage(m_device.device(), &imageCreateInfo, nullptr, &offscreen.image));

        vkGetImageMemoryRequirements(m_device.device(), offscreen.image, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = m_device.getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(m_device.device(), &memAlloc, nullptr, &offscreen.memory));
        VK_CHECK_RESULT(vkBindImageMemory(m_device.device(), offscreen.image, offscreen.memory, 0));

        VkImageViewCreateInfo colorImageView = vkinit::imageViewCreateInfo();
        colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorImageView.format = format;
        colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorImageView.subresourceRange.baseMipLevel = 0;
        colorImageView.subresourceRange.levelCount = 1;
        colorImageView.subresourceRange.baseArrayLayer = 0;
        colorImageView.subresourceRange.layerCount = 1;
        colorImageView.image = offscreen.image;
        VK_CHECK_RESULT(vkCreateImageView(m_device.device(), &colorImageView, nullptr, &offscreen.view));

        VkFramebufferCreateInfo fbufCreateInfo = vkinit::framebufferCreateInfo();
        fbufCreateInfo.renderPass = renderpass;
        fbufCreateInfo.attachmentCount = 1;
        fbufCreateInfo.pAttachments = &offscreen.view;
        fbufCreateInfo.width = dim;
        fbufCreateInfo.height = dim;
        fbufCreateInfo.layers = 1;
        VK_CHECK_RESULT(vkCreateFramebuffer(m_device.device(), &fbufCreateInfo, nullptr, &offscreen.framebuffer));

        VkCommandBuffer layoutCmd = m_device.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        tools::setImageLayout(layoutCmd, offscreen.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        m_device.flushCommandBuffer(layoutCmd, m_transferQueue, true);
    }

    // Descriptor layout/pool/set (same as before)
    VkDescriptorSetLayout descriptorsetlayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vkinit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
    };
    VkDescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vkinit::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device.device(), &descriptorsetlayoutCI, nullptr, &descriptorsetlayout));

    std::vector<VkDescriptorPoolSize> poolSizes = { vkinit::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) };
    VkDescriptorPoolCreateInfo descriptorPoolCI = vkinit::descriptorPoolCreateInfo(poolSizes, 2);
    VkDescriptorPool descriptorpool;
    VK_CHECK_RESULT(vkCreateDescriptorPool(m_device.device(), &descriptorPoolCI, nullptr, &descriptorpool));

    VkDescriptorSet descriptorset;
    VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device.device(), &allocInfo, &descriptorset));
    VkWriteDescriptorSet writeDescriptorSet = vkinit::writeDescriptorSet(descriptorset, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &environmentCube->m_descriptor);
    vkUpdateDescriptorSets(m_device.device(), 1, &writeDescriptorSet, 0, nullptr);

    // Push block
    struct PushBlock {
        glm::mat4 mvp;
        float deltaPhi = (2.0f * float(M_PI)) / 180.0f;
        float deltaTheta = (0.5f * float(M_PI)) / 64.0f;
    } pushBlock;

    // Pipeline config — IMPORTANT: provide vertex input descriptions to match shader (location 0)
    PipelineConfigInfo cfg{};
    VkSandboxPipeline::defaultPipelineConfigInfo(cfg);

    // Vertex input: location 0 is a vec3 position (adjust if your skybox vertex layout differs)
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(glm::vec3);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrDesc{};
    attrDesc.binding = 0;
    attrDesc.location = 0;
    attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT; // vec3
    attrDesc.offset = 0;

    cfg.bindingDescriptions = { bindingDesc };
    cfg.attributeDescriptions = { attrDesc };

    cfg.renderPass = renderpass;
    cfg.pipelineLayout = VK_NULL_HANDLE;
    cfg.dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    cfg.dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    cfg.dynamicStateInfo.pDynamicStates = cfg.dynamicStateEnables.data();
    cfg.dynamicStateInfo.dynamicStateCount = uint32_t(cfg.dynamicStateEnables.size());
    cfg.descriptorSetLayouts = { descriptorsetlayout };

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushBlock);
    cfg.pushConstantRanges = { pushRange };

    cfg.pipelineLayout = VK_NULL_HANDLE;

    std::string vert = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/filtered_cube.vert.spv";
    std::string frag = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/irradiance_cube.frag.spv";
    VkSandboxPipeline irradiancePipeline{ m_device, vert, frag, cfg };

    // COMMAND RECORDING
    VkCommandBuffer cmdBuf = m_device.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Transition irradiance cubemap to TRANSFER_DST (outside any renderpass)
    VkImageSubresourceRange cubemapRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, numMips, 0, 6 };
    tools::setImageLayout(cmdBuf, irradianceCube->m_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, cubemapRange);

    // Setup matrices (same as Sascha)
    std::vector<glm::mat4> matrices = {
        glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    };

    // Main loop: mips and faces (matches Sascha's approach)
    for (uint32_t m = 0; m < numMips; ++m) {
        uint32_t mipDim = static_cast<uint32_t>(dim * std::pow(0.5f, (float)m));
        VkViewport vp = vkinit::viewport((float)mipDim, (float)mipDim, 0.0f, 1.0f);
        VkRect2D sc = vkinit::rect2D(mipDim, mipDim, 0, 0);

        for (uint32_t face = 0; face < 6; ++face) {
            // Begin render pass into offscreen framebuffer
            VkClearValue clear{ {{0.0f, 0.0f, 0.2f, 0.0f}} };
            VkRenderPassBeginInfo rpBI = vkinit::renderPassBeginInfo();
            rpBI.renderPass = renderpass;
            rpBI.framebuffer = offscreen.framebuffer;
            rpBI.renderArea.extent = { mipDim, mipDim };
            rpBI.clearValueCount = 1;
            rpBI.pClearValues = &clear;
            vkCmdBeginRenderPass(cmdBuf, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdSetViewport(cmdBuf, 0, 1, &vp);
            vkCmdSetScissor(cmdBuf, 0, 1, &sc);

            // push constants
            pushBlock.mvp = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 512.0f) * matrices[face];
            vkCmdPushConstants(cmdBuf, irradiancePipeline.getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushBlock), &pushBlock);

            // bind pipeline and descriptor set (USE the allocated VkDescriptorSet)
            irradiancePipeline.bind(cmdBuf);
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, irradiancePipeline.getPipelineLayout(), 0, 1, &descriptorset, 0, nullptr);

            // draw skybox — ensure your skybox.draw binds the vertex buffer that matches location 0 vec3 pos
            if (!m_skyboxModel) {
                spdlog::error("[AssetManager] No skybox model loaded - skipping draw in generateIrradianceMap()");
            }
            else {
                // m_skyboxModel is your GLTFmodelHandle (likely a shared_ptr to vkglTF::Model)
                m_skyboxModel->draw(cmdBuf);// this matches Sascha's models.skybox.draw(cmdBuf)
            }
       

            // END render pass BEFORE any barriers/copies
            vkCmdEndRenderPass(cmdBuf);

            // Transition offscreen image -> TRANSFER_SRC and copy to target cubemap mip/face
            tools::setImageLayout(cmdBuf, offscreen.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, face, 1 };
            copyRegion.extent = { mipDim, mipDim, 1 };

            vkCmdCopyImage(cmdBuf,
                offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                irradianceCube->m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &copyRegion);

            // restore offscreen layout for next render
            tools::setImageLayout(cmdBuf, offscreen.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
    }

    // final transition for cubemap to shader read layout
    tools::setImageLayout(cmdBuf, irradianceCube->m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cubemapRange);

    m_device.flushCommandBuffer(cmdBuf, m_transferQueue);
    vkQueueWaitIdle(m_transferQueue);

    // cleanup (destroy created renderpass/framebuffer)
    vkDestroyFramebuffer(m_device.device(), offscreen.framebuffer, nullptr);
    vkDestroyRenderPass(m_device.device(), renderpass, nullptr);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    spdlog::info("Generating irradiance cube took {} ms", tDiff);
}

void AssetManager::generateBRDFlut() {
    auto tStart = std::chrono::high_resolution_clock::now();

    const VkFormat format = VK_FORMAT_R16G16_SFLOAT;	// R16G16 is supported pretty much everywhere
    const int32_t dim = 512;

    // Image
    VkImageCreateInfo imageCI = vkinit::imageCreateInfo();
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = format;
    imageCI.extent.width = dim;
    imageCI.extent.height = dim;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VK_CHECK_RESULT(vkCreateImage(m_device.device(), &imageCI, nullptr, &lutBrdf->m_image));
    VkMemoryAllocateInfo memAlloc = vkinit::memoryAllocateInfo();
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(m_device.device(), lutBrdf->m_image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = m_device.getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device.device(), &memAlloc, nullptr, &lutBrdf->m_deviceMemory));
    VK_CHECK_RESULT(vkBindImageMemory(m_device.device(), lutBrdf->m_image, lutBrdf->m_deviceMemory, 0));
    // Image view
    VkImageViewCreateInfo viewCI = vkinit::imageViewCreateInfo();
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = format;
    viewCI.subresourceRange = {};
    viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.levelCount = 1;
    viewCI.subresourceRange.layerCount = 1;
    viewCI.image = lutBrdf->m_image;
    VK_CHECK_RESULT(vkCreateImageView(m_device.device(), &viewCI, nullptr, &lutBrdf->m_view));
    // Sampler
    VkSamplerCreateInfo samplerCI = vkinit::samplerCreateInfo();
    samplerCI.magFilter = VK_FILTER_LINEAR;
    samplerCI.minFilter = VK_FILTER_LINEAR;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 1.0f;
    samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(m_device.device(), &samplerCI, nullptr, &lutBrdf->m_sampler));

    lutBrdf->m_descriptor.imageView = lutBrdf->m_view;
    lutBrdf->m_descriptor.sampler = lutBrdf->m_sampler;
    lutBrdf->m_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    lutBrdf->m_pDevice = &m_device;

    // FB, Att, RP, Pipe, etc.
    VkAttachmentDescription attDesc = {};
    // Color attachment
    attDesc.format = format;
    attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
    attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Create the actual renderpass
    VkRenderPassCreateInfo renderPassCI{};
    renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCI.attachmentCount = 1;
    renderPassCI.pAttachments = &attDesc;
    renderPassCI.subpassCount = 1;
    renderPassCI.pSubpasses = &subpassDescription;
    renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassCI.pDependencies = dependencies.data();

    VkRenderPass renderpass = VK_NULL_HANDLE;
    VK_CHECK_RESULT(vkCreateRenderPass(m_device.device(), &renderPassCI, nullptr, &renderpass));

    VkFramebufferCreateInfo framebufferCI = vkinit::framebufferCreateInfo();
    framebufferCI.renderPass = renderpass;
    framebufferCI.attachmentCount = 1;
    framebufferCI.pAttachments = &lutBrdf->m_view;
    framebufferCI.width = dim;
    framebufferCI.height = dim;
    framebufferCI.layers = 1;

    VkFramebuffer framebuffer;
    VK_CHECK_RESULT(vkCreateFramebuffer(m_device.device(), &framebufferCI, nullptr, &framebuffer));

    // Descriptors
    VkDescriptorSetLayout descriptorsetlayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {};
    VkDescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vkinit::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device.device(), &descriptorsetlayoutCI, nullptr, &descriptorsetlayout));

    // Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = { vkinit::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) };
    VkDescriptorPoolCreateInfo descriptorPoolCI = vkinit::descriptorPoolCreateInfo(poolSizes, 2);
    VkDescriptorPool descriptorpool;
    VK_CHECK_RESULT(vkCreateDescriptorPool(m_device.device(), &descriptorPoolCI, nullptr, &descriptorpool));

    // Descriptor sets
    VkDescriptorSet descriptorset;
    VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device.device(), &allocInfo, &descriptorset));

    // Pipeline layout
    VkPipelineLayout pipelinelayout;
    VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_device.device(), &pipelineLayoutCI, nullptr, &pipelinelayout));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vkinit::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState = vkinit::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineColorBlendAttachmentState blendAttachmentState = vkinit::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState = vkinit::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState = vkinit::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState = vkinit::pipelineViewportStateCreateInfo(1, 1);
    VkPipelineMultisampleStateCreateInfo multisampleState = vkinit::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = vkinit::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    VkPipelineVertexInputStateCreateInfo emptyInputState = vkinit::pipelineVertexInputStateCreateInfo();
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    VkGraphicsPipelineCreateInfo pipelineCI = vkinit::pipelineCreateInfo(pipelinelayout, renderpass);
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pDepthStencilState = &depthStencilState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.stageCount = 2;
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.pVertexInputState = &emptyInputState;


    // 4) Fill your PipelineConfigInfo
    PipelineConfigInfo cfg{};
    VkSandboxPipeline::defaultPipelineConfigInfo(cfg);

    cfg.bindingDescriptions.clear();
    cfg.attributeDescriptions.clear();
    cfg.renderPass = renderpass;
    cfg.pipelineLayout = pipelinelayout;
    // viewport & scissor will be dynamic
    cfg.dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    cfg.dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    cfg.dynamicStateInfo.pDynamicStates = cfg.dynamicStateEnables.data();
    cfg.dynamicStateInfo.dynamicStateCount = (uint32_t)cfg.dynamicStateEnables.size();

    // Look-up-table (from BRDF) pipeline
    std::string vert = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/gen_brdflut.vert.spv";
    std::string frag = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/gen_brdflut.frag.spv";
    VkSandboxPipeline brdfPipeline{ m_device, vert, frag, cfg };

    // Render
    VkClearValue clearValues[1];
    clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

    VkRenderPassBeginInfo renderPassBeginInfo = vkinit::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderpass;
    renderPassBeginInfo.renderArea.extent.width = dim;
    renderPassBeginInfo.renderArea.extent.height = dim;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.framebuffer = framebuffer;

    VkCommandBuffer cmdBuf = m_device.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport = vkinit::viewport((float)dim, (float)dim, 0.0f, 1.0f);
    VkRect2D scissor = vkinit::rect2D(dim, dim, 0, 0);
    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
    vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
 
    brdfPipeline.bind(cmdBuf);
    vkCmdDraw(cmdBuf, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmdBuf);
    m_device.flushCommandBuffer(cmdBuf, m_transferQueue);

    vkQueueWaitIdle(m_transferQueue);

    vkDestroyFramebuffer(m_device.device(), framebuffer, nullptr);
    vkDestroyRenderPass(m_device.device(), renderpass, nullptr);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
}

void AssetManager::generatePrefilteredEnvMap() {

}


std::shared_ptr<VkSandboxOBJmodel> AssetManager::loadObjModel(
    const std::string& name,
    const std::string& filepath,
    bool isSkybox
) {
    // 1) cache check
    if (auto it = m_objModelCache.find(name); it != m_objModelCache.end())
        return it->second;

    // 2) load
    auto model = VkSandboxOBJmodel::createModelFromFile(m_device, filepath, isSkybox);

    // 3) cache & return
    m_objModelCache[name] = model;
    return model;
}


std::shared_ptr<vkglTF::Model> AssetManager::loadGLTFmodel(
    const std::string& name,
    const std::string& filepath,
    uint32_t gltfFlags,
    float scale
) {
    if (auto it = m_gltfModelCache.find(name); it != m_gltfModelCache.end())
        return it->second;

    auto model = std::make_shared<vkglTF::Model>();
    model->loadFromFile(filepath, &m_device, m_device.graphicsQueue(), gltfFlags, scale);

    m_gltfModelCache[name] = model;
    return model;
}

std::shared_ptr<VkSandboxTexture> AssetManager::loadCubemap(
    const std::string& name,
    const std::string& ktxFilename,
    VkFormat format,
    VkImageUsageFlags usageFlags,
    VkImageLayout initialLayout)
{
    if (auto it = m_textures.find(name); it != m_textures.end())
        return it->second;

    auto tex = std::make_shared<VkSandboxTexture>();
    tex->m_pDevice = &m_device;
    try {
        tex->KtxLoadCubemapFromFile(
            name,
            ktxFilename,
            format,
            &m_device,
            m_device.graphicsQueue(),
            usageFlags,
            initialLayout
        );
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to load HDR cubemap '" + name + "': " + e.what());
    }

    registerTextureIfNeeded(name, tex, m_textures, m_textureIndexMap, m_textureList);
    return tex;
}




void AssetManager::registerTextureIfNeeded(
    const std::string& name,
    const std::shared_ptr<VkSandboxTexture>& tex,
    std::unordered_map<std::string, std::shared_ptr<VkSandboxTexture>>& textures,
    std::unordered_map<std::string, size_t>& textureIndexMap,
    std::vector<std::shared_ptr<VkSandboxTexture>>& textureList)
{
    if (textures.find(name) == textures.end()) {
        textures[name] = tex;
        textureList.push_back(tex);
        textureIndexMap[name] = textureList.size() - 1;
    }
}