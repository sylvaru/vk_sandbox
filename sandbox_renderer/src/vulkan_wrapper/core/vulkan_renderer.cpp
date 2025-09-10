#include "vulkan_wrapper/core/vulkan_renderer.h"
#include <stdexcept>
#include <array>
#include <cassert>
#include <spdlog/spdlog.h>

VkSandboxRenderer::VkSandboxRenderer(VkSandboxDevice& device, SandboxWindow& window)
    : m_device(device),
    m_window(window)
{
    recreateSwapchain();

    createGlobalDescriptorObjects();
    allocateGlobalDescriptors();
}

VkSandboxRenderer::~VkSandboxRenderer() {
    freeCommandBuffers();
}
void VkSandboxRenderer::createGlobalDescriptorObjects() {


    m_pool = VkSandboxDescriptorPool::Builder{ m_device }
        .setMaxSets(FrameCount + 512) // allow plenty of descriptor sets for materials
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FrameCount)
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256) // accomodate many material textures
        .setPoolFlags(VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT)
        .build();



    // === global UBO layout (set=0) ===
    m_globalLayout = VkSandboxDescriptorSetLayout::Builder{ m_device }
        .addBinding(0,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_ALL_GRAPHICS)
        .build();
}

void VkSandboxRenderer::allocateGlobalDescriptors() {
    
    m_uboBuffers.resize(FrameCount);
    for (uint32_t i = 0; i < FrameCount; i++) {
        m_uboBuffers[i] = std::make_unique<VkSandboxBuffer>(
            m_device,
            sizeof(GlobalUbo),
            1,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        m_uboBuffers[i]->map();
    }

    m_globalDescriptorSets.resize(FrameCount);
    for (uint32_t i = 0; i < FrameCount; i++) {
        auto bufInfo = m_uboBuffers[i]->descriptorInfo();
        VkDescriptorSet set;
        // allocate
        if (!m_pool->allocateDescriptor(
                m_globalLayout->getDescriptorSetLayout(),
                set, 0)) {
            throw std::runtime_error("Failed to allocate global descriptor set");
        }
        // write
        VkSandboxDescriptorWriter(*m_globalLayout, *m_pool)
            .writeBuffer(0, &bufInfo)
            .build(set);

        m_globalDescriptorSets[i] = set;
    }
    
}

void VkSandboxRenderer::initializeSystems(IAssetProvider& provider, IScene& scene) {
    // grab the things every system will need
    VkRenderPass rp = m_swapchain->getRenderPass();
    VkDescriptorSetLayout globalLayout = m_globalLayout->getDescriptorSetLayout();
    VkSandboxDescriptorPool& pool = *m_pool;


    // --- SKYBOX SYSTEM (create into the member, then configure) ---
    m_skyboxSystem = std::make_unique<SkyboxIBLrenderSystem>(m_device, rp, globalLayout);

    // Try to get skybox object from scene
    if (auto skyboxOpt = scene.getSkyboxObject()) {
        IGameObject& skyObj = skyboxOpt.value().get();

        // Get and set model (if available)
        if (auto skyboxModelBase = provider.getGLTFmodel("cube")) {
            m_skyboxSystem->setSkyboxModel(skyboxModelBase);
        }
        else {
            spdlog::warn("Skybox object has no model");
        }

        // Get and set cubemap (may throw)
        auto cubemapName = skyObj.getCubemapTextureName();
        try {
            VkDescriptorImageInfo cubemapDesc = provider.getCubemapDescriptor(cubemapName);

            spdlog::info("[Renderer] got cubemapDesc: view=0x{:x}, sampler=0x{:x}, layout={}",
                (uintptr_t)cubemapDesc.imageView,
                (uintptr_t)cubemapDesc.sampler,
                (int)cubemapDesc.imageLayout);

            if (cubemapDesc.imageView == VK_NULL_HANDLE || cubemapDesc.sampler == VK_NULL_HANDLE) {
                spdlog::error("[Renderer] cubemap descriptor has null view or sampler! This will produce garbage in shader.");
            }

            m_skyboxSystem->setCubemapTexture(cubemapDesc);
            m_skyboxSystem->m_bHasCubemap = true; // if that flag is public/accessible; otherwise let init handle allocation
            m_skyboxSystem->init(
                m_device,
                m_swapchain->getRenderPass(),
                m_globalLayout->getDescriptorSetLayout(),
                *m_pool,
                FrameCount);
        }
        catch (const std::exception& e) {
            spdlog::warn("Skybox cubemap '{}' not found: {}", cubemapName, e.what());
        }
    }
    else {
        spdlog::warn("No skybox object found in scene");
    }



    m_sceneSystem = std::make_unique<SceneRenderSystem>(
        m_device,
        rp,
        globalLayout,
        provider
    );

    m_sceneSystem->init(
        m_device,
        m_swapchain->getRenderPass(),
        m_globalLayout->getDescriptorSetLayout(),
        *m_pool,
        FrameCount);
    


    // gltf pbr system
    m_gltfPbrSystem = std::make_unique<GltfPbrRenderSystem>(
        m_device,
        m_swapchain->getRenderPass(),
        m_globalLayout->getDescriptorSetLayout(),
        provider
    );

    m_gltfPbrSystem->init(
        m_device,
        m_swapchain->getRenderPass(),
        m_globalLayout->getDescriptorSetLayout(),
        *m_pool,
        FrameCount
    );
    // point light system
    m_pointLightSystem = std::make_unique<PointLightRS>(
        m_device,
        m_swapchain->getRenderPass(),
        m_globalLayout->getDescriptorSetLayout()
    );

   
    m_pointLightSystem->init(
        m_device,
        m_swapchain->getRenderPass(),
        m_globalLayout->getDescriptorSetLayout(),
        *m_pool,
        FrameCount
    );

}
void VkSandboxRenderer::updateSystems(FrameInfo& frame, GlobalUbo& ubo, float deltaTime)
{
    m_pointLightSystem->update(frame, ubo);
}

void VkSandboxRenderer::renderSystems(FrameInfo& frame) {

    RenderGraph graph;

    RGHandle color = graph.importExternal("SwapchainColor");
    RGHandle depth = graph.importExternal("Depth");

    RGContext rgCtx;
    rgCtx.device = &m_device;
    rgCtx.cmd = frame.commandBuffer;
    rgCtx.frameIndex = frame.frameIndex;
    rgCtx.globalSet = frame.globalDescriptorSet;

    // --- Skybox: first ---
    if (m_skyboxSystem) {
        graph.addPass("Skybox")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this, &frame](const RGContext& ctx) {
            m_skyboxSystem->record(ctx, frame);
                });
    }

    // --- Scene (static or non-pbr objects) ---
    if (m_sceneSystem) {
        graph.addPass("Scene")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this, &frame](const RGContext& ctx) {
            m_sceneSystem->record(ctx, frame);
                });
    }

    // --- GLTF PBR (models that bind material images) ---
    if (m_gltfPbrSystem) {
        graph.addPass("GltfPbr")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this, &frame](const RGContext& ctx) {
            m_gltfPbrSystem->record(ctx, frame);
                });
    }

    // --- Point light pass (additive) ---
    if (m_pointLightSystem) {
        graph.addPass("PointLights")
            .write(color, RGUsage::WriteColor)
            .setExecute([this, &frame](const RGContext& ctx) {
            m_pointLightSystem->record(ctx, frame);
                });
    }


    graph.compile();
    graph.execute(rgCtx);

}

ISandboxRenderer::FrameContext VkSandboxRenderer::beginFrame() {
    // 1) AcquireNextImage does the fence‐wait for the current in‐flight frame internally
    VkResult result = m_swapchain->acquireNextImage(&m_currentImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return {}; // invalid context
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    // 2) Map imageIndex → frameIndex (in‐flight slot)
    //    Typically they’re the same because swapchain was created with MAX_FRAMES_IN_FLIGHT images.
    m_currentFrameIndex = m_currentImageIndex % FrameCount;

    // 3) Begin recording
    VkCommandBuffer cmd = m_commandBuffers[m_currentImageIndex];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    m_bIsFrameStarted = true;

    // 4) Build the FrameContext
    ISandboxRenderer::FrameContext ctx{};
    ctx.graphicsCommandBuffers = m_commandBuffers;
    ctx.primaryGraphicsCommandBuffer = cmd;
    ctx.frameIndex = m_currentFrameIndex;
    // let the game know which fence is in flight if it wants to wait on it:
    ctx.frameFence = m_swapchain->getFence(m_currentFrameIndex);
    return ctx;
}
void VkSandboxRenderer::endFrame(FrameContext& frame) {
    assert(m_bIsFrameStarted && "endFrame() called when no frame in progress");

    // 1) finish command buffer
    if (vkEndCommandBuffer(frame.primaryGraphicsCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer");
    }

    // 2) hand off to swapchain (which will queue submit & signal its fence)
    VkResult result = m_swapchain->submitCommandBuffers(
        &frame.primaryGraphicsCommandBuffer,
        &m_currentImageIndex  // you could also store this in frame
    );

    // 3) handle resize/out‑of‑date
    if (result == VK_ERROR_OUT_OF_DATE_KHR
        || result == VK_SUBOPTIMAL_KHR
        || m_window.wasWindowResized())
    {
        m_window.resetWindowResizedFlag();
        recreateSwapchain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swap chain image!");
    }


    m_bIsFrameStarted = false;
    m_currentFrameIndex = (m_currentFrameIndex + 1) % FrameCount;
}
void VkSandboxRenderer::beginSwapChainRenderPass(FrameContext& frame)
{
    assert(m_bIsFrameStarted && "Can't call beginSwapChainRenderPass if frame is not in progress");
 

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_swapchain->getRenderPass();
    renderPassInfo.framebuffer = m_swapchain->getFrameBuffer(m_currentImageIndex);

    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = m_swapchain->getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0.5f, 0.5f, 0.5f, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(frame.primaryGraphicsCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchain->getSwapChainExtent().width);
    viewport.height = static_cast<float>(m_swapchain->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{ {0, 0}, m_swapchain->getSwapChainExtent() };
    vkCmdSetViewport(frame.primaryGraphicsCommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(frame.primaryGraphicsCommandBuffer, 0, 1, &scissor);
}

void VkSandboxRenderer::endSwapChainRenderPass(FrameContext& frame)
{
  
    assert(m_bIsFrameStarted && "Can't call endSwapChainRenderPass if frame is not in progress");
    
    vkCmdEndRenderPass(frame.primaryGraphicsCommandBuffer);
}
void VkSandboxRenderer::createCommandBuffers() {
    size_t imageCount = m_swapchain->imageCount();
    m_commandBuffers.resize(imageCount);
    //m_commandBuffers.resize(VkSandboxSwapchain::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

    if (vkAllocateCommandBuffers(
        m_device.device(),
        &allocInfo,
        m_commandBuffers.data()
    ) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}
void VkSandboxRenderer::recreateSwapchain() {

    auto extent = m_window.getExtent();
    glfwWaitEvents();
    extent = m_window.getExtent();

    vkDeviceWaitIdle(m_device.device());

    if (m_swapchain == nullptr) {

        m_swapchain = std::make_unique<VkSandboxSwapchain>(
            m_device,
            extent
        );
    }
    else {
        std::shared_ptr oldSwapchain = std::move(m_swapchain);
        m_swapchain = std::make_unique<VkSandboxSwapchain>(m_device, extent, oldSwapchain);
        if (!oldSwapchain->compareSwapFormats(*m_swapchain.get())) {
            throw std::runtime_error("Swap chain image(or depth) format has changed");
        }
    }
    createCommandBuffers();
}
void VkSandboxRenderer::freeCommandBuffers() {
    if (m_commandBuffers.empty()) { return; }

    vkFreeCommandBuffers(
        m_device.device(),
        m_device.getCommandPool(),
        static_cast<uint32_t>(m_commandBuffers.size()),
        m_commandBuffers.data()
    );

    m_commandBuffers.clear();
}

void VkSandboxRenderer::waitDeviceIdle() {
    vkDeviceWaitIdle(m_device.device());
}