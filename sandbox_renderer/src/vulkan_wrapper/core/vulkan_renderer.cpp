#include "vulkan_wrapper/core/vulkan_renderer.h"
#include <stdexcept>
#include <array>
#include <cassert>
#include <spdlog/spdlog.h>
#include "vulkan_wrapper/core/render_graph.h"

VkSandboxRenderer::VkSandboxRenderer(
    VkSandboxDevice& device,
    SandboxWindow& window)
    : m_device(device),
    m_window(window)
{
    createSwapChain();
    createGlobalDescriptorObjects();
    allocateGlobalDescriptors();
}

VkSandboxRenderer::~VkSandboxRenderer() {
    freeCommandBuffers();
}
void VkSandboxRenderer::createGlobalDescriptorObjects() {


    m_pool = VkSandboxDescriptorPool::Builder{ m_device }
        .setMaxSets(FrameCount + 512)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FrameCount)
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256)
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

void VkSandboxRenderer::initializeSystems(
    IAssetProvider& provider,
    IScene& scene) 
{
    VkRenderPass rp = m_swapchain->getRenderPass();
    VkDescriptorSetLayout globalLayout = m_globalLayout->getDescriptorSetLayout();
    VkSandboxDescriptorPool& pool = *m_pool;

    m_skyboxSystem = std::make_unique<SkyboxIBLrenderSystem>(m_device, rp, globalLayout, pool);

    if (auto skyboxOpt = scene.getSkyboxObject()) {
        IGameObject& skyObj = skyboxOpt.value().get();

        if (auto skyboxModelBase = provider.getGLTFmodel("cube")) {
            m_skyboxSystem->setSkyboxModel(skyboxModelBase);
        }
        else {
            spdlog::warn("Skybox object has no model");
        }

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
            m_skyboxSystem->m_bHasCubemap = true;
            m_skyboxSystem->init(
                m_device, rp, globalLayout, pool);
        }
        catch (const std::exception& e) {
            spdlog::warn("Skybox cubemap '{}' not found: {}", cubemapName, e.what());
        }
    }
    else {
        spdlog::warn("No skybox object found in scene");
    }


    m_gltfSystem = std::make_unique<GltfRenderSystem>(
        m_device,
        rp,
        globalLayout,
        pool,
        provider,
        FrameCount
    );

    m_pointLightSystem = std::make_unique<PointLightRS>(
        m_device,
        rp,
        globalLayout
    );

    m_sceneSystem = std::make_unique<SceneRenderSystem>(
        m_device,
        rp,
        globalLayout,
        provider
    );

    //m_systems.push_back(std::make_unique<ObjRenderSystem>(
    //    m_device,
    //    rp,
    //    globalLayout
    //));

}
void VkSandboxRenderer::updateSystems(FrameInfo& frame, GlobalUbo& ubo, float deltaTime)
{
    m_pointLightSystem->update(frame, ubo);
}

void VkSandboxRenderer::renderSystems(FrameInfo& info, FrameContext& frame) {
    RenderGraph graph;

    uint32_t imgIndex = frame.imageIndex;
    VkImage swapImg = m_swapchain->getImage(imgIndex);
    VkImageView swapView = m_swapchain->getImageView(imgIndex);
    VkImage depthImg = m_swapchain->getDepthImage(imgIndex);
    VkImageView depthView = m_swapchain->getDepthImageView(imgIndex);
    VkImageLayout colorLastLayout = m_swapchainImageLayouts[imgIndex];
    VkImageLayout depthlastLayout = m_depthImageLayouts[imgIndex];

    RGHandle color = graph.importExternal("SwapchainColor", swapImg, swapView, colorLastLayout);
    RGHandle depth = graph.importExternal("Depth", depthImg, depthView, depthlastLayout);

    RGContext rgCtx{};
    rgCtx.device = &m_device;
    rgCtx.cmd = frame.primaryGraphicsCommandBuffer;
    rgCtx.frameIndex = frame.frameIndex;
    rgCtx.frame = &info;
    rgCtx.globalSet = info.globalDescriptorSet;

    if (m_skyboxSystem) {
        graph.addPass("Skybox")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this](const RGContext& ctx) {
            FrameInfo tmpFrame = *ctx.frame;
            tmpFrame.commandBuffer = ctx.cmd;
            tmpFrame.frameIndex = static_cast<int>(ctx.frameIndex);
            tmpFrame.globalDescriptorSet = ctx.globalSet;
            m_skyboxSystem->record(ctx, tmpFrame);
                });
    }

    if (m_gltfSystem) {
        graph.addPass("GLTF")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this](const RGContext& ctx) {
            FrameInfo tmpFrame = *ctx.frame;
            tmpFrame.commandBuffer = ctx.cmd;
            tmpFrame.frameIndex = static_cast<int>(ctx.frameIndex);
            tmpFrame.globalDescriptorSet = ctx.globalSet;
            m_gltfSystem->record(ctx, tmpFrame);
                });
    }
    if (m_pointLightSystem) {
        graph.addPass("PointLight")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this](const RGContext& ctx) {
            FrameInfo tmpFrame = *ctx.frame;
            tmpFrame.commandBuffer = ctx.cmd;
            tmpFrame.frameIndex = static_cast<int>(ctx.frameIndex);
            tmpFrame.globalDescriptorSet = ctx.globalSet;
            m_pointLightSystem->record(ctx, tmpFrame);
                });
    }
    if (m_sceneSystem) {
        graph.addPass("Scene")
            .write(color, RGUsage::WriteColor)
            .write(depth, RGUsage::WriteDepth)
            .setExecute([this](const RGContext& ctx) {
            FrameInfo tmpFrame = *ctx.frame;
            tmpFrame.commandBuffer = ctx.cmd;
            tmpFrame.frameIndex = static_cast<int>(ctx.frameIndex);
            tmpFrame.globalDescriptorSet = ctx.globalSet;
            m_sceneSystem->record(ctx, tmpFrame);
                });
    }

    graph.compile();
    graph.emitPreBarriers(rgCtx);

    ISandboxRenderer::FrameContext localRenderCtx = frame;
    localRenderCtx.primaryGraphicsCommandBuffer = rgCtx.cmd;
    beginSwapChainRenderPass(localRenderCtx);

    graph.executePasses(rgCtx);

    endSwapChainRenderPass(localRenderCtx);

    graph.emitPostBarriers(rgCtx);

}


ISandboxRenderer::FrameContext VkSandboxRenderer::beginFrame() {
    // 1) AcquireNextImage does the fence‐wait for the current in‐flight frame internally
    VkResult result = m_swapchain->acquireNextImage(&m_currentImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        createSwapChain();
        return {};
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    // 2) Map imageIndex → frameIndex (in‐flight slot)
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
    ctx.imageIndex = m_currentImageIndex;
    ctx.frameIndex = m_currentFrameIndex;
    ctx.frameFence = m_swapchain->getFence(m_currentFrameIndex);
    return ctx;
}
void VkSandboxRenderer::endFrame(FrameContext& frame) {
    assert(m_bIsFrameStarted && "endFrame() called when no frame in progress");

    if (vkEndCommandBuffer(frame.primaryGraphicsCommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer");
    }

    VkResult result = m_swapchain->submitCommandBuffers(
        &frame.primaryGraphicsCommandBuffer,
        &m_currentImageIndex
    );

    if (result == VK_ERROR_OUT_OF_DATE_KHR
        || result == VK_SUBOPTIMAL_KHR
        || m_window.wasWindowResized())
    {
        m_window.resetWindowResizedFlag();
        createSwapChain();
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

    VkCommandBuffer cmd = frame.primaryGraphicsCommandBuffer;

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_swapchain->getRenderPass();
    renderPassInfo.framebuffer = m_swapchain->getFrameBuffer(frame.imageIndex);
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = m_swapchain->getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0.5f, 0.5f, 0.5f, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchain->getSwapChainExtent().width);
    viewport.height = static_cast<float>(m_swapchain->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{ {0, 0}, m_swapchain->getSwapChainExtent() };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void VkSandboxRenderer::endSwapChainRenderPass(FrameContext& frame)
{
    assert(m_bIsFrameStarted && "Can't call endSwapChainRenderPass if frame is not in progress");

    vkCmdEndRenderPass(frame.primaryGraphicsCommandBuffer);
}

void VkSandboxRenderer::createCommandBuffers() {
    size_t imageCount = m_swapchain->imageCount();
    m_commandBuffers.resize(imageCount);

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


void VkSandboxRenderer::createSwapChain() {

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

    m_swapchainImageLayouts.assign(m_swapchain->imageCount(), VK_IMAGE_LAYOUT_UNDEFINED);
    m_depthImageLayouts.assign(m_swapchain->imageCount(), VK_IMAGE_LAYOUT_UNDEFINED);


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