#include "vulkan_wrapper/vulkan_renderer.h"
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
    // === build pool exactly like old buildLayouts() ===
    m_pool = VkSandboxDescriptorPool::Builder{ m_device }
        .setMaxSets(FrameCount + 3 /*texture+sky+ibl*/)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FrameCount)
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10)
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

void VkSandboxRenderer::initializeSystems(IAssetProvider& provider) {
    // grab the things every system will need
    VkRenderPass rp = m_swapchain->getRenderPass();
    VkDescriptorSetLayout globalLayout = m_globalLayout->getDescriptorSetLayout();
    VkSandboxDescriptorPool& pool = *m_pool;

    // Create skybox system (note: only construct, do not init yet)
    //auto skyboxSystem = std::make_unique<SkyboxIBLrenderSystem>(m_device, rp, globalLayout);

    // Ask provider for the assets we need
    // model
    auto skyModel = provider.getGLTFmodel("cube"); // or the name you set in assets.json
    if (skyModel) {
        //skyboxSystem->setSkyboxModel(skyModel);
    }
    else {
        spdlog::warn("No skybox model found in provider for 'cube'");
    }

    // cubemap descriptor (VkDescriptorImageInfo)
    // choose the name you used in default_scene_assets.json (e.g. "skybox_hdr")
    try {
        VkDescriptorImageInfo cubemapDesc = provider.getCubemapDescriptor("skybox_hdr");
        //skyboxSystem->setSkyboxCubemap(cubemapDesc);
    }
    catch (const std::exception& e) {
        spdlog::warn("Skybox: cubemap not found: {}", e.what());
    }

    // push it into systems list, before we init them
    //m_systems.push_back(std::move(skyboxSystem));
  
    m_systems.push_back(std::make_unique<ObjRenderSystem>(
        m_device,
        rp,
        globalLayout
    ));

    m_systems.push_back(std::make_unique<GltfRenderSystem>(
        m_device,
        rp,
        globalLayout,
        provider
    ));

    m_systems.push_back(std::make_unique<PointLightRS>(
        m_device,
        rp,
        globalLayout
    ));


    // Now call init() hooks
    for (auto& sys : m_systems) {
        sys->init(
            m_device,
            rp,
            globalLayout,
            pool,
            FrameCount
        );
    }
}
void VkSandboxRenderer::updateSystems(FrameInfo& frame, GlobalUbo& ubo, float deltaTime)
{
    for (auto& renderSystem : m_systems) {
        renderSystem->update(frame, ubo);
    }
}

void VkSandboxRenderer::renderSystems(FrameInfo& frame) {
    // upload camera UBO into m_uboBuffers[frame.frameIndex]...
    // loop all your render systems:
    for (auto& sys : m_systems) {
        sys->render(frame);
    }
}

void VkSandboxRenderer::recreateSwapchain() {

    auto extent = m_window.getExtent();
    while (extent.width == 0 || extent.width == 0)
    {
        glfwWaitEvents();
        extent = m_window.getExtent();
    }

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

void VkSandboxRenderer::freeCommandBuffers() {
    if (m_commandBuffers.empty()) {
        return;
    }

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