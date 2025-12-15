#include "common/renderer_pch.h"
#include "vulkan_wrapper/core/vulkan_renderer.h"
#include "vulkan_wrapper/core/render_graph.h"
#include "vulkan_wrapper/render_systems/gltf_render_system.h"
#include "vulkan_wrapper/render_systems/scene_render_system.h"
#include "vulkan_wrapper/render_systems/skybox_render_system.h"
#include "vulkan_wrapper/render_systems/pointlight_render_system.h"

constexpr uint32_t MAX_BINDLESS_TEXTURES = 8192;


VkSandboxRenderer::VkSandboxRenderer(
    VkSandboxDevice& device,
    IWindow& window)
    : m_device(device),
    m_window(window)
{
    createSwapChain();
    createGlobalDescriptorObjects();
    allocateGlobalDescriptors();
}

VkSandboxRenderer::~VkSandboxRenderer() 
{
    freeCommandBuffers();
}
void VkSandboxRenderer::createGlobalDescriptorObjects() 
{

    // TODO: stop using magic numbers here
    m_pool = VkSandboxDescriptorPool::Builder{ m_device }
        .setMaxSets(FrameCount + 512)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FrameCount)
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256)
        .setPoolFlags(VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT)
        .build();

    m_globalLayout = VkSandboxDescriptorSetLayout::Builder{ m_device }
        .addBinding(0,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_ALL_GRAPHICS)
        .build();

    m_bindlessLayout = VkSandboxDescriptorSetLayout::Builder{ m_device }
        .addBinding(0,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            MAX_BINDLESS_TEXTURES,
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        )
        .build();
}

void VkSandboxRenderer::allocateGlobalDescriptors() 
{
    
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

        if (!m_pool->allocateDescriptor(
                m_globalLayout->getDescriptorSetLayout(),
                set, 0)) {
            throw std::runtime_error("Failed to allocate global descriptor set");
        }

        VkSandboxDescriptorWriter(*m_globalLayout, *m_pool)
            .writeBuffer(0, &bufInfo)
            .build(set);

        m_globalDescriptorSets[i] = set;
    }
    
}

void VkSandboxRenderer::allocateBindlessDescriptors(
    const std::vector<VkDescriptorImageInfo>& images,
    VkDescriptorBufferInfo& materialBufferInfo) 
{
    VkSandboxDescriptorWriter(*m_bindlessLayout, *m_pool)
        .writeImage(0, images.data(),
            static_cast<uint32_t>(images.size()))
        .writeBuffer(1, &materialBufferInfo)
        .build(m_bindlessDescriptorSet);
}

void VkSandboxRenderer::initializeSystems(
    IAssetProvider& provider,
    IScene& scene) 
{
    VkRenderPass rp = m_swapchain->getRenderPass();
    VkDescriptorSetLayout globalLayout = m_globalLayout->getDescriptorSetLayout();
    VkSandboxDescriptorPool& pool = *m_pool;

    // create ibl descriptors
    createIblDescriptors(provider);

    // create skybox render system
    m_skyboxSystem = std::make_unique<SkyboxRenderSystem>(m_device, rp, globalLayout, pool);

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

            if (cubemapDesc.imageView == VK_NULL_HANDLE || cubemapDesc.sampler == VK_NULL_HANDLE) {
                spdlog::error("[Renderer] cubemap descriptor has null view or sampler! This will produce garbage in shader.");
            }

            m_skyboxSystem->setCubemapTexture(cubemapDesc);
            m_skyboxSystem->m_bHasCubemap = true;
            m_skyboxSystem->init(m_device, rp, globalLayout, pool);
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
        m_iblSetLayout,
        m_iblDescriptorSets
    );

    m_sceneSystem = std::make_unique<SceneRenderSystem>(
        m_device,
        rp,
        globalLayout,
        m_iblSetLayout,
        m_iblDescriptorSets
    );



    m_pointLightSystem = std::make_unique<PointLightRenderSystem>(
        m_device,
        rp,
        globalLayout
    );


}

void VkSandboxRenderer::updateSystems(FrameInfo& frame, GlobalUbo& ubo, float deltaTime)
{
    m_pointLightSystem->update(frame, ubo);
}

void VkSandboxRenderer::renderSystems(FrameInfo& info, FrameContext& frame) 
{
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

    if (m_imguiInitialized) {
        graph.addPass("ImGui")
            .write(color, RGUsage::WriteColor)
            .setExecute([this](const RGContext& ctx) {
            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();
            if (draw_data && draw_data->CmdListsCount > 0) {
                ImGui_ImplVulkan_RenderDrawData(draw_data, ctx.cmd);
            }
                });
    }


    graph.compile();
    graph.emitPreBarriers(rgCtx);
    beginSwapChainRenderPass(frame);
    graph.executePasses(rgCtx);
    endSwapChainRenderPass(frame);
    graph.emitPostBarriers(rgCtx);
}


ISandboxRenderer::FrameContext VkSandboxRenderer::beginFrame() 
{
    // AcquireNextImage does the fence‐wait for the current in‐flight frame internally
    VkResult result = m_swapchain->acquireNextImage(&m_currentImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        createSwapChain();
        return {};
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    // Map imageIndex → frameIndex (in‐flight slot)
    m_currentFrameIndex = m_currentImageIndex % FrameCount;

    // Begin recording
    VkCommandBuffer cmd = m_commandBuffers[m_currentImageIndex];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    m_bIsFrameStarted = true;

    // Build the FrameContext
    ISandboxRenderer::FrameContext ctx{};
    ctx.graphicsCommandBuffers = m_commandBuffers;
    ctx.primaryGraphicsCommandBuffer = cmd;
    ctx.imageIndex = m_currentImageIndex;
    ctx.frameIndex = m_currentFrameIndex;
    ctx.frameFence = m_swapchain->getFence(m_currentFrameIndex);
    return ctx;
}
void VkSandboxRenderer::endFrame(FrameContext& frame) 
{
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

void VkSandboxRenderer::createCommandBuffers() 
{
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


void VkSandboxRenderer::createSwapChain()
{
    // Wait for valid framebuffer size
    VkExtent2D extent{};
    do {
        m_window.waitEvents(); // block until an event occurs

        int width = 0, height = 0;
        m_window.getFramebufferSize(width, height);

        extent.width = static_cast<uint32_t>(width);
        extent.height = static_cast<uint32_t>(height);

    } while (extent.width == 0 || extent.height == 0);

    vkDeviceWaitIdle(m_device.device());

    if (!m_swapchain) {
        m_swapchain = std::make_unique<VkSandboxSwapchain>(
            m_device,
            extent
        );
    }
    else {
        std::shared_ptr<VkSandboxSwapchain> oldSwapchain = std::move(m_swapchain);

        m_swapchain = std::make_unique<VkSandboxSwapchain>(
            m_device,
            extent,
            oldSwapchain
        );

        if (!oldSwapchain->compareSwapFormats(*m_swapchain)) {
            throw std::runtime_error(
                "Swapchain image or depth format changed");
        }
    }

    m_swapchainImageLayouts.assign(
        m_swapchain->imageCount(),
        VK_IMAGE_LAYOUT_UNDEFINED);

    m_depthImageLayouts.assign(
        m_swapchain->imageCount(),
        VK_IMAGE_LAYOUT_UNDEFINED);

    createCommandBuffers();
}

void VkSandboxRenderer::freeCommandBuffers() 
{
    if (m_commandBuffers.empty()) { return; }

    vkFreeCommandBuffers(
        m_device.device(),
        m_device.getCommandPool(),
        static_cast<uint32_t>(m_commandBuffers.size()),
        m_commandBuffers.data()
    );

    m_commandBuffers.clear();
}


VkCommandBuffer VkSandboxRenderer::createSingleUseCommandBuffer() 
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device.device(), &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    return cmd;
}

void VkSandboxRenderer::flushAndSubmitSingleUseCommandBuffer(VkCommandBuffer cmd) 
{
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(m_device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_device.graphicsQueue());

    vkFreeCommandBuffers(m_device.device(), m_commandPool, 1, &cmd);
}

void VkSandboxRenderer::waitDeviceIdle() 
{
    vkDeviceWaitIdle(m_device.device());
}

void VkSandboxRenderer::createIblDescriptors(IAssetProvider& provider)
{
    if (!m_iblLayout) {
        m_iblLayout = VkSandboxDescriptorSetLayout::Builder{ m_device }
            .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build();
     
        // Keep raw layout handle for pipeline creation
        m_iblSetLayout = m_iblLayout->getDescriptorSetLayout();
    }

    m_iblDescriptorSets.resize(FrameCount);

    // Get descriptor infos from asset provider (these must be valid VkDescriptorImageInfo)
    VkDescriptorImageInfo brdfInfo = provider.getBRDFLUTDescriptor();
    VkDescriptorImageInfo irradianceInfo = provider.getIrradianceDescriptor();
    VkDescriptorImageInfo prefilterInfo = provider.getPrefilteredDescriptor();

    for (uint32_t i = 0; i < FrameCount; ++i) {
        VkDescriptorSet set;
        if (!m_pool->allocateDescriptor(m_iblLayout->getDescriptorSetLayout(), set, 0)) {
            throw std::runtime_error("Failed to allocate IBL descriptor set");
        }

        VkSandboxDescriptorWriter(*m_iblLayout, *m_pool)
            .writeImage(0, &brdfInfo)
            .writeImage(1, &irradianceInfo)
            .writeImage(2, &prefilterInfo)
            .build(set);

        m_iblDescriptorSets[i] = set;
    }
}


// Imgui stuff
void VkSandboxRenderer::initImGui(
    VkInstance instance,
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkQueue graphicsQueue,
    uint32_t queueFamily)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::StyleColorsDark();


    m_imguiDescriptorPool = create_imgui_descriptor_pool(device);

    GLFWwindow* glfwWindow =
        static_cast<GLFWwindow*>(m_window.getNativeHandle());

    ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);


    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = queueFamily;
    init_info.Queue = graphicsQueue;
    init_info.DescriptorPool = m_imguiDescriptorPool;
    init_info.MinImageCount = FrameCount;
    init_info.ImageCount = FrameCount;
    //init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;


    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_PipelineInfo pipeline_info{};
    pipeline_info.RenderPass = m_swapchain->getRenderPass();
    pipeline_info.Subpass = 0;
    ImGui_ImplVulkan_CreateMainPipeline(&pipeline_info);

    m_imguiInitialized = true;
}


void VkSandboxRenderer::beginImGuiFrame() const
{
    if (!m_imguiInitialized) return;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();

}

void VkSandboxRenderer::renderImGui(FrameContext& frame) const
{
    if (!m_imguiInitialized) return;
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, frame.primaryGraphicsCommandBuffer);
}

void VkSandboxRenderer::shutdownImGui() 
{
    if (!m_imguiInitialized) return;

    vkDestroyDescriptorPool(m_device.device(), m_imguiDescriptorPool, nullptr);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    m_imguiInitialized = false;
}

VkDescriptorPool VkSandboxRenderer::create_imgui_descriptor_pool(VkDevice device) 
{
    std::array<VkDescriptorPoolSize, 11> poolSizes = {};
    poolSizes[0] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 };
    poolSizes[1] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 };
    poolSizes[2] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 };
    poolSizes[3] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 };
    poolSizes[4] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 };
    poolSizes[5] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 };
    poolSizes[6] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 };
    poolSizes[7] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 };
    poolSizes[8] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 };
    poolSizes[9] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 };
    poolSizes[10] = VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * static_cast<uint32_t>(poolSizes.size());
    pool_info.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    pool_info.pPoolSizes = poolSizes.data();

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ImGui descriptor pool");
    }
    return descriptorPool;
}

