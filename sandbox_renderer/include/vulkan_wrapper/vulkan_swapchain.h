#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

class VkSandboxDevice;

class VkSandboxSwapchain {
public:

    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    VkSandboxSwapchain(
        VkSandboxDevice& deviceRef,
        VkExtent2D extent);

    VkSandboxSwapchain(VkSandboxDevice& device,
        VkExtent2D      extent,
        std::shared_ptr<VkSandboxSwapchain> oldSwapchain);
    ~VkSandboxSwapchain();

    VkSandboxSwapchain(const VkSandboxSwapchain&) = delete;
    VkSandboxSwapchain& operator=(const VkSandboxSwapchain&) = delete;


    VkFramebuffer getFrameBuffer(int index) { return m_swapChainFramebuffers[index]; }
    VkRenderPass getRenderPass() { return m_renderPass; }
    VkImageView getImageView(int index) { return m_swapChainImageViews[index]; }
    size_t imageCount() { return m_swapChainImages.size(); }
    VkFormat getSwapChainImageFormat() { return m_swapChainImageFormat; }
    VkExtent2D getSwapChainExtent() { return m_swapChainExtent; }
    uint32_t width() { return m_swapChainExtent.width; }
    uint32_t height() { return m_swapChainExtent.height; }
    VkImage getImage(size_t index) const { return m_swapChainImages.at(index); }
    VkImage getDepthImage(size_t index) const { return m_depthImages.at(index); }
    VkImageView getDepthImageView(size_t index) const { return m_depthImageViews.at(index); }
    static constexpr int GetMaxFramesInFlight(){ return MAX_FRAMES_IN_FLIGHT; }

    VkResult acquireNextImage(uint32_t* imageIndex);
    VkResult submitCommandBuffers(const VkCommandBuffer* buffers, uint32_t* imageIndex);

    float    extentAspectRatio() {
             return static_cast<float>(m_swapChainExtent.width) / static_cast<float>(m_swapChainExtent.height);
    }
    VkFormat findDepthFormat();

    bool     compareSwapFormats(const VkSandboxSwapchain& swapChain) const {
             return swapChain.m_swapChainDepthFormat == m_swapChainDepthFormat &&
             swapChain.m_swapChainImageFormat == m_swapChainImageFormat;
    }
    VkFence getFence(uint32_t frameIndex) const {
        return m_inFlightFences[frameIndex];
    }

private:
    void init();
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();


    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& availablePresentModes);

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);


    VkFormat m_swapChainImageFormat;
    VkFormat m_swapChainDepthFormat;
    VkExtent2D m_swapChainExtent;

    std::vector<VkFramebuffer> m_swapChainFramebuffers;
    VkRenderPass m_renderPass;

    std::vector<VkImage> m_depthImages;
    std::vector<VkDeviceMemory> m_depthImageMemory;
    std::vector<VkImageView> m_depthImageViews;
    std::vector<VkImage> m_swapChainImages;
    std::vector<VkImageView> m_swapChainImageViews;

    VkSandboxDevice& m_device;
    VkExtent2D m_windowExtent;

    VkSwapchainKHR m_swapChain;
    std::shared_ptr<VkSandboxSwapchain> m_oldSwapChain;

    size_t                  m_swapChainImageCount = 0;
    std::vector<VkSemaphore> m_imageAvailableSemaphores; // one per image
    std::vector<VkSemaphore> m_renderFinishedSemaphores; // one per image
    std::vector<VkFence>     m_inFlightFences;           // one per CPU-frame
    std::vector<VkFence>     m_imagesInFlight;           // last-used fence per image
    size_t                  m_currentFrame = 0;
};
