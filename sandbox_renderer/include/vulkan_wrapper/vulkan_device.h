#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vector>

#include "vk_tools/vk_tools.h"
#include "vulkan_instance.h"
#include "interfaces/window_i.h"

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    bool graphicsFamilyHasValue = false;
    bool presentFamilyHasValue = false;
    bool isComplete() { return graphicsFamilyHasValue && presentFamilyHasValue; }
};

class VkSandboxDevice
{
public:

    VkSandboxDevice(VkSandboxInstance& instance, IWindow& window);
    ~VkSandboxDevice();

    // Not copyable or movable
    VkSandboxDevice(const VkSandboxDevice&) = delete;
    void operator=(const VkSandboxDevice&) = delete;
    VkSandboxDevice(VkDevice&&) = delete;
    VkSandboxDevice& operator=(VkSandboxDevice&&) = delete;

    VkCommandPool getCommandPool() { return m_commandPool; }

    uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound = nullptr) const;

    VkDevice device() const { return m_logicalDevice; }

    VkSurfaceKHR surface() const { return m_surface; }
    VkQueue graphicsQueue() const { return m_graphicsQueue; }
    VkQueue presentQueue() const { return m_presentQueue; }
    uint32_t graphicsQueueFamilyIndex() const { return m_queueFamilyIndices.graphicsFamily; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkExtent2D getSwapchainExtent() const;

    SwapChainSupportDetails getSwapChainSupport() { return querySwapChainSupport(m_physicalDevice); }
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(m_physicalDevice); }
    VkFormat findSupportedFormat(
        const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

    // Buffer Helper Functions
    void createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void copyBufferToImage(
        VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layerCount);

    void createImageWithInfo(
        const VkImageCreateInfo& imageInfo,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& imageMemory);

    void transitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t mipLevels,
        uint32_t layerCount);

    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin = false);
    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);

    /// Ends, submits and frees a one‑time command buffer
    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);
    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    VkPhysicalDeviceProperties m_deviceProperties{};

    VkPhysicalDeviceMemoryProperties m_deviceMemoryProperties;
    VkPhysicalDeviceFeatures m_enabledFeatures;

    VkResult createBuffer(VkBufferUsageFlags usageFlags,
                          VkMemoryPropertyFlags memoryPropertyFlags,
                          VkDeviceSize size,
                          VkBuffer* buffer,
                          VkDeviceMemory* memory,
                          void* data = nullptr);

    
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    bool hasStencilComponent(VkFormat format);

    // Query all the swapchain-support information for OUR surface:
    SwapChainSupportDetails querySwapchainSupport(VkSurfaceKHR surface) const;

    // Pick the best surface format you like (srgb first, else the first you get):
    VkSurfaceFormatKHR
        chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;

    // Pick the best present mode (mailbox if available, else FIFO):
    VkPresentModeKHR
        chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& presentModes) const;

    // Clamp the extent to the device’s capabilities:
    VkExtent2D
        chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D desiredExtent) const;

private:


    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createSurface();
    bool isDeviceSuitable(VkPhysicalDevice device);


public:
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_logicalDevice;

private:

    IWindow& m_window;
    VkSandboxInstance& m_instance;
    
    VkCommandPool m_commandPool;
    VkSurfaceKHR m_surface;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    QueueFamilyIndices m_queueFamilyIndices;


    const std::vector<const char*> m_deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_MAINTENANCE1_EXTENSION_NAME,
        VK_KHR_MAINTENANCE3_EXTENSION_NAME
        //VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        //VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        //VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        //VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        //VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
    };
};
