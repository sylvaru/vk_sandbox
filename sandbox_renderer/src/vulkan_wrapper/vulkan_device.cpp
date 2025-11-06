#include "vulkan_wrapper/vulkan_device.h"


#include <cstring>
#include <iostream>
#include <set>
#include <unordered_set>



VkSandboxDevice::VkSandboxDevice(VkSandboxInstance& instance, SandboxWindow& window)
    : m_window(window),
      m_instance(instance)
{
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
}

VkSandboxDevice::~VkSandboxDevice() {
    vkDestroyCommandPool(m_logicalDevice, m_commandPool, nullptr);
    vkDestroyDevice(m_logicalDevice, nullptr);
}


void VkSandboxDevice::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance.instance(), &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::cout << "Device count: " << deviceCount << std::endl;

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance.instance(), &deviceCount, devices.data());

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            m_physicalDevice = device;
            break;
        }
    }

    if (m_physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);
    std::cout << "physical device: " << m_deviceProperties.deviceName << std::endl;

    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_deviceMemoryProperties);
}
void VkSandboxDevice::createSurface() {
    m_window.createSurface(m_instance.instance(), &m_surface);
}
void VkSandboxDevice::createLogicalDevice() {
    m_queueFamilyIndices = findQueueFamilies(m_physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { m_queueFamilyIndices.graphicsFamily, m_queueFamilyIndices.presentFamily };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Descriptor Indexing Features
    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    // Vulkan 1.3 Features
    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.dynamicRendering = VK_TRUE;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.maintenance4 = VK_TRUE;
    vulkan13Features.robustImageAccess = VK_TRUE;

    indexingFeatures.pNext = &vulkan13Features;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.pNext = &indexingFeatures;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &features2;  // for any Vulkan 1.1+ features structs
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(m_deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = m_deviceExtensions.data();
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;

    createInfo.pEnabledFeatures = nullptr;

    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_logicalDevice) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(m_logicalDevice, m_queueFamilyIndices.graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_logicalDevice, m_queueFamilyIndices.presentFamily, 0, &m_presentQueue);
}

void VkSandboxDevice::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findPhysicalQueueFamilies();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_logicalDevice, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

bool VkSandboxDevice::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}


bool VkSandboxDevice::checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    std::set<std::string> requiredExtensions(m_deviceExtensions.begin(), m_deviceExtensions.end());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

QueueFamilyIndices VkSandboxDevice::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        const auto& queueFamily = queueFamilies[i];
        if ((queueFamily.queueCount > 0) && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.graphicsFamily = i;
            indices.graphicsFamilyHasValue = true;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);
        if ((queueFamily.queueCount > 0) && presentSupport) {
            indices.presentFamily = i;
            indices.presentFamilyHasValue = true;
        }
        if (indices.isComplete()) break;
    }
    return indices;
}

SwapChainSupportDetails VkSandboxDevice::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, details.presentModes.data());
    }
    return details;
}

VkFormat VkSandboxDevice::findSupportedFormat(
    const std::vector<VkFormat>& candidates,
    VkImageTiling tiling,
    VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);
        if ((tiling == VK_IMAGE_TILING_LINEAR) && ((props.linearTilingFeatures & features) == features)) {
            return format;
        }
        else if ((tiling == VK_IMAGE_TILING_OPTIMAL) && ((props.optimalTilingFeatures & features) == features)) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

uint32_t VkSandboxDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i))
            && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void VkSandboxDevice::createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create vertex buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_logicalDevice, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory!");
    }

    vkBindBufferMemory(m_logicalDevice, buffer, bufferMemory, 0);
}
VkResult VkSandboxDevice::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* memory, void* data)
{
    // Create the buffer handle
    VkBufferCreateInfo bufferCreateInfo = vkinit::bufferCreateInfo(usageFlags, size);
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(m_logicalDevice, &bufferCreateInfo, nullptr, buffer));

    // Create the memory backing up the buffer handle
    VkMemoryRequirements memReqs;
    VkMemoryAllocateInfo memAlloc = vkinit::memoryAllocateInfo();
    vkGetBufferMemoryRequirements(m_logicalDevice, *buffer, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    // Find a memory type index that fits the properties of the buffer
    memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
    // If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to enable the appropriate flag during allocation
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
    if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        memAlloc.pNext = &allocFlagsInfo;
    }
    VK_CHECK_RESULT(vkAllocateMemory(m_logicalDevice, &memAlloc, nullptr, memory));

    // If a pointer to the buffer data has been passed, map the buffer and copy over the data
    if (data != nullptr)
    {
        void* mapped;
        VK_CHECK_RESULT(vkMapMemory(m_logicalDevice, *memory, 0, size, 0, &mapped));
        memcpy(mapped, data, size);
        // If host coherency hasn't been requested, do a manual flush to make writes visible
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
        {
            VkMappedMemoryRange mappedRange = vkinit::mappedMemoryRange();
            mappedRange.memory = *memory;
            mappedRange.offset = 0;
            mappedRange.size = size;
            vkFlushMappedMemoryRanges(m_logicalDevice, 1, &mappedRange);
        }
        vkUnmapMemory(m_logicalDevice, *memory);
    }

    // Attach the memory to the buffer object
    VK_CHECK_RESULT(vkBindBufferMemory(m_logicalDevice, *buffer, *memory, 0));

    return VK_SUCCESS;
}
VkCommandBuffer VkSandboxDevice::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void VkSandboxDevice::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);

    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &commandBuffer);
}

void VkSandboxDevice::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

void VkSandboxDevice::copyBufferToImage(
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height,
    uint32_t layerCount) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = layerCount;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );
    endSingleTimeCommands(commandBuffer);
}

void VkSandboxDevice::createImageWithInfo(
    const VkImageCreateInfo& imageInfo,
    VkMemoryPropertyFlags properties,
    VkImage& image,
    VkDeviceMemory& imageMemory) {
    if (vkCreateImage(m_logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_logicalDevice, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    if (vkBindImageMemory(m_logicalDevice, image, imageMemory, 0) != VK_SUCCESS) {
        throw std::runtime_error("failed to bind image memory!");
    }
}

void VkSandboxDevice::transitionImageLayout(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t mizLevels,
    uint32_t layerCount) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;

    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;

    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }

    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mizLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layerCount;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if ((oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) && (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if ((oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) && (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if ((oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(commandBuffer);
}

bool VkSandboxDevice::hasStencilComponent(VkFormat format) {
    return (format == VK_FORMAT_D32_SFLOAT_S8_UINT) || (format == VK_FORMAT_D24_UNORM_S8_UINT);
}


VkCommandBuffer VkSandboxDevice::createCommandBuffer(VkCommandBufferLevel level, bool begin)
{
    return createCommandBuffer(level, m_commandPool, begin);
}

VkCommandBuffer VkSandboxDevice::createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin)
{
    VkCommandBufferAllocateInfo cmdBufAllocateInfo = vkinit::commandBufferAllocateInfo(pool, level, 1);
    VkCommandBuffer cmdBuffer;
    vkAllocateCommandBuffers(m_logicalDevice, &cmdBufAllocateInfo, &cmdBuffer);
    // If requested, also start recording for the new command buffer
    if (begin)
    {
        VkCommandBufferBeginInfo cmdBufInfo = vkinit::commandBufferBeginInfo();
        vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo);
    }
    return cmdBuffer;
}
void VkSandboxDevice::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free)
{
    if (commandBuffer == VK_NULL_HANDLE)
    {
        return;
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = vkinit::submitInfo();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceInfo = vkinit::fenceCreateInfo(VK_FLAGS_NONE);
    VkFence fence;
    VK_CHECK_RESULT(vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &fence));
    // Submit to the queue
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK_RESULT(vkWaitForFences(m_logicalDevice, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
    vkDestroyFence(m_logicalDevice, fence, nullptr);
    if (free)
    {
        vkFreeCommandBuffers(m_logicalDevice, pool, 1, &commandBuffer);
    }
}

void VkSandboxDevice::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free) {
    return flushCommandBuffer(commandBuffer, queue, m_commandPool, free);
}

// 1) Query swapchain support for our chosen physical device + surface:
SwapChainSupportDetails VkSandboxDevice::querySwapchainSupport(VkSurfaceKHR surface) const {
    SwapChainSupportDetails details;

    // Capabilities:
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        m_physicalDevice, surface, &details.capabilities);

    // Surface formats:
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        m_physicalDevice, surface, &count, nullptr);
    if (count > 0) {
        details.formats.resize(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            m_physicalDevice, surface, &count,
            details.formats.data());
    }

    // Present modes:
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        m_physicalDevice, surface, &count, nullptr);
    if (count > 0) {
        details.presentModes.resize(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            m_physicalDevice, surface, &count,
            details.presentModes.data());
    }

    return details;
}

// 2) Choose the best surface format:
VkSurfaceFormatKHR VkSandboxDevice::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& available) const
{
    for (auto const& fmt : available) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return fmt;
        }
    }
    // fallback
    return available[0];
}

// 3) Choose the best present mode:
VkPresentModeKHR VkSandboxDevice::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& available) const
{
    for (auto const& mode : available) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            return mode;
    }
    // guaranteed to exist
    return VK_PRESENT_MODE_FIFO_KHR;
}

// 4) Clamp the swap extent to supported min/max:
VkExtent2D VkSandboxDevice::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR& caps,
    VkExtent2D desired) const
{
    if (caps.currentExtent.width != UINT32_MAX) {
        // special value means “we must use exactly what they say”
        return caps.currentExtent;
    }
    VkExtent2D actual = desired;
    actual.width = std::max(
        caps.minImageExtent.width,
        std::min(caps.maxImageExtent.width, actual.width));
    actual.height = std::max(
        caps.minImageExtent.height,
        std::min(caps.maxImageExtent.height, actual.height));
    return actual;
}
uint32_t VkSandboxDevice::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound) const
{
    for (uint32_t i = 0; i < m_deviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((typeBits & 1) == 1)
        {
            if ((m_deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                if (memTypeFound)
                {
                    *memTypeFound = true;
                }
                return i;
            }
        }
        typeBits >>= 1;
    }

    if (memTypeFound)
    {
        *memTypeFound = false;
        return 0;
    }
    else
    {
        throw std::runtime_error("Could not find a matching memory type");
    }
}

