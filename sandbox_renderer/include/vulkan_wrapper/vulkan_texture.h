#pragma once


#include <vulkan/vulkan.h>
#include <cstring>
#include <cassert>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <array>

#include "vk_tools/vk_init.h"
#include "vk_tools/vk_tools.h"

#include "ktx.h"
#include "ktxvulkan.h"


class VkSandboxDevice;

class VkSandboxTexture {
public:
	VkSandboxTexture() = default;
	VkSandboxTexture(VkSandboxDevice* device);
	~VkSandboxTexture();

	bool STBLoadFromFile(const std::string& filename);
	bool KTXLoadFromFile(
		const std::string& filename,
		VkFormat           format,
		VkSandboxDevice* device,
		VkQueue            copyQueue,
		VkImageUsageFlags  imageUsageFlags,
		VkImageLayout      imageLayout,
		bool               forceLinear
	);
	bool LoadCubemap(const std::array<std::string, 6>& faceFilePaths);
	void KtxLoadCubemapFromFile(
		const std::string& name,
		std::string filename,
		VkFormat format,
		VkSandboxDevice* device,
		VkQueue copyQueue,
		VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
		VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	void STBLoadCubemapFromFile(
		const std::string& filename,
		VkFormat           format,
		VkSandboxDevice* device,
		VkQueue            copyQueue,
		VkImageUsageFlags  imageUsageFlags,
		VkImageLayout      finalImageLayout,
		bool               forceLinear
	);

	void Destroy();
	void UpdateDescriptor();
	void fromBuffer(
		void* buffer,
		VkDeviceSize       bufferSize,
		VkFormat           format,
		uint32_t           texWidth,
		uint32_t           texHeight,
		VkSandboxDevice* device,
		VkQueue            copyQueue,
		VkFilter           filter = VK_FILTER_LINEAR,
		VkImageUsageFlags  imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
		VkImageLayout      imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	VkDescriptorImageInfo GetDescriptor() const { return m_descriptor; }
	VkSampler GetSampler() const { return m_sampler; }
	VkImageView GetImageView() const { return m_view; }
private:
	bool CreateImage(
		uint32_t width, uint32_t height,
		VkFormat format,
		VkImageTiling tiling,
		VkImageUsageFlags usage,
		VkMemoryPropertyFlags properties,
		uint32_t arrayLayers,
		VkImageCreateFlags flags = 0);
	void CreateImageView(
		VkFormat format,
		VkImageViewType viewType,
		uint32_t layerCount);

	void CreateSampler();
	void TransitionImageLayout(
		VkImageLayout oldLayout,
		VkImageLayout newLayout,
		uint32_t    layerCount);

	void CopyBufferToImage(VkBuffer buffer, uint32_t width, uint32_t height, uint32_t layerCount);
	VkDeviceMemory AllocateMemory(VkMemoryRequirements memRequirements, VkMemoryPropertyFlags properties);



public:
	VkSandboxDevice* m_pDevice;
	uint32_t              m_width{ 0 }, m_height{ 0 };
	VkImage m_image = VK_NULL_HANDLE;
	VkDeviceMemory m_deviceMemory = VK_NULL_HANDLE;
	VkImageView m_view = VK_NULL_HANDLE;
	VkSampler m_sampler = VK_NULL_HANDLE;

	VkImageLayout m_imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkDescriptorImageInfo m_descriptor{};

	VkFormat m_format = VK_FORMAT_UNDEFINED;
	uint32_t m_mipLevels = 1;
	uint32_t m_layerCount = 1;


	void destroy();
	ktxResult loadKTXFile(std::string filename, ktxTexture** target);

	bool m_bIsCubemap{ false };
	bool IsCubemap() const { return m_bIsCubemap; }
};
