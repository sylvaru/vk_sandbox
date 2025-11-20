#include "vulkan_wrapper/vulkan_texture.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"

#include <spdlog/spdlog.h>
#include <stdexcept>





VkSandboxTexture::VkSandboxTexture(VkSandboxDevice* device) : m_pDevice(device) {}

VkSandboxTexture::~VkSandboxTexture()
{
	Destroy();
}

bool VkSandboxTexture::STBLoadFromFile(const std::string& filename)
{
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	if (!pixels)
	{
		throw std::runtime_error("Failed to load texture image: " + filename);
	}
	VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

	// Stage data
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	m_pDevice->createBuffer(
		imageSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer,
		stagingMemory);

	void* data;
	vkMapMemory(m_pDevice->m_logicalDevice, stagingMemory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(m_pDevice->m_logicalDevice, stagingMemory);
	stbi_image_free(pixels);

	// Create and upload to 2D image
	CreateImage(
		static_cast<uint32_t>(texWidth),
		static_cast<uint32_t>(texHeight),
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		1, // arrayLayers
		0  // flags
	);

	TransitionImageLayout(
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1 // layerCount
	);
	CopyBufferToImage(
		stagingBuffer,
		static_cast<uint32_t>(texWidth),
		static_cast<uint32_t>(texHeight),
		1 // layerCount
	);
	TransitionImageLayout(
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		1 // layerCount
	);

	vkDestroyBuffer(m_pDevice->m_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(m_pDevice->m_logicalDevice, stagingMemory, nullptr);

	// Create view and sampler for 2D
	CreateImageView(
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_VIEW_TYPE_2D,
		1 // layerCount
	);
	CreateSampler();
	UpdateDescriptor();

	return true;
}

bool VkSandboxTexture::KTXLoadFromFile(
	const std::string& filename,
	VkFormat           format,
	VkSandboxDevice* device,
	VkQueue            copyQueue,
	VkImageUsageFlags  imageUsageFlags,
	VkImageLayout      imageLayout,
	bool               forceLinear
)
{
	ktxTexture* ktxTexture;
	ktxResult result = loadKTXFile(filename, &ktxTexture);
	assert(result == KTX_SUCCESS);

	this->m_pDevice = device;
	m_width = ktxTexture->baseWidth;
	m_height = ktxTexture->baseHeight;
	m_mipLevels = ktxTexture->numLevels;

	ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
	ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(device->m_physicalDevice, format, &formatProperties);

	VkBool32 useStaging = !forceLinear;

	VkMemoryAllocateInfo memAllocInfo = vkinit::memoryAllocateInfo();
	VkMemoryRequirements memReqs;

	VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	if (useStaging)
	{
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;

		VkBufferCreateInfo bufferCreateInfo = vkinit::bufferCreateInfo();
		bufferCreateInfo.size = ktxTextureSize;
		// This buffer is used as a transfer source for the buffer copy
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VK_CHECK_RESULT(vkCreateBuffer(device->m_logicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));

		// Get memory requirements for the staging buffer (alignment, memory type bits)
		vkGetBufferMemoryRequirements(device->m_logicalDevice, stagingBuffer, &memReqs);

		memAllocInfo.allocationSize = memReqs.size;
		// Get memory type index for a host visible buffer
		memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device->m_logicalDevice, &memAllocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device->m_logicalDevice, stagingBuffer, stagingMemory, 0));

		// Copy texture data into staging buffer
		uint8_t* data;
		VK_CHECK_RESULT(vkMapMemory(device->m_logicalDevice, stagingMemory, 0, memReqs.size, 0, (void**)&data));
		memcpy(data, ktxTextureData, ktxTextureSize);
		vkUnmapMemory(device->m_logicalDevice, stagingMemory);

		// Setup buffer copy regions for each mip level
		std::vector<VkBufferImageCopy> bufferCopyRegions;

		for (uint32_t i = 0; i < m_mipLevels; i++)
		{
			ktx_size_t offset;
			KTX_error_code result = ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
			assert(result == KTX_SUCCESS);

			VkBufferImageCopy bufferCopyRegion = {};
			bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			bufferCopyRegion.imageSubresource.mipLevel = i;
			bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
			bufferCopyRegion.imageSubresource.layerCount = 1;
			bufferCopyRegion.imageExtent.width = std::max(1u, ktxTexture->baseWidth >> i);
			bufferCopyRegion.imageExtent.height = std::max(1u, ktxTexture->baseHeight >> i);
			bufferCopyRegion.imageExtent.depth = 1;
			bufferCopyRegion.bufferOffset = offset;


			bufferCopyRegions.push_back(bufferCopyRegion);
		}

		// Create optimal tiled target image
		VkImageCreateInfo imageCreateInfo = vkinit::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.mipLevels = m_mipLevels;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.extent = { m_width, m_height, 1 };
		imageCreateInfo.usage = imageUsageFlags;
		// Ensure that the TRANSFER_DST bit is set for staging
		if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
		{
			imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}
		VK_CHECK_RESULT(vkCreateImage(device->m_logicalDevice, &imageCreateInfo, nullptr, &m_image));

		vkGetImageMemoryRequirements(device->m_logicalDevice, m_image, &memReqs);

		memAllocInfo.allocationSize = memReqs.size;

		memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device->m_logicalDevice, &memAllocInfo, nullptr, &m_deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device->m_logicalDevice, m_image, m_deviceMemory, 0));

		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = m_mipLevels;
		subresourceRange.layerCount = 1;


		// Image barrier for optimal image (target)
		// Optimal image will be used as destination for the copy
		tools::setImageLayout(
			copyCmd,
			m_image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			subresourceRange);

		// Copy mip levels from staging buffer
		vkCmdCopyBufferToImage(
			copyCmd,
			stagingBuffer,
			m_image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			static_cast<uint32_t>(bufferCopyRegions.size()),
			bufferCopyRegions.data()
		);

		// Change texture image layout to shader read after all mip levels have been copied
		this->m_imageLayout = imageLayout;
		tools::setImageLayout(
			copyCmd,
			m_image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			imageLayout,
			subresourceRange);

		device->flushCommandBuffer(copyCmd, copyQueue);

		// Clean up staging resources
		vkDestroyBuffer(device->m_logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(device->m_logicalDevice, stagingMemory, nullptr);
	}
	else
	{
		// Prefer using optimal tiling, as linear tiling 
		// may support only a small set of features 
		// depending on implementation (e.g. no mip maps, only one layer, etc.)

		// Check if this support is supported for linear tiling
		assert(formatProperties.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

		VkImage mappableImage;
		VkDeviceMemory mappableMemory;

		VkImageCreateInfo imageCreateInfo = vkinit::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = { m_width, m_height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
		imageCreateInfo.usage = imageUsageFlags;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		// Load mip map level 0 to linear tiling image
		VK_CHECK_RESULT(vkCreateImage(device->m_logicalDevice, &imageCreateInfo, nullptr, &mappableImage));

		// Get memory requirements for this image 
		// like size and alignment
		vkGetImageMemoryRequirements(device->m_logicalDevice, mappableImage, &memReqs);
		// Set memory allocation size to required memory size
		memAllocInfo.allocationSize = memReqs.size;

		// Get memory type that can be mapped to host memory
		memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		// Allocate host memory
		VK_CHECK_RESULT(vkAllocateMemory(device->m_logicalDevice, &memAllocInfo, nullptr, &mappableMemory));

		// Bind allocated image for use
		VK_CHECK_RESULT(vkBindImageMemory(device->m_logicalDevice, mappableImage, mappableMemory, 0));

		// Get sub resource layout
		// Mip map count, array layer, etc.
		VkImageSubresource subRes = {};
		subRes.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subRes.mipLevel = 0;

		VkSubresourceLayout subResLayout;
		void* data;

		// Get sub resources layout 
		// Includes row pitch, size offsets, etc.
		vkGetImageSubresourceLayout(device->m_logicalDevice, mappableImage, &subRes, &subResLayout);

		// Map image memory
		VK_CHECK_RESULT(vkMapMemory(device->m_logicalDevice, mappableMemory, 0, memReqs.size, 0, &data));

		// Copy image data into memory
		memcpy(data, ktxTextureData, memReqs.size);

		vkUnmapMemory(device->m_logicalDevice, mappableMemory);

		// Linear tiled images don't need to be staged
		// and can be directly used as textures
		m_image = mappableImage;
		m_deviceMemory = mappableMemory;
		this->m_imageLayout = imageLayout;

		// Setup image memory barrier
		tools::setImageLayout(copyCmd, m_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, imageLayout);

		device->flushCommandBuffer(copyCmd, copyQueue);
	}
	ktxTexture_Destroy(ktxTexture);
	// Create sampler with anisotropic filtering
	VkSamplerCreateInfo samplerCreateInfo{};
	samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCreateInfo.minLod = 0.0f;
	samplerCreateInfo.maxLod = static_cast<float>(m_mipLevels);
	samplerCreateInfo.mipLodBias = 0.0f;
	samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;

	// Enable anisotropy if supported
	samplerCreateInfo.anisotropyEnable =
		device->m_enabledFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
	samplerCreateInfo.maxAnisotropy =
		samplerCreateInfo.anisotropyEnable
		? device->m_deviceProperties.limits.maxSamplerAnisotropy
		: 1.0f;

	samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

	VK_CHECK_RESULT(vkCreateSampler(device->m_logicalDevice, &samplerCreateInfo, nullptr, &m_sampler));

	// Create image view
	VkImageViewCreateInfo viewCreateInfo{};
	viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewCreateInfo.format = format;
	viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_mipLevels, 0, 1 };
	viewCreateInfo.image = m_image;
	VK_CHECK_RESULT(vkCreateImageView(device->m_logicalDevice, &viewCreateInfo, nullptr, &m_view));

	// Update descriptor for shader sampling
	UpdateDescriptor();

	return true;
}
void VkSandboxTexture::STBLoadCubemapFromFile(
	const std::string& filename,
	VkFormat format,
	VkSandboxDevice* device,
	VkQueue copyQueue,
	VkImageUsageFlags imageUsageFlags,
	VkImageLayout finalImageLayout,
	bool forceLinear,
	vkglTF::Model* skyboxModel)
{
	m_bIsCubemap = true;
	m_pDevice = device;

	int texWidth = 0, texHeight = 0, texChannels = 0;
	bool isHDR = stbi_is_hdr(filename.c_str());

	void* pixels = nullptr;

	if (isHDR) {
		float* data = stbi_loadf(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		if (!data) throw std::runtime_error("Failed to load HDR image: " + filename);
		pixels = data;
		texChannels = 4;
		format = VK_FORMAT_R32G32B32A32_SFLOAT;
	}
	else {
		stbi_uc* data = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		if (!data) throw std::runtime_error("Failed to load LDR image: " + filename);
		pixels = data;
		texChannels = 4;
		if (format != VK_FORMAT_R16G16B16A16_SFLOAT && format != VK_FORMAT_R32G32B32A32_SFLOAT)
			format = VK_FORMAT_R16G16B16A16_SFLOAT;	
	}

	m_width = texWidth;
	m_height = texHeight;
	m_mipLevels = static_cast<uint32_t>(floor(log2(std::max(texWidth, texHeight)))) + 1;

	VkDeviceSize componentSize = isHDR ? sizeof(float) : sizeof(stbi_uc);
	VkDeviceSize bytesPerPixel = texChannels * componentSize;
	VkDeviceSize imageSize = texWidth * texHeight * bytesPerPixel;

	// Create staging buffer for the full equirectangular texture
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	device->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		imageSize,
		&stagingBuffer,
		&stagingMemory);

	void* mapped = nullptr;
	VK_CHECK_RESULT(vkMapMemory(device->m_logicalDevice, stagingMemory, 0, imageSize, 0, &mapped));
	memcpy(mapped, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(device->m_logicalDevice, stagingMemory);
	stbi_image_free(pixels);

	// Detect if this is an equirectangular (2:1 aspect) map
	bool srcIsEquirectangular = (texWidth == texHeight * 2);

	// Create the image (2D if equirectangular, cube if 6 faces are provided)
	VkImageCreateInfo imageCI{};
	imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = format;
	imageCI.extent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 };
	imageCI.mipLevels = m_mipLevels;
	imageCI.arrayLayers = srcIsEquirectangular ? 1u : 6u;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = imageUsageFlags | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	imageCI.flags = srcIsEquirectangular ? 0 : VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VK_CHECK_RESULT(vkCreateImage(device->m_logicalDevice, &imageCI, nullptr, &m_image));

	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(device->m_logicalDevice, m_image, &memReqs);

	VkMemoryAllocateInfo memAllocInfo{};
	memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllocInfo.allocationSize = memReqs.size;
	memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(device->m_logicalDevice, &memAllocInfo, nullptr, &m_deviceMemory));
	VK_CHECK_RESULT(vkBindImageMemory(device->m_logicalDevice, m_image, m_deviceMemory, 0));

	// Begin copy commands
	VkCommandBuffer cmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	VkImageSubresourceRange subresourceRange{};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0;
	subresourceRange.levelCount = m_mipLevels;
	subresourceRange.baseArrayLayer = 0;
	subresourceRange.layerCount = srcIsEquirectangular ? 1u : 6u;

	tools::setImageLayout(cmd, m_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = srcIsEquirectangular ? 1u : 6u;
	region.imageExtent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 };

	vkCmdCopyBufferToImage(cmd, stagingBuffer, m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	// Generate mipmaps (safe for 2D as well)
	tools::generateMipmaps(cmd, device, m_image, format, texWidth, texHeight, m_mipLevels);

	// Final layout transition
	tools::setImageLayout(cmd, m_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, finalImageLayout, subresourceRange);

	device->flushCommandBuffer(cmd, copyQueue, true);

	vkDestroyBuffer(device->m_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(device->m_logicalDevice, stagingMemory, nullptr);

	// Create sampler
	VkSamplerCreateInfo samplerInfo = vkinit::samplerCreateInfo();
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = samplerInfo.addressModeU;
	samplerInfo.addressModeW = samplerInfo.addressModeU;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = static_cast<float>(m_mipLevels);
	samplerInfo.maxAnisotropy = device->m_enabledFeatures.samplerAnisotropy
		? device->m_deviceProperties.limits.maxSamplerAnisotropy
		: 1.0f;
	samplerInfo.anisotropyEnable = device->m_enabledFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
	VK_CHECK_RESULT(vkCreateSampler(device->m_logicalDevice, &samplerInfo, nullptr, &m_sampler));

	// Image view
	VkImageViewCreateInfo viewCI{};
	viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCI.image = m_image;
	viewCI.format = format;
	viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewCI.subresourceRange.baseMipLevel = 0;
	viewCI.subresourceRange.levelCount = m_mipLevels;
	viewCI.subresourceRange.baseArrayLayer = 0;
	viewCI.subresourceRange.layerCount = srcIsEquirectangular ? 1u : 6u;
	viewCI.viewType = srcIsEquirectangular ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_CUBE;

	VK_CHECK_RESULT(vkCreateImageView(device->m_logicalDevice, &viewCI, nullptr, &m_view));

	// Update descriptor
	UpdateDescriptor();

	if (srcIsEquirectangular)
	{
		spdlog::info("[VkSandboxTexture] '{}' detected as equirectangular HDR ({}x{}). Converting to cubemap...",
			filename, texWidth, texHeight);

		if (!skyboxModel) {
			spdlog::error("[VkSandboxTexture] Equirectangular detected but no cube model provided for conversion!");
		}
		else {
			try {
				ConvertEquirectangularToCubemap(copyQueue, skyboxModel);
			}
			catch (const std::exception& e) {
				spdlog::error("[VkSandboxTexture] Equirectangular-to-cubemap conversion failed: {}", e.what());
			}
		}
	}
}


void VkSandboxTexture::ConvertEquirectangularToCubemap(VkQueue copyQueue, vkglTF::Model* skyboxModel)
{
	if (!m_pDevice)
		throw std::runtime_error("ConvertEquirectangularToCubemap: missing device");
	if (!skyboxModel)
		throw std::runtime_error("ConvertEquirectangularToCubemap: missing skybox model");

	const VkFormat targetFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
	const uint32_t dim = 512;
	const uint32_t numMips = 1;

	// Save the original source descriptor and view
	VkDescriptorImageInfo srcDescriptor = m_descriptor;
	VkImage srcImage = m_image;
	VkImageView srcView = m_view;
	VkFormat srcFormat = m_format;
	uint32_t srcMipLevels = m_mipLevels;
	uint32_t srcLayerCount = m_layerCount;

	// Create target cubemap
	VkImage cubeImage = VK_NULL_HANDLE;
	VkDeviceMemory cubeMemory = VK_NULL_HANDLE;
	VkImageView cubeView = VK_NULL_HANDLE;
	VkSampler cubeSampler = VK_NULL_HANDLE;
	VkDescriptorImageInfo cubeDescriptor{};

	// create image
	VkImageCreateInfo imageCI = vkinit::imageCreateInfo();
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = targetFormat;
	imageCI.extent = { dim, dim, 1 };
	imageCI.mipLevels = numMips;
	imageCI.arrayLayers = 6;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VK_CHECK_RESULT(vkCreateImage(m_pDevice->device(), &imageCI, nullptr, &cubeImage));

	VkMemoryRequirements memReqs{};
	vkGetImageMemoryRequirements(m_pDevice->device(), cubeImage, &memReqs);
	VkMemoryAllocateInfo cubeAlloc = vkinit::memoryAllocateInfo();
	cubeAlloc.allocationSize = memReqs.size;
	cubeAlloc.memoryTypeIndex = m_pDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(m_pDevice->device(), &cubeAlloc, nullptr, &cubeMemory));
	VK_CHECK_RESULT(vkBindImageMemory(m_pDevice->device(), cubeImage, cubeMemory, 0));

	// Cube view & sampler
	{
		VkImageViewCreateInfo viewCI = vkinit::imageViewCreateInfo();
		viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		viewCI.format = targetFormat;
		viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewCI.subresourceRange.baseMipLevel = 0;
		viewCI.subresourceRange.levelCount = numMips;
		viewCI.subresourceRange.baseArrayLayer = 0;
		viewCI.subresourceRange.layerCount = 6;
		viewCI.image = cubeImage;
		VK_CHECK_RESULT(vkCreateImageView(m_pDevice->device(), &viewCI, nullptr, &cubeView));

		VkSamplerCreateInfo sampCI = vkinit::samplerCreateInfo();
		sampCI.magFilter = VK_FILTER_LINEAR;
		sampCI.minFilter = VK_FILTER_LINEAR;
		sampCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampCI.minLod = 0.0f;
		sampCI.maxLod = 0.0f;
		sampCI.anisotropyEnable = m_pDevice->m_enabledFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
		sampCI.maxAnisotropy = m_pDevice->m_enabledFeatures.samplerAnisotropy ? m_pDevice->m_deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
		VK_CHECK_RESULT(vkCreateSampler(m_pDevice->device(), &sampCI, nullptr, &cubeSampler));

		cubeDescriptor = vkinit::descriptorImageInfo(cubeSampler, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	// Create offscreen 2D render target
	VkImage offImage = VK_NULL_HANDLE;
	VkDeviceMemory offMemory = VK_NULL_HANDLE;
	VkImageView offView = VK_NULL_HANDLE;
	{
		VkImageCreateInfo offCI = vkinit::imageCreateInfo();
		offCI.imageType = VK_IMAGE_TYPE_2D;
		offCI.format = targetFormat;
		offCI.extent = { dim, dim, 1 };
		offCI.mipLevels = 1;
		offCI.arrayLayers = 1;
		offCI.samples = VK_SAMPLE_COUNT_1_BIT;
		offCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		offCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		offCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VK_CHECK_RESULT(vkCreateImage(m_pDevice->device(), &offCI, nullptr, &offImage));
		vkGetImageMemoryRequirements(m_pDevice->device(), offImage, &memReqs);
		VkMemoryAllocateInfo offAlloc = vkinit::memoryAllocateInfo();
		offAlloc.allocationSize = memReqs.size;
		offAlloc.memoryTypeIndex = m_pDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(m_pDevice->device(), &offAlloc, nullptr, &offMemory));
		VK_CHECK_RESULT(vkBindImageMemory(m_pDevice->device(), offImage, offMemory, 0));

		VkImageViewCreateInfo offViewCI = vkinit::imageViewCreateInfo();
		offViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		offViewCI.format = targetFormat;
		offViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		offViewCI.subresourceRange.baseMipLevel = 0;
		offViewCI.subresourceRange.levelCount = 1;
		offViewCI.subresourceRange.baseArrayLayer = 0;
		offViewCI.subresourceRange.layerCount = 1;
		offViewCI.image = offImage;
		VK_CHECK_RESULT(vkCreateImageView(m_pDevice->device(), &offViewCI, nullptr, &offView));
	}

	// Renderpass & framebuffer
	VkAttachmentDescription attDesc{};
	attDesc.format = targetFormat;
	attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
	attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorRef;

	VkRenderPassCreateInfo rpCI = vkinit::renderPassCreateInfo();
	rpCI.attachmentCount = 1;
	rpCI.pAttachments = &attDesc;
	rpCI.subpassCount = 1;
	rpCI.pSubpasses = &subpass;

	VkRenderPass renderPass = VK_NULL_HANDLE;
	VK_CHECK_RESULT(vkCreateRenderPass(m_pDevice->device(), &rpCI, nullptr, &renderPass));

	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	{
		VkFramebufferCreateInfo fbCI = vkinit::framebufferCreateInfo();
		fbCI.renderPass = renderPass;
		fbCI.attachmentCount = 1;
		fbCI.pAttachments = &offView;
		fbCI.width = dim;
		fbCI.height = dim;
		fbCI.layers = 1;
		VK_CHECK_RESULT(vkCreateFramebuffer(m_pDevice->device(), &fbCI, nullptr, &framebuffer));
	}

	// Descriptor for source equirectangular texture
	// IMPORTANT: use the original source descriptor (srcDescriptor) so the shader sees a 2D view
	// Make sure the source is in SHADER_READ_ONLY layout before binding and sampling
	VkCommandBuffer layoutCmd = m_pDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
	{
		VkImageSubresourceRange srcRange{};
		srcRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		srcRange.baseMipLevel = 0;
		srcRange.levelCount = srcMipLevels;
		srcRange.baseArrayLayer = 0;
		srcRange.layerCount = srcLayerCount;

		tools::setImageLayout(layoutCmd, srcImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, srcRange);
	}
	m_pDevice->flushCommandBuffer(layoutCmd, copyQueue, true);

	// Create descriptor set layout / pool / set that expects sampler2D
	VkDescriptorSetLayout srcDescLayout = VK_NULL_HANDLE;
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			vkinit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0)
		};
		VkDescriptorSetLayoutCreateInfo dslCI = vkinit::descriptorSetLayoutCreateInfo(bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_pDevice->device(), &dslCI, nullptr, &srcDescLayout));
	}

	VkDescriptorPool srcDescriptorPool = VK_NULL_HANDLE;
	VkDescriptorSet srcDescriptorSet = VK_NULL_HANDLE;
	{
		VkDescriptorPoolSize poolSize = vkinit::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1);
		VkDescriptorPoolCreateInfo poolCI = vkinit::descriptorPoolCreateInfo(1, &poolSize, 1);
		VK_CHECK_RESULT(vkCreateDescriptorPool(m_pDevice->device(), &poolCI, nullptr, &srcDescriptorPool));

		VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorSetAllocateInfo(srcDescriptorPool, &srcDescLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(m_pDevice->device(), &allocInfo, &srcDescriptorSet));

		VkWriteDescriptorSet write = vkinit::writeDescriptorSet(srcDescriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &srcDescriptor);
		vkUpdateDescriptorSets(m_pDevice->device(), 1, &write, 0, nullptr);
	}

	// Pipeline layout / pipeline (push constant mvp)
	struct PushBlock { 
		glm::mat4 mvp; 
	};
	VkPushConstantRange pushRange = vkinit::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(PushBlock), 0);

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	{
		VkPipelineLayoutCreateInfo plCI = vkinit::pipelineLayoutCreateInfo(&srcDescLayout, 1);
		plCI.pushConstantRangeCount = 1;
		plCI.pPushConstantRanges = &pushRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(m_pDevice->device(), &plCI, nullptr, &pipelineLayout));
	}

	PipelineConfigInfo cfg{};
	VkSandboxPipeline::defaultPipelineConfigInfo(cfg);
	cfg.renderPass = renderPass;
	cfg.pipelineLayout = pipelineLayout;
	cfg.descriptorSetLayouts = { srcDescLayout };
	cfg.pushConstantRanges = { pushRange };
	cfg.bindingDescriptions = { vkinit::vertexInputBindingDescription(0, sizeof(vkglTF::Vertex), VK_VERTEX_INPUT_RATE_VERTEX) };
	cfg.attributeDescriptions = { 
	vkinit::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, pos)),
	vkinit::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(vkglTF::Vertex, normal))
	};

	std::string vert = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/cubemap.vert.spv";
	std::string frag = std::string(PROJECT_ROOT_DIR) + "/res/shaders/spirV/equirect_to_cube.frag.spv";
	VkSandboxPipeline pipeline{ *m_pDevice, vert, frag, cfg };

	VkCommandBuffer cmdBuf = m_pDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	// Transition cube image to TRANSFER_DST for copies
	VkImageSubresourceRange cubeRange{};
	cubeRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	cubeRange.baseMipLevel = 0;
	cubeRange.levelCount = numMips;
	cubeRange.baseArrayLayer = 0;
	cubeRange.layerCount = 6;
	tools::setImageLayout(cmdBuf, cubeImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, cubeRange);

	glm::mat4 captureProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 512.0f);
	captureProj[1][1] *= -1.0f; // Vulkan requires Y flip


	std::vector<glm::mat4> matrices = {
		// +X
		glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
		// -X
		glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
		// +Y (TOP)
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
		// -Y (BOTTOM)
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
		// +Z
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
		// -Z
		glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
	};

	VkViewport vp = vkinit::viewport((float)dim, (float)dim, 0.0f, 1.0f);
	VkRect2D sc = vkinit::rect2D(dim, dim, 0, 0);
	vkCmdSetViewport(cmdBuf, 0, 1, &vp);
	vkCmdSetScissor(cmdBuf, 0, 1, &sc);

	for (uint32_t face = 0; face < 6; ++face) {
		VkClearValue clearColor{};
		clearColor.color.float32[0] = 0.0f;
		clearColor.color.float32[1] = 0.0f;
		clearColor.color.float32[2] = 0.0f;
		clearColor.color.float32[3] = 1.0f;

		VkRenderPassBeginInfo rpBI = vkinit::renderPassBeginInfo();
		rpBI.renderPass = renderPass;
		rpBI.framebuffer = framebuffer;
		rpBI.renderArea.extent = { dim, dim };
		rpBI.clearValueCount = 1;
		rpBI.pClearValues = &clearColor;
		vkCmdBeginRenderPass(cmdBuf, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

		PushBlock push{};
		push.mvp = captureProj * matrices[face];

		vkCmdPushConstants(cmdBuf,
			pipeline.getPipelineLayout(),
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0,
			sizeof(PushBlock),
			&push);

		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getPipelineLayout(), 0, 1, &srcDescriptorSet, 0, nullptr);
	
		pipeline.bind(cmdBuf);
		skyboxModel->bind(cmdBuf);
		skyboxModel->gltfDraw(cmdBuf);

		vkCmdEndRenderPass(cmdBuf);

		// copy offscreen -> cube face
		tools::setImageLayout(cmdBuf, offImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

		VkImageCopy region{};
		region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.srcSubresource.mipLevel = 0;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.layerCount = 1;
		region.srcOffset = { 0, 0, 0 };

		region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.dstSubresource.mipLevel = 0;
		region.dstSubresource.baseArrayLayer = face;
		region.dstSubresource.layerCount = 1;
		region.dstOffset = { 0, 0, 0 };

		region.extent = { dim, dim, 1 };

		vkCmdCopyImage(
			cmdBuf,
			offImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			cubeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &region);

		// restore offscreen layout for next render
		tools::setImageLayout(cmdBuf, offImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}

	// transition cube image -> shader read
	tools::setImageLayout(cmdBuf, cubeImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cubeRange);

	// submit and wait
	m_pDevice->flushCommandBuffer(cmdBuf, copyQueue, true);

	// Replace this texture's GPU data with the new cubemap
	// destroy old resources that belong to this object (but do NOT destroy the source image/view that are managed externally if you rely on them elsewhere)
	if (m_view != VK_NULL_HANDLE) vkDestroyImageView(m_pDevice->device(), m_view, nullptr);
	if (m_image != VK_NULL_HANDLE) vkDestroyImage(m_pDevice->device(), m_image, nullptr);
	if (m_deviceMemory != VK_NULL_HANDLE) vkFreeMemory(m_pDevice->device(), m_deviceMemory, nullptr);
	if (m_sampler != VK_NULL_HANDLE) vkDestroySampler(m_pDevice->device(), m_sampler, nullptr);

	// adopt cube resources
	m_image = cubeImage;
	m_deviceMemory = cubeMemory;
	m_view = cubeView;
	m_sampler = cubeSampler;
	m_descriptor = cubeDescriptor;
	m_format = targetFormat;
	m_width = dim;
	m_height = dim;
	m_mipLevels = numMips;
	m_layerCount = 6;
	m_bIsCubemap = true;
	m_imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// cleanup temporaries used during conversion
	vkDestroyImageView(m_pDevice->device(), offView, nullptr);
	vkDestroyImage(m_pDevice->device(), offImage, nullptr);
	vkFreeMemory(m_pDevice->device(), offMemory, nullptr);

	// destroy descriptor pool & layout used for source sampling
	vkDestroyDescriptorPool(m_pDevice->device(), srcDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(m_pDevice->device(), srcDescLayout, nullptr);

	// destroy renderpass/framebuffer/pipeline resources we created (but DO NOT destroy pipeline.getPipelineLayout() - pipeline wrapper manages that)
	vkDestroyFramebuffer(m_pDevice->device(), framebuffer, nullptr);
	vkDestroyRenderPass(m_pDevice->device(), renderPass, nullptr);

	// pipeline destructor will cleanup pipeline; destroy pipelineLayout we created only if still valid and not owned by pipeline
	// (some wrappers take ownership; to be safe we avoid destroying it here)

	spdlog::info("[VkSandboxTexture] Equirectangular -> cubemap conversion done ({}x{})", dim, dim);
}




// Loads a cubemap from a single KTX file
void VkSandboxTexture::KtxLoadCubemapFromFile(const std::string& name, std::string filename, VkFormat format, VkSandboxDevice* device, VkQueue copyQueue, VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout)
{
	m_bIsCubemap = true;

	ktxTexture* ktxTexture;
	ktxResult result = loadKTXFile(filename, &ktxTexture);
	assert(result == KTX_SUCCESS);

	this->m_pDevice = device;
	m_width = ktxTexture->baseWidth;
	m_height = ktxTexture->baseHeight;
	m_mipLevels = ktxTexture->numLevels;

	ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
	ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

	VkMemoryAllocateInfo memAllocInfo = vkinit::memoryAllocateInfo();
	VkMemoryRequirements memReqs;

	// Create a host-visible staging buffer that contains the raw image data
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;

	VkBufferCreateInfo bufferCreateInfo = vkinit::bufferCreateInfo();
	bufferCreateInfo.size = ktxTextureSize;
	// This buffer is used as a transfer source for the buffer copy
	bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VK_CHECK_RESULT(vkCreateBuffer(device->m_logicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));

	// Get memory requirements for the staging buffer (alignment, memory type bits)
	vkGetBufferMemoryRequirements(device->m_logicalDevice, stagingBuffer, &memReqs);

	memAllocInfo.allocationSize = memReqs.size;
	// Get memory type index for a host visible buffer
	memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	VK_CHECK_RESULT(vkAllocateMemory(device->m_logicalDevice, &memAllocInfo, nullptr, &stagingMemory));
	VK_CHECK_RESULT(vkBindBufferMemory(device->m_logicalDevice, stagingBuffer, stagingMemory, 0));

	// Copy texture data into staging buffer
	uint8_t* data;
	VK_CHECK_RESULT(vkMapMemory(device->m_logicalDevice, stagingMemory, 0, memReqs.size, 0, (void**)&data));
	memcpy(data, ktxTextureData, ktxTextureSize);
	vkUnmapMemory(device->m_logicalDevice, stagingMemory);

	// Setup buffer copy regions for each face including all of its mip levels
	std::vector<VkBufferImageCopy> bufferCopyRegions;

	for (uint32_t face = 0; face < 6; face++)
	{
		for (uint32_t level = 0; level < m_mipLevels; level++)
		{
			ktx_size_t offset;
			KTX_error_code result = ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
			assert(result == KTX_SUCCESS);

			VkBufferImageCopy bufferCopyRegion = {};
			bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			bufferCopyRegion.imageSubresource.mipLevel = level;
			bufferCopyRegion.imageSubresource.baseArrayLayer = face;
			bufferCopyRegion.imageSubresource.layerCount = 1;
			bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> level;
			bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> level;
			bufferCopyRegion.imageExtent.depth = 1;
			bufferCopyRegion.bufferOffset = offset & ~0xF;

			bufferCopyRegions.push_back(bufferCopyRegion);
		}
	}

	// Create optimal tiled target image
	VkImageCreateInfo imageCreateInfo = vkinit::imageCreateInfo();
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.format = format;
	imageCreateInfo.mipLevels = m_mipLevels;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.extent = { m_width, m_height, 1 };
	imageCreateInfo.usage = imageUsageFlags;
	// Ensure that the TRANSFER_DST bit is set for staging
	if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
	{
		imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}
	// Cube faces count as array layers in Vulkan
	imageCreateInfo.arrayLayers = 6;
	// This flag is required for cube map images
	imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;


	VK_CHECK_RESULT(vkCreateImage(device->m_logicalDevice, &imageCreateInfo, nullptr, &m_image));

	vkGetImageMemoryRequirements(device->m_logicalDevice, m_image, &memReqs);

	memAllocInfo.allocationSize = memReqs.size;
	memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VK_CHECK_RESULT(vkAllocateMemory(device->m_logicalDevice, &memAllocInfo, nullptr, &m_deviceMemory));
	VK_CHECK_RESULT(vkBindImageMemory(device->m_logicalDevice, m_image, m_deviceMemory, 0));

	// Use a separate command buffer for texture loading
	VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	// Image barrier for optimal image (target)
	// Set initial layout for all array layers (faces) of the optimal (target) tiled texture
	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0;
	subresourceRange.levelCount = m_mipLevels;
	subresourceRange.layerCount = 6;

	tools::setImageLayout(
		copyCmd,
		m_image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		subresourceRange);

	// Copy the cube map faces from the staging buffer to the optimal tiled image
	vkCmdCopyBufferToImage(
		copyCmd,
		stagingBuffer,
		m_image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		static_cast<uint32_t>(bufferCopyRegions.size()),
		bufferCopyRegions.data());

	// Change texture image layout to shader read after all faces have been copied
	this->m_imageLayout = imageLayout;
	tools::setImageLayout(
		copyCmd,
		m_image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		imageLayout,
		subresourceRange);

	device->flushCommandBuffer(copyCmd, copyQueue);

	// Create sampler
	VkSamplerCreateInfo samplerCreateInfo = vkinit::samplerCreateInfo();
	samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
	samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
	samplerCreateInfo.mipLodBias = 0.0f;
	samplerCreateInfo.maxAnisotropy = device->m_enabledFeatures.samplerAnisotropy ? device->m_deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
	samplerCreateInfo.anisotropyEnable = device->m_enabledFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
	samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
	samplerCreateInfo.minLod = 0.0f;
	samplerCreateInfo.maxLod = (float)m_mipLevels;
	samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	VK_CHECK_RESULT(vkCreateSampler(device->m_logicalDevice, &samplerCreateInfo, nullptr, &m_sampler));

	// Create image view
	VkImageViewCreateInfo viewCreateInfo = vkinit::imageViewCreateInfo();
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	viewCreateInfo.format = format;
	viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewCreateInfo.subresourceRange.baseMipLevel = 0;
	viewCreateInfo.subresourceRange.levelCount = m_mipLevels;
	viewCreateInfo.subresourceRange.baseArrayLayer = 0;
	viewCreateInfo.subresourceRange.layerCount = 6;
	viewCreateInfo.image = m_image;
	VK_CHECK_RESULT(vkCreateImageView(device->m_logicalDevice, &viewCreateInfo, nullptr, &m_view));

	// Clean up staging resources
	ktxTexture_Destroy(ktxTexture);
	vkDestroyBuffer(device->m_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(device->m_logicalDevice, stagingMemory, nullptr);

	// Update descriptor image info member that can be used for setting up descriptor sets
	UpdateDescriptor();
}
bool VkSandboxTexture::LoadCubemap(const std::array<std::string, 6>& faces) {
	m_bIsCubemap = true;

	int w, h, c;
	std::vector<stbi_uc*> images(6);
	for (int i = 0; i < 6; i++) {
		images[i] = stbi_load(faces[i].c_str(), &w, &h, &c, STBI_rgb_alpha);
		if (!images[i]) {
			throw std::runtime_error("Failed to load skybox face: " + faces[i]);
		}
	}
	VkDeviceSize layerSize = w * h * 4;
	VkDeviceSize totalSize = layerSize * 6;

	// create a single staging buffer large enough for 6 faces
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	m_pDevice->createBuffer(totalSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer, stagingMemory);

	// copy all 6 faces into the staging buffer, one after another
	void* data;
	vkMapMemory(m_pDevice->m_logicalDevice, stagingMemory, 0, totalSize, 0, &data);
	for (int i = 0; i < 6; i++) {
		memcpy((char*)data + layerSize * i, images[i], layerSize);
		stbi_image_free(images[i]);
	}
	vkUnmapMemory(m_pDevice->m_logicalDevice, stagingMemory);

	// Create the cube image: 6 layers, CUBE_COMPATIBLE flag
	CreateImage(w, h,
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		6,                            // arrayLayers
		VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

	// Transition and copy each layer
	TransitionImageLayout(
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		/*layerCount=*/6);
	CopyBufferToImage(
		stagingBuffer, w, h,
		/*layerCount=*/6);
	TransitionImageLayout(
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		/*layerCount=*/6);

	vkDestroyBuffer(m_pDevice->m_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(m_pDevice->m_logicalDevice, stagingMemory, nullptr);

	// Create a CUBE view
	CreateImageView(
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_VIEW_TYPE_CUBE,
		/*layerCount=*/6);

	CreateSampler();
	UpdateDescriptor();
	return true;
}

void VkSandboxTexture::Destroy()
{
	if (m_sampler) vkDestroySampler(m_pDevice->m_logicalDevice, m_sampler, nullptr);
	if (m_view) vkDestroyImageView(m_pDevice->m_logicalDevice, m_view, nullptr);
	if (m_image) vkDestroyImage(m_pDevice->m_logicalDevice, m_image, nullptr);
	if (m_deviceMemory) vkFreeMemory(m_pDevice->m_logicalDevice, m_deviceMemory, nullptr);
}

void VkSandboxTexture::UpdateDescriptor()
{
	m_descriptor.sampler = m_sampler;
	m_descriptor.imageView = m_view;
	m_descriptor.imageLayout = m_imageLayout;
}

bool VkSandboxTexture::CreateImage(
	uint32_t width, uint32_t height,
	VkFormat format,
	VkImageTiling tiling,
	VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties,
	uint32_t arrayLayers,
	VkImageCreateFlags flags)
{
	VkImageCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.flags = flags;
	info.imageType = VK_IMAGE_TYPE_2D;
	info.extent = { width, height, 1 };
	info.mipLevels = 1;
	info.arrayLayers = arrayLayers;
	info.format = format;
	info.tiling = tiling;
	info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	info.usage = usage;
	info.samples = VK_SAMPLE_COUNT_1_BIT;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateImage(m_pDevice->m_logicalDevice, &info, nullptr, &m_image) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
	VkMemoryRequirements memReq;
	vkGetImageMemoryRequirements(m_pDevice->m_logicalDevice, m_image, &memReq);
	m_deviceMemory = AllocateMemory(memReq, properties);
	vkBindImageMemory(m_pDevice->m_logicalDevice, m_image, m_deviceMemory, 0);
	return true;
}

VkDeviceMemory VkSandboxTexture::AllocateMemory(VkMemoryRequirements memRequirements, VkMemoryPropertyFlags properties)
{
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = m_pDevice->findMemoryType(memRequirements.memoryTypeBits, properties);

	VkDeviceMemory memory;
	if (vkAllocateMemory(m_pDevice->m_logicalDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate image memory!");
	}

	return memory;
}


void VkSandboxTexture::destroy()
{
}

ktxResult VkSandboxTexture::loadKTXFile(std::string filename, ktxTexture** target)
{
	ktxResult result = KTX_SUCCESS;
	if (!tools::fileExists(filename)) {
		throw std::runtime_error("KTX file not found: " + filename);
	}
	result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, target);

	return result;
}

void VkSandboxTexture::CreateImageView(
	VkFormat format,
	VkImageViewType viewType,
	uint32_t layerCount)
{
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_image;
	viewInfo.viewType = viewType;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = layerCount;

	if (vkCreateImageView(m_pDevice->m_logicalDevice, &viewInfo, nullptr, &m_view) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image view!");
	}
}

void VkSandboxTexture::CreateSampler()
{
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = 16.0f;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	if (vkCreateSampler(m_pDevice->m_logicalDevice, &samplerInfo, nullptr, &m_sampler) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create texture sampler!");
	}
}

void VkSandboxTexture::TransitionImageLayout(
	VkImageLayout oldLayout,
	VkImageLayout newLayout,
	uint32_t    layerCount)
{
	VkCommandBuffer cmd = m_pDevice->beginSingleTimeCommands();
	m_pDevice->transitionImageLayout(
		m_image, m_format,
		oldLayout, newLayout,
		m_mipLevels, layerCount);
	m_pDevice->endSingleTimeCommands(cmd);
}

void VkSandboxTexture::CopyBufferToImage(
	VkBuffer buffer,
	uint32_t width,
	uint32_t height,
	uint32_t layerCount)
{
	m_pDevice->copyBufferToImage(buffer,
		m_image,
		width, height,
		layerCount);
}
void VkSandboxTexture::fromBuffer(void* buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight, VkSandboxDevice* pdevice, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout)
{
	assert(buffer);

	this->m_pDevice = pdevice;
	m_width = texWidth;
	m_height = texHeight;
	m_mipLevels = 1;


	VkMemoryAllocateInfo memAllocInfo = vkinit::memoryAllocateInfo();
	VkMemoryRequirements memReqs;

	// Use a separate command buffer for texture loading
	VkCommandBuffer copyCmd = pdevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	// Create a host-visible staging buffer that contains the raw image data
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;

	VkBufferCreateInfo bufferCreateInfo = vkinit::bufferCreateInfo();
	bufferCreateInfo.size = bufferSize;
	// This buffer is used as a transfer source for the buffer copy
	bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	vkCreateBuffer(pdevice->m_logicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer);

	// Get memory requirements for the staging buffer (alignment, memory type bits)
	vkGetBufferMemoryRequirements(pdevice->m_logicalDevice, stagingBuffer, &memReqs);

	memAllocInfo.allocationSize = memReqs.size;
	// Get memory type index for a host visible buffer
	memAllocInfo.memoryTypeIndex = pdevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	vkAllocateMemory(pdevice->m_logicalDevice, &memAllocInfo, nullptr, &stagingMemory);
	vkBindBufferMemory(pdevice->m_logicalDevice, stagingBuffer, stagingMemory, 0);

	// Copy texture data into staging buffer
	uint8_t* data;
	vkMapMemory(pdevice->m_logicalDevice, stagingMemory, 0, memReqs.size, 0, (void**)&data);
	memcpy(data, buffer, bufferSize);
	vkUnmapMemory(pdevice->m_logicalDevice, stagingMemory);

	VkBufferImageCopy bufferCopyRegion = {};
	bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	bufferCopyRegion.imageSubresource.mipLevel = 0;
	bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
	bufferCopyRegion.imageSubresource.layerCount = 1;
	bufferCopyRegion.imageExtent.width = m_width;
	bufferCopyRegion.imageExtent.height = m_height;
	bufferCopyRegion.imageExtent.depth = 1;
	bufferCopyRegion.bufferOffset = 0;

	// Create optimal tiled target image
	VkImageCreateInfo imageCreateInfo = vkinit::imageCreateInfo();
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.format = format;
	imageCreateInfo.mipLevels = m_mipLevels;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.extent = { m_width, m_height, 1 };
	imageCreateInfo.usage = imageUsageFlags;
	// Ensure that the TRANSFER_DST bit is set for staging
	if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
	{
		imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}
	vkCreateImage(pdevice->m_logicalDevice, &imageCreateInfo, nullptr, &m_image);

	vkGetImageMemoryRequirements(pdevice->m_logicalDevice, m_image, &memReqs);

	memAllocInfo.allocationSize = memReqs.size;

	memAllocInfo.memoryTypeIndex = pdevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(pdevice->m_logicalDevice, &memAllocInfo, nullptr, &m_deviceMemory);
	vkBindImageMemory(pdevice->m_logicalDevice, m_image, m_deviceMemory, 0);

	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	subresourceRange.baseMipLevel = 0;
	subresourceRange.levelCount = m_mipLevels;
	subresourceRange.layerCount = 1;

	// Image barrier for optimal image (target)
	// Optimal image will be used as destination for the copy
	tools::setImageLayout(
		copyCmd,
		m_image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		subresourceRange);

	// Copy mip levels from staging buffer
	vkCmdCopyBufferToImage(
		copyCmd,
		stagingBuffer,
		m_image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&bufferCopyRegion
	);

	// Change texture image layout to shader read after all mip levels have been copied
	this->m_imageLayout = imageLayout;
	tools::setImageLayout(
		copyCmd,
		m_image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		imageLayout,
		subresourceRange);

	//device->flushCommandBuffer(copyCmd, copyQueue);


	// Clean up staging resources
	vkDestroyBuffer(pdevice->m_logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(pdevice->m_logicalDevice, stagingMemory, nullptr);

	// Create sampler
	VkSamplerCreateInfo samplerCreateInfo = {};
	samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCreateInfo.magFilter = filter;
	samplerCreateInfo.minFilter = filter;
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCreateInfo.mipLodBias = 0.0f;
	samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
	samplerCreateInfo.minLod = 0.0f;
	samplerCreateInfo.maxLod = 0.0f;
	samplerCreateInfo.maxAnisotropy = 1.0f;
	vkCreateSampler(pdevice->m_logicalDevice, &samplerCreateInfo, nullptr, &m_sampler);

	// Create image view
	VkImageViewCreateInfo viewCreateInfo = {};
	viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCreateInfo.pNext = NULL;
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewCreateInfo.format = format;
	viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	viewCreateInfo.subresourceRange.levelCount = 1;
	viewCreateInfo.image = m_image;
	vkCreateImageView(pdevice->m_logicalDevice, &viewCreateInfo, nullptr, &m_view);

	// Update descriptor image info member that can be used for setting up descriptor sets
	UpdateDescriptor();
}