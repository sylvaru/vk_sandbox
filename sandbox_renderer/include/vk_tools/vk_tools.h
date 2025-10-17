#pragma once

#include "vulkan/vulkan.h"
#include "vk_tools/vk_init.h"
#include "vulkan_wrapper/vulkan_device.h"



#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>


#ifndef NOMINMAX
#  define NOMINMAX
#endif
#if defined(_WIN32)
#include <windows.h>
#include <fcntl.h>
#include <io.h>

// Math
#include <corecrt.h>


#pragma warning(push)
#pragma warning(disable: _UCRT_DISABLED_WARNINGS)
_UCRT_DISABLE_CLANG_WARNINGS

// Definitions of useful mathematical constants
//
// Define _USE_MATH_DEFINES before including <math.h> to expose these macro
// definitions for common math constants.  These are placed under an #ifdef
// since these commonly-defined names are not part of the C or C++ standards
#define M_E        2.71828182845904523536   // e
#define M_LOG2E    1.44269504088896340736   // log2(e)
#define M_LOG10E   0.434294481903251827651  // log10(e)
#define M_LN2      0.693147180559945309417  // ln(2)
#define M_LN10     2.30258509299404568402   // ln(10)
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2)

_UCRT_RESTORE_CLANG_WARNINGS
#pragma warning(pop) // _UCRT_DISABLED_WARNINGS
#endif // _MATH_DEFINES_DEFINED


// Custom define for better code readability
#define VK_FLAGS_NONE 0
// Default fence timeout in nanoseconds
#define DEFAULT_FENCE_TIMEOUT 100000000000

// Check for Vulkan result and print error message if not successful
#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		std::cout << "Fatal : VkResult is \"" << tools::errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \
		assert(res == VK_SUCCESS);																		\
	}																									\
}

#include <string>

inline const std::string getAssetPath()
{
#ifdef PROJECT_ROOT_DIR
	return std::string(PROJECT_ROOT_DIR) + "/assets/";
#else
	return "./../assets/";
#endif
}

class VkSandboxDevice;

namespace tools
{



	void generateMipmaps(
		VkCommandBuffer cmdBuffer,
		VkSandboxDevice* device,
		VkImage image,
		VkFormat imageFormat,
		int32_t texWidth,
		int32_t texHeight,
		uint32_t mipLevels);


	// from: https://stackoverflow.com/a/57595105
	template <typename T, typename... Rest>
	void hashCombine(std::size_t& seed, const T& v, const Rest&... rest) {
		seed ^= std::hash<T>{}(v)+0x9e3779b9 + (seed << 6) + (seed >> 2);
		(hashCombine(seed, rest), ...);
	};


	/** @brief Setting this path chnanges the place where the samples looks for assets and shaders */
	extern std::string resourcePath;

	/** @brief Disable message boxes on fatal errors */
	extern bool errorModeSilent;

	/** @brief Returns an error code as a string */
	std::string errorString(VkResult errorCode);

	/** @brief Returns the device type as a string */
	std::string physicalDeviceTypeString(VkPhysicalDeviceType type);

	// Selected a suitable supported depth format starting with 32 bit down to 16 bit
	// Returns false if none of the depth formats in the list is supported by the device
	VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat);
	// Same as getSupportedDepthFormat but will only select formats that also have stencil
	VkBool32 getSupportedDepthStencilFormat(VkPhysicalDevice physicalDevice, VkFormat* depthStencilFormat);

	// Returns true a given format support LINEAR filtering
	VkBool32 formatIsFilterable(VkPhysicalDevice physicalDevice, VkFormat format, VkImageTiling tiling);
	// Returns true if a given format has a stencil part
	VkBool32 formatHasStencil(VkFormat format);

	// Put an image memory barrier for setting an image layout on the sub resource into the given command buffer
	void setImageLayout(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkImageSubresourceRange subresourceRange,
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	// Uses a fixed sub resource layout with first mip level and layer
	void setImageLayout(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageAspectFlags aspectMask,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

	/** @brief Insert an image memory barrier into the command buffer */
	void insertImageMemoryBarrier(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkImageSubresourceRange subresourceRange);

	// Display error message and exit on fatal error
	void exitFatal(const std::string& message, int32_t exitCode);
	void exitFatal(const std::string& message, VkResult resultCode);

	// Load a SPIR-V shader (binary)
#if defined(__ANDROID__)
	VkShaderModule loadShader(AAssetManager* assetManager, const char* fileName, VkDevice device);
#else
	VkShaderModule loadShader(const char* fileName, VkDevice device);
#endif

	/** @brief Checks if a file exists */
	bool fileExists(const std::string& filename);

	uint32_t alignedSize(uint32_t value, uint32_t alignment);
	VkDeviceSize alignedVkSize(VkDeviceSize value, VkDeviceSize alignment);



}