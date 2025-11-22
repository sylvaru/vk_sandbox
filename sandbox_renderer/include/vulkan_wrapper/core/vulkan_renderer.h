// vulkan_renderer.h
#pragma once
#include <memory>
#include "interfaces/renderer_i.h"
#include "interfaces/render_system_i.h"
#include "interfaces/scene_i.h"
#include "interfaces/asset_provider_i.h"

#include "window.h"

#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_swapchain.h"
#include "vulkan_wrapper/vulkan_descriptor.h"
#include "vulkan_wrapper/vulkan_buffer.h"


#include <vector>
#include <array>
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"


class GltfRenderSystem;
class PointLightRenderSystem;
class SceneRenderSystem;
class SkyboxRenderSystem;

class VkSandboxRenderer : public ISandboxRenderer
{
public:

	static constexpr size_t FrameCount =
		VkSandboxSwapchain::MAX_FRAMES_IN_FLIGHT;


	VkSandboxRenderer(VkSandboxDevice& device, SandboxWindow& window);
	VkSandboxRenderer(const VkSandboxRenderer&) = delete;
	VkSandboxRenderer& operator=(const VkSandboxRenderer&) = delete;
	~VkSandboxRenderer() override;
	

	FrameContext beginFrame() override;
	
	void endFrame(FrameContext& frame) override;
	void beginSwapChainRenderPass(FrameContext& frame)override;
	void endSwapChainRenderPass(FrameContext& frame)override;

	void initializeSystems(IAssetProvider& assets, IScene& scene);

	void renderSystems(FrameInfo& info, FrameContext& frame)override;

	void waitDeviceIdle() override;

	void updateSystems(FrameInfo& frame, GlobalUbo& ubo, float deltaTime)override;

	void initImGui(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue graphicsQueue, uint32_t queueFamily);
	void shutdownImGui();

	const void beginImGuiFrame();
	const void renderImGui(FrameContext& frame);
	
	// Helpers
	VkRenderPass getSwapChainRenderPass() const { return m_swapchain->getRenderPass(); }
	float getAspectRatio() const { return m_swapchain->extentAspectRatio(); }
	bool isFrameInProgress() const { return m_bIsFrameStarted; }

	VkCommandBuffer getCurrentCommandBuffer() const {
		assert(m_bIsFrameStarted && "Cannot get command buffer when frame not in progress");
		return m_commandBuffers[m_currentFrameIndex];
	}

	int getFrameIndex() const {
		assert(m_bIsFrameStarted && "Cannot get frame index when frame not in progress");
		return m_currentFrameIndex;
	}

	const std::vector<VkDescriptorSet>& getGlobalDescriptorSet() const {
		return m_globalDescriptorSets;
	}
	const std::vector<std::unique_ptr<VkSandboxBuffer>>& getUboBuffers() const {
		return m_uboBuffers;
	}
	bool isImGuiInitialized() const { return m_imguiInitialized; }

	std::unique_ptr<VkSandboxDescriptorPool>                      m_pool;
private:

	std::vector<VkCommandBuffer>					    m_commandBuffers;
	VkCommandPool										   m_commandPool = VK_NULL_HANDLE;
	uint32_t								         m_currentImageIndex = 0;
	int												 m_currentFrameIndex = 0;
	bool										       m_bIsFrameStarted = false;

	std::unique_ptr<VkSandboxDescriptorSetLayout>		  m_globalLayout;

	VkSandboxDevice&											m_device;
	SandboxWindow&											    m_window;
	std::vector<std::unique_ptr<IRenderSystem>>				   m_systems;

	std::unique_ptr<VkSandboxSwapchain>					     m_swapchain;
	std::shared_ptr<VkSandboxSwapchain>					  m_oldSwapchain;
	VkInstance												  m_instance = VK_NULL_HANDLE;
	
	uint32_t								 m_width{ 0 }, m_height{ 0 };
	std::vector<std::unique_ptr<VkSandboxBuffer>>			m_uboBuffers;
	std::vector<VkDescriptorSet>				  m_globalDescriptorSets;
	std::vector<VkFence>							    m_inFlightFences;
	std::vector<VkImageLayout>					 m_swapchainImageLayouts;
	std::vector<VkImageLayout>						 m_depthImageLayouts;


	std::unique_ptr<SkyboxRenderSystem>					  m_skyboxSystem;
	std::unique_ptr<GltfRenderSystem>					    m_gltfSystem;
	std::unique_ptr<PointLightRenderSystem>			  m_pointLightSystem;
	std::unique_ptr<SceneRenderSystem>					   m_sceneSystem;

	VkDescriptorSetLayout m_iblSetLayout;
	std::vector<VkDescriptorSet> m_iblDescriptorSets;
	std::unique_ptr<VkSandboxDescriptorSetLayout> m_iblLayout;  


	void createGlobalDescriptorObjects();
	void allocateGlobalDescriptors();
	void createIblDescriptors(IAssetProvider& provider);

	void createSwapChain();
	void createCommandBuffers();
	void freeCommandBuffers();

	// imgui 
	bool m_imguiInitialized = false;
	VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;

	VkDescriptorPool create_imgui_descriptor_pool(VkDevice device);
	VkCommandBuffer createSingleUseCommandBuffer();
	void flushAndSubmitSingleUseCommandBuffer(VkCommandBuffer cmd);
};