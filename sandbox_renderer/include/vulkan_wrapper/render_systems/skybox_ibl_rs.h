#pragma once
#include "interfaces/render_system_i.h"
#include "interfaces/asset_provider_i.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"
#include "vulkan_wrapper/vulkan_descriptor.h"
#include "vulkan_wrapper/vulkan_gltf.h"
#include <vulkan/vulkan.h>

// STD
#include <memory>
#include <vector>
#include <array>


class SkyboxIBLrenderSystem : public IRenderSystem {
public:
    SkyboxIBLrenderSystem(VkSandboxDevice& device, VkRenderPass renderPass, VkDescriptorSetLayout globalSetLayout);
    ~SkyboxIBLrenderSystem();

    SkyboxIBLrenderSystem(const SkyboxIBLrenderSystem&) = delete;
    SkyboxIBLrenderSystem& operator=(const SkyboxIBLrenderSystem&) = delete;

    void init(
        VkSandboxDevice& device,
        VkRenderPass            renderPass,
        VkDescriptorSetLayout   globalSetLayout,
        VkSandboxDescriptorPool& descriptorPool,
        size_t frameCount)override;

    // Call this inside your scene render loop, after global descriptors are bound
    void render(FrameInfo& frameInfo) override;


    inline void setCubemapTexture(const VkDescriptorImageInfo& info) {
        m_skyboxImageInfo = info;
        m_bHasCubemap = true;
        allocateAndWriteSkyboxDescriptorSet();
    }

    inline void setCubemapByName(const std::string& name, const IAssetProvider& provider) {
        VkDescriptorImageInfo desc = provider.getCubemapDescriptor(name);
        setCubemapTexture(desc); // <--- this is where we can handle descriptor set allocation
    }

    // Set the skybox model (shared_ptr from asset provider)
    void setSkyboxModel(std::shared_ptr<vkglTF::Model> model) {
        m_skyboxModel = std::move(model);
    }

    // Set the cubemap descriptor image info (for descriptor writes)
    void setSkyboxCubemap(const VkDescriptorImageInfo& cubemapDesc) {
        m_skyboxImageInfo = cubemapDesc;
    }

    // Flag cubemap presence
    void setHasCubemap(bool hasCubemap) {
        m_bHasCubemap = hasCubemap;
    }

    void createSkyboxDescriptorSetLayout();
    void allocateAndWriteSkyboxDescriptorSet();
private:
    void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
    void createPipeline(VkRenderPass renderPass);


    std::shared_ptr<vkglTF::Model> m_skyboxModel;
    VkDescriptorImageInfo m_skyboxImageInfo{};
    bool m_bHasCubemap = false;

    VkDescriptorSetLayout m_skyboxLayout;
    VkSandboxDevice& m_device;
    std::unique_ptr<VkSandboxPipeline> m_pipeline;
    VkPipelineLayout m_pipelineLayout;

    std::unique_ptr<VkSandboxDescriptorSetLayout> m_skyboxSetLayout;
    VkDescriptorSet m_skyboxDescriptorSet;

    VkSandboxDescriptorPool* m_descriptorPool = nullptr;

};
