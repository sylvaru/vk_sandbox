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

    void render(FrameInfo& frameInfo) override;


    inline void setCubemapTexture(const VkDescriptorImageInfo& info) {
        m_skyboxImageInfo = info;
        m_bHasCubemap = true;
    }

    inline void setCubemapByName(const std::string& name, const IAssetProvider& provider) {
        VkDescriptorImageInfo desc = provider.getCubemapDescriptor(name);
        setCubemapTexture(desc); // <--- this is where we can handle descriptor set allocation
    }

    void setSkyboxModel(const std::shared_ptr<IModel>& model) { m_skyboxModel = model; }

    void createSkyboxDescriptorSetLayout();
    void allocateAndWriteSkyboxDescriptorSet();
private:
    void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
    void createPipeline(VkRenderPass renderPass);


    VkDescriptorImageInfo m_skyboxImageInfo{};
    bool m_bHasCubemap = false;

    VkDescriptorSetLayout m_skyboxLayout;
    VkSandboxDevice& m_device;
    std::unique_ptr<VkSandboxPipeline> m_pipeline;
    VkPipelineLayout m_pipelineLayout;

    std::unique_ptr<VkSandboxDescriptorSetLayout> m_skyboxSetLayout;
    VkDescriptorSet m_skyboxDescriptorSet;

    VkSandboxDescriptorPool* m_descriptorPool = nullptr;

    std::shared_ptr<IModel> m_skyboxModel;
};
