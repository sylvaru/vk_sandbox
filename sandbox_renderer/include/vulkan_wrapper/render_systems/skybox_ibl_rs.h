#pragma once
#include "interfaces/render_system_i.h"
#include "interfaces/asset_provider_i.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_pipeline.h"
#include "vulkan_wrapper/vulkan_descriptor.h"
#include "vulkan_wrapper/vulkan_gltf.h"
#include "vulkan_wrapper/core/render_graph.h"
#include <vulkan/vulkan.h>

// STD
#include <memory>
#include <vector>
#include <array>


class SkyboxIBLrenderSystem : public IRenderSystem {
public:
    SkyboxIBLrenderSystem(
        VkSandboxDevice& device,
        VkRenderPass renderPass,
        VkDescriptorSetLayout globalSetLayout,
        VkSandboxDescriptorPool& pool);

    ~SkyboxIBLrenderSystem();

    SkyboxIBLrenderSystem(const SkyboxIBLrenderSystem&) = delete;
    SkyboxIBLrenderSystem& operator=(const SkyboxIBLrenderSystem&) = delete;

    void init(
        VkSandboxDevice& device,
        VkRenderPass            renderPass,
        VkDescriptorSetLayout   globalSetLayout,
        VkSandboxDescriptorPool& descriptorPool);

    void render(FrameInfo& frameInfo) override;
    void record(const RGContext& rgctx, FrameInfo& frame);

    inline void setCubemapTexture(const VkDescriptorImageInfo& info) {
        m_skyboxImageInfo = info;
        m_bHasCubemap = true;
    }

    inline void setCubemapByName(const std::string& name, const IAssetProvider& provider) {
        VkDescriptorImageInfo desc = provider.getCubemapDescriptor(name);
        setCubemapTexture(desc);
    }

    void setSkyboxModel(const std::shared_ptr<IModel>& model) { m_skyboxModel = model; }

    void createSkyboxDescriptorSetLayout();
    void allocateAndWriteSkyboxDescriptorSet();

    bool m_bHasCubemap = false;
private:
    void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
    void createPipeline(VkRenderPass renderPass);


    VkDescriptorImageInfo m_skyboxImageInfo{};


    VkDescriptorSetLayout m_skyboxLayout;
    VkSandboxDevice& m_device;
    std::unique_ptr<VkSandboxPipeline> m_pipeline;
    VkPipelineLayout m_pipelineLayout;

    std::unique_ptr<VkSandboxDescriptorSetLayout> m_skyboxSetLayout;
    VkDescriptorSet m_skyboxDescriptorSet;

    VkSandboxDescriptorPool* m_descriptorPool = nullptr;

    std::shared_ptr<IModel> m_skyboxModel;
};
