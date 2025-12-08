#pragma once

#include "vulkan_device.h"

// std
#include <memory>
#include <unordered_map>
#include <vector>


class VkSandboxDescriptorSetLayout
{
public:
    class Builder
    {
    public:
        Builder(VkSandboxDevice& device) : m_device{ device } {}

        Builder& addBinding(
            uint32_t binding,
            VkDescriptorType descriptorType,
            VkShaderStageFlags stageFlags,
            uint32_t count,
            VkDescriptorBindingFlags flags);

        Builder& addBinding(
            uint32_t binding,
            VkDescriptorType descriptorType,
            VkShaderStageFlags stageFlags);

        std::unique_ptr<VkSandboxDescriptorSetLayout> build() const;

    private:
        VkSandboxDevice& m_device;
        std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> m_bindings{};
        std::unordered_map<uint32_t, VkDescriptorBindingFlags> m_bindingFlags{};
    };

    VkSandboxDescriptorSetLayout(
        VkSandboxDevice& device,
        const std::vector<VkDescriptorSetLayoutBinding>& bindingsVec,
        VkDescriptorSetLayout layout
    );
    VkSandboxDescriptorSetLayout(VkSandboxDevice& device, VkDescriptorSetLayout layout)
        : m_device{ device }, m_descriptorSetLayout{ layout } {
    }
    ~VkSandboxDescriptorSetLayout();
    VkSandboxDescriptorSetLayout(const VkSandboxDescriptorSetLayout&) = delete;
    VkSandboxDescriptorSetLayout& operator=(const VkSandboxDescriptorSetLayout&) = delete;

    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }

private:
    VkSandboxDevice& m_device;
    VkDescriptorSetLayout m_descriptorSetLayout;
    friend class VkcDescriptorWriter;
public:
    std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> m_bindings;
    

  
};

class VkSandboxDescriptorPool
{
public:
    class Builder
    {
    public:
        Builder(VkSandboxDevice& device) : m_device{ device } {}

        Builder& addPoolSize(VkDescriptorType descriptorType, uint32_t count);
        Builder& setPoolFlags(VkDescriptorPoolCreateFlags flags);
        Builder& setMaxSets(uint32_t count);
        std::unique_ptr<VkSandboxDescriptorPool> build() const;

    private:
        VkSandboxDevice& m_device;
        std::vector<VkDescriptorPoolSize> m_poolSizes{};
        uint32_t m_maxSets = 1000;
        VkDescriptorPoolCreateFlags m_poolFlags = 0;
    };

    VkSandboxDescriptorPool(
        VkSandboxDevice& vkcDevice,
        uint32_t maxSets,
        VkDescriptorPoolCreateFlags poolFlags,
        const std::vector<VkDescriptorPoolSize>& poolSizes);
    ~VkSandboxDescriptorPool();
    VkSandboxDescriptorPool(const VkSandboxDescriptorPool&) = delete;
    VkSandboxDescriptorPool& operator=(const VkSandboxDescriptorPool&) = delete;

    bool allocateDescriptor(
        const VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet& descriptor, uint32_t variableDescriptorCount) const;

    void freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const;

    void resetPool();

    VkDescriptorPool getHandle() const { return m_descriptorPool; }


    VkSandboxDevice& m_device;
    VkDescriptorPool m_descriptorPool;
private:
 


    friend class VkcDescriptorWriter;
};

class VkSandboxDescriptorWriter
{
public:
    VkSandboxDescriptorWriter(VkSandboxDescriptorSetLayout& setLayout, VkSandboxDescriptorPool& pool);

    VkSandboxDescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorBufferInfo* bufferInfo);
    VkSandboxDescriptorWriter& writeImage(uint32_t binding,const VkDescriptorImageInfo* imageInfo);
    VkSandboxDescriptorWriter& writeImage(uint32_t binding,const VkDescriptorImageInfo* imageInfos, uint32_t count);
    VkSandboxDescriptorWriter& writeImageArray(uint32_t binding, const std::vector<VkDescriptorImageInfo>& imageInfos);
    bool build(VkDescriptorSet& set);
    void overwrite(VkDescriptorSet& set);

private:
    VkSandboxDescriptorSetLayout& m_setLayout;
    VkSandboxDescriptorPool& m_pool;
    std::vector<VkWriteDescriptorSet> m_writes;
    uint32_t m_variableDescriptorCount = 0;
};