#include "vulkan_wrapper/vulkan_descriptor.h"


// std
#include <cassert>
#include <stdexcept>

uint32_t maxTextures = 1000;



// *************** Descriptor Set Layout Builder *********************

VkSandboxDescriptorSetLayout::Builder& VkSandboxDescriptorSetLayout::Builder::addBinding(
    uint32_t binding,
    VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags,
    uint32_t count,
    VkDescriptorBindingFlags flags) {

    assert(m_bindings.count(binding) == 0 && "Binding already in use");

    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = descriptorType;
    layoutBinding.descriptorCount = count;
    layoutBinding.stageFlags = stageFlags;
    layoutBinding.pImmutableSamplers = nullptr;

    m_bindings[binding] = layoutBinding;
    m_bindingFlags[binding] = flags;

    return *this;
}
VkSandboxDescriptorSetLayout::Builder& VkSandboxDescriptorSetLayout::Builder::addBinding(
    uint32_t binding,
    VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags) {

    // Default count = 1, no special flags
    return addBinding(binding, descriptorType, stageFlags, 1, 0);
}
std::unique_ptr<VkSandboxDescriptorSetLayout> VkSandboxDescriptorSetLayout::Builder::build() const {
    std::vector<VkDescriptorSetLayoutBinding> setBindings;
    std::vector<VkDescriptorBindingFlags> setBindingFlags;

    for (const auto& [binding, layoutBinding] : m_bindings) {
        setBindings.push_back(layoutBinding);
        setBindingFlags.push_back(m_bindingFlags.at(binding));
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(setBindingFlags.size());
    bindingFlagsInfo.pBindingFlags = setBindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(setBindings.size());
    layoutInfo.pBindings = setBindings.data();
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(m_device.device(), &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    return std::make_unique<VkSandboxDescriptorSetLayout>(m_device, setBindings, layout);
}
// *************** Descriptor Set Layout *********************

VkSandboxDescriptorSetLayout::VkSandboxDescriptorSetLayout(
    VkSandboxDevice& device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindingsVec,
    VkDescriptorSetLayout layout
) : m_device{ device }, m_descriptorSetLayout{ layout } {

    for (const auto& binding : bindingsVec) {
        m_bindings[binding.binding] = binding;
    }
}


VkSandboxDescriptorSetLayout::~VkSandboxDescriptorSetLayout() {
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device.device(), m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = VK_NULL_HANDLE;
    }
}

// *************** Descriptor Pool Builder *********************

VkSandboxDescriptorPool::Builder& VkSandboxDescriptorPool::Builder::addPoolSize(
    VkDescriptorType descriptorType, uint32_t count) {
    m_poolSizes.push_back({ descriptorType, count });
    return *this;
}

VkSandboxDescriptorPool::Builder& VkSandboxDescriptorPool::Builder::setPoolFlags(
    VkDescriptorPoolCreateFlags flags) {
    m_poolFlags = flags;
    return *this;
}
VkSandboxDescriptorPool::Builder& VkSandboxDescriptorPool::Builder::setMaxSets(uint32_t count) {
    m_maxSets = count;
    return *this;
}

std::unique_ptr<VkSandboxDescriptorPool> VkSandboxDescriptorPool::Builder::build() const {
    return std::make_unique<VkSandboxDescriptorPool>(m_device, m_maxSets, m_poolFlags, m_poolSizes);
}

// *************** Descriptor Pool *********************

VkSandboxDescriptorPool::VkSandboxDescriptorPool(
    VkSandboxDevice& device,
    uint32_t maxSets,
    VkDescriptorPoolCreateFlags poolFlags,
    const std::vector<VkDescriptorPoolSize>& poolSizes)
    : m_device{device} {
    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPoolInfo.maxSets = maxSets;
    descriptorPoolInfo.flags = poolFlags;

    if (vkCreateDescriptorPool(m_device.device(), &descriptorPoolInfo, nullptr, &m_descriptorPool) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

VkSandboxDescriptorPool::~VkSandboxDescriptorPool() {
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device.device(), m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
}
bool VkSandboxDescriptorPool::allocateDescriptor(
    const VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorSet& descriptor,
    uint32_t variableDescriptorCount
) const {
    // Set up allocation info
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    // Optional variable descriptor count extension
    VkDescriptorSetVariableDescriptorCountAllocateInfo countInfo{};
    if (variableDescriptorCount > 0) {
        countInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
        countInfo.descriptorSetCount = 1;
        countInfo.pDescriptorCounts = &variableDescriptorCount;

        allocInfo.pNext = &countInfo; // chain if used
    }

    // Try to allocate
    if (vkAllocateDescriptorSets(m_device.device(), &allocInfo, &descriptor) != VK_SUCCESS) {
        return false;
    }

    return true;
}


void VkSandboxDescriptorPool::freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const {
    vkFreeDescriptorSets(
        m_device.device(),
        m_descriptorPool,
        static_cast<uint32_t>(descriptors.size()),
        descriptors.data());
}

void VkSandboxDescriptorPool::resetPool() {
    vkResetDescriptorPool(m_device.device(), m_descriptorPool, 0);
}

// *************** Descriptor Writer *********************

VkSandboxDescriptorWriter::VkSandboxDescriptorWriter(VkSandboxDescriptorSetLayout& setLayout, VkSandboxDescriptorPool& pool)
    : m_setLayout{ setLayout }, m_pool{ pool } {
}

VkSandboxDescriptorWriter& VkSandboxDescriptorWriter::writeBuffer(
    uint32_t binding, VkDescriptorBufferInfo* bufferInfo) {
    assert(m_setLayout.m_bindings.count(binding) == 1 && "Layout does not contain specified binding");

    auto& bindingDescription = m_setLayout.m_bindings[binding];

    assert(
        bindingDescription.descriptorCount == 1 &&
        "Binding single descriptor info, but binding expects multiple");

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.descriptorType = bindingDescription.descriptorType;
    write.dstBinding = binding;
    write.pBufferInfo = bufferInfo;
    write.descriptorCount = 1;

    m_writes.push_back(write);
    return *this;
}

VkSandboxDescriptorWriter& VkSandboxDescriptorWriter::writeImage(uint32_t binding, const VkDescriptorImageInfo* imageInfo)
{
    auto& bindingDescription = m_setLayout.m_bindings[binding];
    assert(m_setLayout.m_bindings.count(binding) == 1 && "Layout does not contain specified binding");



    assert(
        bindingDescription.descriptorCount == 1 &&
        "Binding single descriptor info, but binding expects multiple");

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.descriptorType = bindingDescription.descriptorType;
    write.dstBinding = binding;
    write.pImageInfo = imageInfo;
    write.descriptorCount = 1;

    m_writes.push_back(write);
    return *this;
}
VkSandboxDescriptorWriter& VkSandboxDescriptorWriter::writeImage(uint32_t binding, const VkDescriptorImageInfo* imageInfos, uint32_t count) {
    assert(m_setLayout.m_bindings.count(binding) == 1 && "Layout does not contain specified binding");
    assert(m_setLayout.m_bindings.at(binding).descriptorCount >= count && "Too many image descriptors for binding");


    m_variableDescriptorCount = count;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = m_setLayout.m_bindings[binding].descriptorType;
    write.descriptorCount = count;
    write.pImageInfo = imageInfos;
    write.pBufferInfo = nullptr;
    write.pTexelBufferView = nullptr;

    m_writes.push_back(write);
    return *this;
}

VkSandboxDescriptorWriter& VkSandboxDescriptorWriter::writeImageArray(
    uint32_t binding,
    const std::vector<VkDescriptorImageInfo>& imageInfos)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = static_cast<uint32_t>(imageInfos.size());
    write.pImageInfo = imageInfos.data();

    m_writes.push_back(write);
    return *this;
}

bool VkSandboxDescriptorWriter::build(VkDescriptorSet& set) {
    bool success = m_pool.allocateDescriptor(m_setLayout.getDescriptorSetLayout(), set, m_variableDescriptorCount);
    if (!success) return false;
    overwrite(set);
    return true;
}

void VkSandboxDescriptorWriter::overwrite(VkDescriptorSet& set) {
    for (auto& write : m_writes) {
        write.dstSet = set;
    }
    vkUpdateDescriptorSets(m_pool.m_device.device(), m_writes.size(), m_writes.data(), 0, nullptr);
}