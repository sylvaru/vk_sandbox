#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <memory>
#include <vulkan_wrapper/vulkan_gltf.h>

struct IAssetProvider {
	virtual VkDescriptorImageInfo getCubemapDescriptor(const std::string& name) const = 0;

    virtual VkDescriptorImageInfo getBRDFLUTDescriptor()    const = 0;
    virtual VkDescriptorImageInfo getIrradianceDescriptor() const = 0;
    virtual VkDescriptorImageInfo getPrefilteredDescriptor() const = 0;

    virtual VkDescriptorImageInfo getTextureDescriptor(const std::string& name) const = 0;

    virtual std::vector<std::string> listTextureNames()    const = 0;
    virtual std::shared_ptr<vkglTF::Model> getGLTFmodel(const std::string& name) const = 0;
};