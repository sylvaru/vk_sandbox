#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <memory>
#include <unordered_map>

namespace vkglTF { class Model; }
class VkSandboxTexture;

struct IAssetProvider {
	virtual VkDescriptorImageInfo getCubemapDescriptor(const std::string& name) const = 0;

    virtual VkDescriptorImageInfo getBRDFLUTDescriptor()    const = 0;
    virtual VkDescriptorImageInfo getIrradianceDescriptor() const = 0;
    virtual VkDescriptorImageInfo getPrefilteredDescriptor() const = 0;

    virtual VkDescriptorImageInfo getTextureDescriptor(const std::string& name) const = 0;

    virtual std::vector<std::string> listTextureNames()    const = 0;
    virtual std::shared_ptr<vkglTF::Model> getGLTFmodel(const std::string& name) const = 0;

    virtual size_t registerTextureIfNeeded(
        const std::string& name,
        const std::shared_ptr<VkSandboxTexture>& tex,
        std::unordered_map<std::string, std::shared_ptr<VkSandboxTexture>>& textures,
        std::unordered_map<std::string, size_t>& textureIndexMap,
        std::vector<std::shared_ptr<VkSandboxTexture>>& textureList) = 0;
};