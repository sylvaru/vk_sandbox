#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_obj.h"
#include "vulkan_wrapper/vulkan_gltf.h"
#include "vulkan_wrapper/vulkan_texture.h"
#include "vulkan_wrapper/vulkan_pipeline.h"

#include "interfaces/asset_provider_i.h"



class AssetManager : public IAssetProvider {
public:
	AssetManager(VkSandboxDevice& device);
	~AssetManager();
	void preloadGlobalAssets();
	std::shared_ptr<VkSandboxOBJmodel> loadObjModel(const std::string& name, const std::string& filepath, bool isSkybox = false);
	std::shared_ptr<vkglTF::Model> loadGLTFmodel(const std::string& name, const std::string& filepath, uint32_t gltfFlags = 0u, float scale = 1.f);
	std::shared_ptr<VkSandboxTexture> loadCubemap(
		const std::string& name,
		const std::string& ktxFilename,
		VkFormat format,
		VkImageUsageFlags usageFlags,
		VkImageLayout initialLayout
	);

    void generateBRDFlut();
    void generateIrradianceMap();
    void generatePrefilteredEnvMap();
	
	using OBJmodelHandle = std::shared_ptr<VkSandboxOBJmodel>;
	using GLTFmodelHandle = std::shared_ptr<vkglTF::Model>;
	//using TextureHandle  = std::shared_ptr<VulkanTexture>;
	//using ShaderHandle = std::shared_ptr<ShaderModule>;

	VkDescriptorImageInfo getCubemapDescriptor(const std::string& name) const override {
		auto it = m_textures.find(name);
		if (it == m_textures.end()) {
			throw std::runtime_error("Cubemap not found: " + name);
		}
		return it->second->GetDescriptor();
	}

    // Inline getters
    std::shared_ptr<VkSandboxOBJmodel> getOBJModel(const std::string& name) const {
        auto it = m_objModelCache.find(name);
        return (it != m_objModelCache.end()) ? it->second : nullptr;
    }

    std::shared_ptr<vkglTF::Model> getGLTFmodel(const std::string& name) const override {
        auto it = m_gltfModelCache.find(name);
        return (it != m_gltfModelCache.end()) ? it->second : nullptr;
    }

    std::shared_ptr<VkSandboxTexture> getTexture(const std::string& name) const {
        auto it = m_textures.find(name);
        if (it == m_textures.end()) {
            throw std::runtime_error("Texture not found: " + name);
        }
        return it->second;
    }

    std::shared_ptr<VkSandboxTexture> getTexture(size_t index) const {
        if (index >= m_textureList.size()) {
            throw std::runtime_error("Texture index out of range: " + std::to_string(index));
        }
        return m_textureList[index];
    }

    size_t getTextureIndex(const std::string& name) const {
        auto it = m_textureIndexMap.find(name);
        if (it == m_textureIndexMap.end()) {
            throw std::runtime_error("Texture not found in index map: " + name);
        }
        return it->second;
    }

    const std::vector<std::shared_ptr<VkSandboxTexture>>& getAllTextures() const {
        return m_textureList;
    }

    bool hasTexture(const std::string& name) const {
        return m_textures.find(name) != m_textures.end();
    }


    VkDescriptorImageInfo getBRDFLUTDescriptor()    const override { return lutBrdf->GetDescriptor(); }
    VkDescriptorImageInfo getIrradianceDescriptor() const override { return irradianceCube->GetDescriptor(); }
    VkDescriptorImageInfo getPrefilteredDescriptor() const override { return prefilteredCube->GetDescriptor(); }

    VkDescriptorImageInfo getTextureDescriptor(const std::string& name) const override {
        return getTexture(name)->GetDescriptor();
    }
    GLTFmodelHandle getSkyboxModel() const { return m_skyboxModel; }// make this override if necessary

    std::vector<std::string> listTextureNames()    const override {
        std::vector<std::string> keys;
        keys.reserve(m_textures.size());
        for (const auto& [n, _] : m_textures) keys.push_back(n);
        return keys;
    }

private:
	std::unordered_map<std::string, OBJmodelHandle> m_objModelCache;
	std::unordered_map<std::string, GLTFmodelHandle> m_gltfModelCache;

	std::unordered_map<std::string, std::shared_ptr<VkSandboxTexture>>  m_textures; // name → texture
	std::unordered_map<std::string, size_t>                      m_textureIndexMap; // name → index
	std::vector<std::shared_ptr<VkSandboxTexture>>                   m_textureList; // index → texture

	VkSandboxDevice&			m_device;
	VkQueue						m_transferQueue;

	// caches
	std::shared_ptr<VkSandboxTexture> lutBrdf, irradianceCube, prefilteredCube, environmentCube;

    GLTFmodelHandle m_skyboxModel;

	static void registerTextureIfNeeded(
		const std::string& name,
		const std::shared_ptr<VkSandboxTexture>& tex,
		std::unordered_map<std::string, std::shared_ptr<VkSandboxTexture>>& textures,
		std::unordered_map<std::string, size_t>& textureIndexMap,
		std::vector<std::shared_ptr<VkSandboxTexture>>& textureList);



};