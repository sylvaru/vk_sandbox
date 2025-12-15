#pragma once
#include "global_common/render_data.h"
#include "vulkan_wrapper/vulkan_gltf.h"
#include <vector>
#include <unordered_map>
#include <mutex>

struct MeshInfo {
    VkBuffer vertexBuffer;
    VkBuffer indexBuffer;
    VkDeviceSize vertexOffset;
    VkDeviceSize indexOffset;
    uint32_t indexCount;
    VkIndexType indexType;
};

struct MaterialInfo {
    uint32_t materialIndex;
};

class RenderableRegistry {
public:
    RenderableRegistry() = default;

    // Register a mesh entry (call at model load time). Returns meshIndex.
    uint32_t registerMesh(const MeshInfo& mesh);

    // Register a material (returns materialIndex)
    uint32_t registerMaterial(const MaterialInfo& mat);

    // Create an instance (one instance per game-object) and return RenderableID
    RenderableID createInstance(
        uint32_t meshIndex,
        uint32_t materialIndex,
        const TransformData& transform,
        RenderableType type
    );

    // Update transform for an instance (call each frame or when object moves)
    void updateInstanceTransform(RenderableID id, const TransformData& t);

    // Remove instance
    void removeInstance(RenderableID id);

    // Query pools
    const std::vector<MeshInfo>& getMeshPool() const { return m_meshPool; }
    const std::vector<MaterialInfo>& getMaterialPool() const { return m_materialPool; }
    const std::vector<MeshInstance>& getInstancePool() const { return m_instances; }
    const std::vector<TransformData>& getTransformPool() const { return m_transforms; }

    void setModelPointer(RenderableID id, vkglTF::Model* model) {
        for (auto& inst : m_instances) {
            if (inst.id == id) {
                inst.model = model;
                break;
            }
        }
    }

    // Give access to lists for culling
    size_t instanceCount() const { return m_instances.size(); }

    // Map from RenderableID -> index inside pools for quick lookup
    int instanceIndexFromID(RenderableID id) const;

    const std::vector<MeshInstance*> getInstancesByType(RenderableType type) const;
    std::vector<const MeshInstance*> getInstances() const;

    MeshInstance* getInstanceMutable(RenderableID id);


private:
    std::vector<MeshInfo> m_meshPool;
    std::vector<MaterialInfo> m_materialPool;

    // Instances are compacted vectors; index == transform index
    std::vector<MeshInstance> m_instances;
    std::vector<TransformData> m_transforms;

    // ID -> dense index mapping
    std::vector<RenderableID> m_idToIndex;
    std::vector<RenderableID> m_freeList;
    RenderableID m_nextId{ 0 };

    mutable std::mutex m_mutex; // simple thread-safety
};
