#include "common/renderer_pch.h"
#include "vulkan_wrapper/core/renderable_registry.h"

uint32_t RenderableRegistry::registerMesh(const MeshInfo& mesh) {
    std::lock_guard<std::mutex> lk(m_mutex);
    m_meshPool.push_back(mesh);
    return static_cast<uint32_t>(m_meshPool.size() - 1);
}

uint32_t RenderableRegistry::registerMaterial(const MaterialInfo& mat) {
    std::lock_guard<std::mutex> lk(m_mutex);
    m_materialPool.push_back(mat);
    return static_cast<uint32_t>(m_materialPool.size() - 1);
}

RenderableID RenderableRegistry::createInstance(
    uint32_t meshIndex,
    uint32_t materialIndex,
    const TransformData& transform,
    RenderableType type) 
{
    std::lock_guard<std::mutex> lk(m_mutex);
    RenderableID id = m_nextId++;
    MeshInstance inst{};
    inst.id = id;
    inst.meshIndex = meshIndex;
    inst.materialIndex = materialIndex;
    inst.type = type;
    inst.transform = transform;

    uint32_t idx = static_cast<uint32_t>(m_instances.size());
    m_instances.push_back(inst);
    m_transforms.push_back(transform);

    if (idx >= m_idToIndex.size()) m_idToIndex.resize(idx + 1, INVALID_RENDERABLE);
    m_idToIndex[idx] = id;

    return id;
}


const std::vector<MeshInstance*> RenderableRegistry::getInstancesByType(RenderableType type) const {
    std::vector<MeshInstance*> out;
    out.reserve(m_instances.size());
    for (auto& inst : m_instances) {
        if (inst.type == type)
            out.push_back(const_cast<MeshInstance*>(&inst));
    }
    return out;
}


void RenderableRegistry::updateInstanceTransform(RenderableID id, const TransformData& t) {
    std::lock_guard<std::mutex> lk(m_mutex);
    // find dense index; search linear for brevity (optimize: keep map id->index)
    for (size_t i = 0; i < m_instances.size(); ++i) {
        if (m_instances[i].id == id) {
            m_transforms[i] = t;
            m_instances[i].transform = t;
            return;
        }
    }
}

void RenderableRegistry::removeInstance(RenderableID id) {
    std::lock_guard<std::mutex> lk(m_mutex);
    for (size_t i = 0; i < m_instances.size(); ++i) {
        if (m_instances[i].id == id) {
            // swap-pop
            size_t last = m_instances.size() - 1;
            if (i != last) {
                std::swap(m_instances[i], m_instances[last]);
                std::swap(m_transforms[i], m_transforms[last]);
            }
            m_instances.pop_back();
            m_transforms.pop_back();
            return;
        }
    }
}

int RenderableRegistry::instanceIndexFromID(RenderableID id) const {
    // linear search for now (fast enough for few thousands); optimize later with unordered_map<id,index>
    for (size_t i = 0; i < m_instances.size(); ++i) {
        if (m_instances[i].id == id) return static_cast<int>(i);
    }
    return -1;
}


MeshInstance* RenderableRegistry::getInstanceMutable(RenderableID id) {
    for (auto& inst : m_instances) {
        if (inst.id == id)
            return &inst;
    }
    return nullptr;
}

std::vector<const MeshInstance*> RenderableRegistry::getInstances() const {
	std::lock_guard<std::mutex> lk(m_mutex);
    std::vector<const MeshInstance*> out;
	out.reserve(m_instances.size());
    for (const auto& inst : m_instances) {
        out.push_back(&inst);
    }
	return out;
}