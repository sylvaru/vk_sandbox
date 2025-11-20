#include "base/engine_scene_base.h"

EngineSceneBase::EngineSceneBase(Core::AssetManager* assetManager)
    : m_assetManager(assetManager)
{
}
void EngineSceneBase::init()
{
    for (auto& [id, obj] : m_gameObjects) {
        obj->onInit();
    }
}
void EngineSceneBase::update(float dt)
{
    if (!m_physicsEngine) return;

    // Step simulation
    m_physicsEngine->stepSimulation(dt);

    // Update all game objects
    for (auto& [id, obj] : m_gameObjects)
        obj->onUpdate(dt);
}


void EngineSceneBase::addGameObject(std::shared_ptr<IGameObject> obj)
{
    m_gameObjects[obj->getId()] = obj;
}

void EngineSceneBase::removeGameObject(uint32_t id)
{
    removeRenderable(id);
    m_gameObjects.erase(id);
}



RenderableID EngineSceneBase::createRenderable(
    uint32_t gameObjectId,
    uint32_t meshIndex,
    uint32_t materialIndex,
    const TransformData& t,
    const glm::vec3& bsCenter,
    float bsRadius,
    RenderableType type)
{
    // Ask the registry to create a new renderable
    RenderableID rid = m_renderRegistry.createInstance(
        meshIndex, materialIndex, t, bsCenter, bsRadius, type);

    // Track which renderable belongs to which game object
    m_goRenderable[gameObjectId] = rid;

    // Retrieve the object
    auto it = m_gameObjects.find(gameObjectId);
    if (it == m_gameObjects.end())
        return rid;

    auto& go = it->second;

    if (auto model = go->getModel()) {

        if (auto gltfModel = std::dynamic_pointer_cast<vkglTF::Model>(model)) {
            m_renderRegistry.setModelPointer(rid, gltfModel.get());

            // create static physics mesh
            if (m_physicsEngine) {
                const auto& vertices = gltfModel->cpuPositions;
                const auto& indices = gltfModel->cpuIndices;

                auto filtered = m_physicsEngine->extractCollisionMesh(vertices, indices);

                m_physicsEngine->createStaticTriangleMesh(
                    filtered.vertices,
                    filtered.indices,
                    t.model
                );
            }
        }
    }
    return rid;
}

void EngineSceneBase::removeRenderable(uint32_t id)
{
    if (auto it = m_goRenderable.find(id); it != m_goRenderable.end())
    {
        m_renderRegistry.removeInstance(it->second);
        m_goRenderable.erase(it);
    }
}
void EngineSceneBase::clearScene()
{
    for (auto& [id, _] : m_gameObjects)
        removeRenderable(id);

    m_gameObjects.clear();
    m_skyboxId.reset();
}

void EngineSceneBase::setSkyboxObject(std::shared_ptr<IGameObject> obj)
{
    m_skyboxId = obj->getId();
    addGameObject(obj);
}

std::optional<std::reference_wrapper<IGameObject>>
EngineSceneBase::getSkyboxObject() const
{
    if (!m_skyboxId) return std::nullopt;
    auto it = m_gameObjects.find(*m_skyboxId);
    if (it == m_gameObjects.end()) return std::nullopt;
    return *it->second;
}