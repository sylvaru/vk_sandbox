#pragma once

#include "interfaces/game_object_i.h"
#include "interfaces/scene_i.h"
#include "vulkan_wrapper/core/renderable_registry.h"
#include "physics/physics_engine.h"

#include <unordered_set>
#include "asset_manager.h"

// EngineSceneBase class contains the low level scene mechanisms (renderables, physics, object lifetime, etc)
// It is the central owner of all game objects


class EngineSceneBase : public IScene {
public:
    EngineSceneBase(Core::AssetManager* assetManager);
  

    // Default behavior shared by all scenes
    void init() override;
    void update(float dt) override;

    // Game objects
    void addGameObject(std::shared_ptr<IGameObject> obj);
    void removeGameObject(uint32_t id);

    std::unordered_map<uint32_t, std::shared_ptr<IGameObject>>&
        getGameObjects() override { return m_gameObjects; }

    // Skybox
    void setSkyboxObject(std::shared_ptr<IGameObject> obj);
    std::optional<std::reference_wrapper<IGameObject>>
        getSkyboxObject() const override;

    void setPhysicsEngine(PhysicsEngine* physics) { m_physicsEngine = physics; }
    PhysicsEngine* getPhysicsEngine() const { return m_physicsEngine; }

    void initPhysics() { m_physicsEngine->initPhysx(); }

    // Rendering registry
    RenderableRegistry& getRenderRegistry() { return m_renderRegistry; }
    const RenderableRegistry* getRenderableRegistry() const override {
        return &m_renderRegistry;
    }

    void clearScene();

protected:
    RenderableID createRenderable(
        uint32_t gameObjectId,
        uint32_t meshIndex,
        uint32_t materialIndex,
        const TransformData& t,
        RenderableType type);

    void removeRenderable(uint32_t gameObjectId);

protected:
    std::unordered_map<uint32_t, std::shared_ptr<IGameObject>> m_gameObjects;
    RenderableRegistry                                          m_renderRegistry;
    std::unordered_map<uint32_t, RenderableID>                  m_goRenderable;

    std::optional<uint32_t>            m_skyboxId;
    std::shared_ptr<IGameObject>       m_skyboxObject;
    std::string                        m_skyboxCubemapName = "skybox_hdr";

    PhysicsEngine* m_physicsEngine = nullptr;

    std::unordered_set<int> m_playerIds;
    Core::AssetManager* m_assetManager;
};