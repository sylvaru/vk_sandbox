// game_layer.cpp
#include "common/game_pch.h"
#include "layers/game_layer.h"
#include "engine.h"


void MyGameLayer::onAttach(Core::SandboxEngine* engine) {
    m_isAttached = true;
    m_engine = engine;
}

void MyGameLayer::onInit()
{
    m_window = &m_engine->getWindow();
    m_assetManager = &m_engine->getAssetManager();
    spdlog::info("MyGameLayer::onInit");
    m_scene = std::make_unique<SandboxScene>(*m_window, *m_assetManager);
    m_scene->setPhysicsEngine(&m_engine->getPhysicsEngine());
    m_scene->loadSceneFile("default_scene"); 
    m_engine->setActiveScene(m_scene.get(), this);
}

void MyGameLayer::onUpdate(float dt)
{
    if (m_scene) m_scene->update(dt);
}

IScene* MyGameLayer::getSceneInterface() {
    return m_scene ? m_scene.get() : nullptr;
}
