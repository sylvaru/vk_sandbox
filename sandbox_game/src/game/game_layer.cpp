// game_layer.cpp
#include "game/game_layer.h"
#include "engine.h"
#include <spdlog/spdlog.h>


void MyGameLayer::onAttach(Core::SandboxEngine* engine) {
    m_engine = engine;
}
void MyGameLayer::onInit()
{
    m_windowInput = m_engine->getInputSharedPtr();
    m_assetManager = &m_engine->getAssetManager();
    spdlog::info("MyGameLayer::onInit");
    m_scene = std::make_unique<SandboxScene>(m_windowInput, *m_assetManager);
    m_scene->loadSceneFile("default_scene"); // TODO: Eventually specify which scene file to load in a better way than this probably via UI 
    m_scene->init();
}

void MyGameLayer::onUpdate(float dt)
{
    if (m_scene) m_scene->update(dt);
}


IScene* MyGameLayer::getSceneInterface() {
    return m_scene ? m_scene.get() : nullptr;
}
