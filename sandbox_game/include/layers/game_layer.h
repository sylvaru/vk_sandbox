#pragma once
#include "interfaces/layer_i.h"
#include "interfaces/window_input_i.h"
#include "interfaces/renderer_i.h"
#include "entities/player.h"
#include "scene/scene.h"
#include <iostream>
#include <memory>

class MyGameLayer : public ILayer {
public:
    void onAttach(Core::SandboxEngine* engine) override;
    void onInit() override;
    void onUpdate(float dt) override;

    IScene* getSceneInterface() override;
private:
    Core::SandboxEngine* m_engine = nullptr;
    std::unique_ptr<SandboxScene> m_scene;
    std::shared_ptr<IWindowInput> m_windowInput;
    Core::AssetManager* m_assetManager = nullptr;

};