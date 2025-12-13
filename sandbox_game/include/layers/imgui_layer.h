#pragma once
#include "interfaces/layer_i.h"
#include "interfaces/window_i.h"
#include "interfaces/renderer_i.h"
#include <iostream>
#include <memory>
#include "engine.h"

class ImGuiLayer : public ILayer {
public:
    void onAttach(Core::SandboxEngine* engine) override;
    void onInit() override;
    void onUpdate(float dt) override;
    void onRender(ISandboxRenderer::FrameContext& frame) override;
    void onDetach() override {}

    IScene* getSceneInterface() override;
private:
    Core::SandboxEngine* m_engine = nullptr;
    IWindow* m_window;
    Core::AssetManager* m_assetManager = nullptr;
    VkSandboxRenderer* m_prenderer = nullptr;

};