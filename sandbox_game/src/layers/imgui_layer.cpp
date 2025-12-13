// imgui_layer.cpp
#include "common/game_pch.h"
#include "layers/imgui_layer.h"
#include "engine.h"



void ImGuiLayer::onAttach(Core::SandboxEngine* engine) {
    m_engine = engine;
    m_prenderer = &static_cast<VkSandboxRenderer&>(m_engine->renderer());
    m_window = &m_engine->getWindow();
    m_assetManager = &m_engine->getAssetManager();

    auto& device = m_engine->getDevice();
 
    m_prenderer->initImGui(
        m_engine->getInstance().instance(),
        device.physicalDevice(),
        device.device(),
        device.graphicsQueue(),
        device.graphicsQueueFamilyIndex()
    );
}
void ImGuiLayer::onInit()
{
    spdlog::info("ImGuiLayer::onInit");
}

void ImGuiLayer::onUpdate(float dt)
{
}

void ImGuiLayer::onRender(ISandboxRenderer::FrameContext& frame) {

    m_prenderer->beginImGuiFrame(); 
    ImGui::SetNextWindowSize(ImVec2(300, 120), ImGuiCond_FirstUseEver);
    ImGui::Begin("Engine Debug");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::End();
}

IScene* ImGuiLayer::getSceneInterface() {
    return nullptr;
}



