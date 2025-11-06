// imgui_layer.cpp
#include "layers/imgui_layer.h"
#include "engine.h"
#include <spdlog/spdlog.h>


void ImGuiLayer::onAttach(Core::SandboxEngine* engine) {
    m_engine = engine;
    m_prenderer = &static_cast<VkSandboxRenderer&>(m_engine->renderer());
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
    m_windowInput = m_engine->getInputSharedPtr();
    m_assetManager = &m_engine->getAssetManager();
    spdlog::info("MyGameLayer::onInit");

}

void ImGuiLayer::onUpdate(float dt)
{
}

void ImGuiLayer::onRender(ISandboxRenderer::FrameContext& frame) {

    m_prenderer->beginImGuiFrame();
    ImGui::Begin("Engine Debug");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::End();

}



IScene* ImGuiLayer::getSceneInterface() {
    return nullptr;
}



