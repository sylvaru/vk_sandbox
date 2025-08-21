#include "entities/player.h"
#include <glm/gtc/matrix_transform.hpp>
#include "key_codes.h"
#include <spdlog/spdlog.h>

SandboxPlayer::SandboxPlayer(std::shared_ptr<IWindowInput> input)
    : m_pInput(std::move(input))
    , m_camera(glm::vec3(0.f, 0.f, 3.f))
   // , m_controller(m_moveSpeed, m_mouseSensitivity)
{

}

void SandboxPlayer::onInit() {  

}

void SandboxPlayer::onUpdate(float dt) {
    if (!m_pInput) return;

    try {
        double dx = 0, dy = 0;
        m_pInput->getMouseDelta(dx, dy);
        m_controller.mouseCallback(glm::vec2(dx, dy));
        m_controller.update(dt, m_pInput, m_transform);

        m_camera.setPosition(m_transform.translation);
        m_camera.setRotation(m_transform.rotation);

        int w, h;
        m_pInput->getFramebufferSize(w, h);
        float aspect = h == 0 ? 1.0f : static_cast<float>(w) / h;
        m_camera.updateProjection(aspect, 0.1f, 300.f);
    }
    catch (const std::exception& e) {
        spdlog::error("Exception in SandboxPlayer::onUpdate(): {}", e.what());
    }
    catch (...) {
        spdlog::error("Unknown exception in SandboxPlayer::onUpdate()");
    }
}

TransformComponent& SandboxPlayer::getTransform() {
    return m_transform;
}

std::shared_ptr<IModel> SandboxPlayer::getModel() const {
    return nullptr; 
}


SandboxCamera& SandboxPlayer::getCamera() {
    return m_camera;
}