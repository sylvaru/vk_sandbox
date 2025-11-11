#include "entities/player.h"
#include <glm/gtc/matrix_transform.hpp>
#include "key_codes.h"
#include <spdlog/spdlog.h>

SandboxPlayer::SandboxPlayer(std::shared_ptr<IWindowInput> input,
    const glm::vec3& startPos,
    const glm::vec3& startRotRad,
    float fov,
    float sensitivity,
    float moveSpeed)
    : m_pInput(std::move(input))
    , m_camera(startPos, glm::degrees(startRotRad).y, glm::degrees(startRotRad).x, fov)
    , m_mouseSensitivity(sensitivity)
    , m_moveSpeed(moveSpeed)
    , m_controller(moveSpeed, sensitivity) // <- propagate into controller
{
    m_transform.translation = startPos;
    m_transform.rotation = startRotRad;
}
void SandboxPlayer::onInit() {
    // Initialize controller yaw/pitch from transform
    m_controller.mouseCallback(glm::vec2(0.f)); // reset delta
    const glm::vec3 rot = m_transform.rotation;
    m_controller.setOrientation(glm::degrees(rot.y), glm::degrees(rot.x));
}

void SandboxPlayer::onUpdate(float dt) {
    if (!m_pInput) return;

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

TransformComponent& SandboxPlayer::getTransform() {
    return m_transform;
}

std::shared_ptr<IModel> SandboxPlayer::getModel() const {
    return nullptr; 
}


SandboxCamera& SandboxPlayer::getCamera() {
    return m_camera;
}