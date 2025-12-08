#include "entities/player.h"
#include <glm/gtc/matrix_transform.hpp>
#include "key_codes.h"
#include <spdlog/spdlog.h>

SandboxPlayer::SandboxPlayer(std::shared_ptr<IWindowInput> input,
    const glm::vec3& startPos,
    const glm::vec3& startRotRad,
    float fov,
    float sensitivity,
    float moveSpeed,
    PhysicsEngine* physics)
    : m_pInput(std::move(input))
    , m_camera(startPos, glm::degrees(startRotRad).y, glm::degrees(startRotRad).x, fov)
    , m_mouseSensitivity(sensitivity)
    , m_moveSpeed(moveSpeed)
    , m_controller(moveSpeed, sensitivity, physics)
    , m_physics(physics)
{
    m_transform.translation = startPos;
    m_transform.rotation = startRotRad;
}
void SandboxPlayer::onInit() {

    m_pInput->setCursorCallback(
        [this](double dx, double dy) {
            m_controller.mouseCallback(glm::vec2(dx, dy));
        }
    );

    // Initialize controller yaw/pitch from transform
    m_controller.mouseCallback(glm::vec2(0.f)); // reset delta
    const glm::vec3 rot = m_transform.rotation;
    m_controller.setOrientation(glm::degrees(rot.y), glm::degrees(rot.x));

    m_physics->createFPScontroller(m_transform.translation, 0.3f, 1.8f);
    m_controller.setPhysicsController(m_physics->getFPScontroller());
}

void SandboxPlayer::onUpdate(float dt) {
    if (!m_pInput) return;

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