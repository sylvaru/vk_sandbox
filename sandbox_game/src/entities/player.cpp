#include "common/game_pch.h"
#include "entities/player.h"
#include "global_common/key_codes.h"


SandboxPlayer::SandboxPlayer(
    IWindow& window,
    const glm::vec3& startPos,
    const glm::vec3& startRotRad,
    float fov,
    float sensitivity,
    float moveSpeed,
    PhysicsEngine* physics)
    : m_window(window)
    , m_initialPosition(startPos)
    , m_initialRotation(startRotRad)
    , m_controller(startPos,
        glm::degrees(startRotRad).y,
        glm::degrees(startRotRad).x,
        fov,
        moveSpeed,
        sensitivity,
        physics)
    , m_physics(physics)
    , m_initialFov(fov)
    , m_initialSensitivity(sensitivity)
    , m_initialMoveSpeed(moveSpeed)
{
}

void SandboxPlayer::onInit() {
    // Ensure controller starts at scene JSON defaults
    const glm::vec3 startPos = m_initialPosition; // from scene JSON
    const glm::vec3 startRot = m_initialRotation; // radians from scene JSON

    m_controller.setYawPitch(glm::degrees(startRot.y), glm::degrees(startRot.x));

    // Create physics controller at the correct position
    m_physics->createFPScontroller(startPos, 0.3f, 1.8f);
    m_controller.setPhysicsController(m_physics->getFPScontroller());

    // Sync camera to physics controller
    m_controller.getCamera().setPosition(startPos);
}


void SandboxPlayer::onUpdate(float dt) {

    double dx, dy;
    m_window.consumeMouseDelta(dx, dy);
    m_controller.mouseCallback(glm::vec2(dx, dy));

    m_controller.update(dt, m_window);

    auto& cam = m_controller.getCamera();

    int w, h;
    m_window.getFramebufferSize(w, h);
    float aspect = h == 0 ? 1.0f : static_cast<float>(w) / h;
    cam.updateProjection(aspect, 0.1f, 300.f);
}

TransformComponent& SandboxPlayer::getTransform() {
    return m_transform;
}

std::shared_ptr<IModel> SandboxPlayer::getModel() const {
    return nullptr;
}

