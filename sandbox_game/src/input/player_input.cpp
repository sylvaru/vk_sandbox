// player_input.cpp
#include "common/game_pch.h"
#include "input/player_input.h"

SandboxMNKController::SandboxMNKController(
    const glm::vec3& startPos,
    float yawDeg,
    float pitchDeg,
    float fov,
    float moveSpeed,
    float sensitivity,
    PhysicsEngine* physics)
    : m_moveSpeed(moveSpeed)
    , m_mouseSensitivity(sensitivity)
    , m_yaw(yawDeg)
    , m_pitch(pitchDeg)
    , m_physics(physics)
    , m_camera(startPos, yawDeg, pitchDeg, fov)
{
}


void SandboxMNKController::mouseCallback(glm::vec2 delta) {
    m_rawDelta += delta;
}

void SandboxMNKController::update(float dt, std::shared_ptr<IWindowInput> input) {
    if (!input || dt <= 0.0f) return;

    // --- Update rotation from mouse ---
    float deltaYaw = -m_rawDelta.x * m_mouseSensitivity;
    float deltaPitch = -m_rawDelta.y * m_mouseSensitivity;

    m_yaw += deltaYaw;
    m_pitch += deltaPitch;
    m_pitch = glm::clamp(m_pitch, -89.f, 89.f);

    // --- Apply rotation to camera ---
    // Explicit yaw/pitch quaternions to avoid euler conversion issues
    glm::quat qYaw = glm::angleAxis(glm::radians(m_yaw), glm::vec3(0.f, 1.f, 0.f));
    glm::quat qPitch = glm::angleAxis(glm::radians(m_pitch), glm::vec3(1.f, 0.f, 0.f)); // world right
    m_camera.setOrientation(glm::normalize(qYaw * qPitch));


    // --- Compute input direction ---
    glm::vec3 front = m_camera.getForwardVector();
    glm::vec3 right = m_camera.getRightVector();
    glm::vec3 up = glm::vec3(0.f, 1.f, 0.f); // world up

    glm::vec3 dir(0.f);
    if (input->isKeyPressed(SandboxKey::W)) dir += front;
    if (input->isKeyPressed(SandboxKey::S)) dir -= front;
    if (input->isKeyPressed(SandboxKey::A)) dir -= right;
    if (input->isKeyPressed(SandboxKey::D)) dir += right;
    if (input->isKeyPressed(SandboxKey::Q)) dir -= up;
    if (input->isKeyPressed(SandboxKey::E)) dir += up;

    if (glm::length2(dir) > 1e-6f) {
        dir = glm::normalize(dir);
        float speed = m_moveSpeed * (input->isKeyPressed(SandboxKey::LEFT_SHIFT) ? 4.f : 1.f);
        glm::vec3 displacement = dir * speed * dt;

        if (m_controller) {
            m_physics->moveFPSController(displacement, dt);
            // Sync camera to physics controller position
            m_camera.setPosition(m_physics->getFPScontrollerPosition());
        }
        else {
            m_camera.move(displacement);
        }
    }

    // Reset mouse delta
    m_rawDelta = glm::vec2(0.f);
}

void SandboxMNKController::setYawPitch(float yawDeg, float pitchDeg) {
    m_yaw = yawDeg;
    m_pitch = glm::clamp(pitchDeg, -89.f, 89.f);

    // Update camera orientation immediately
    glm::quat qYaw = glm::angleAxis(glm::radians(m_yaw), glm::vec3(0.f, 1.f, 0.f));
    glm::quat qPitch = glm::angleAxis(glm::radians(m_pitch), m_camera.getRightVector());
    m_camera.setOrientation(glm::normalize(qYaw * qPitch));
}
