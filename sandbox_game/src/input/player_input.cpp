// player_input.cpp
#include "input/player_input.h"

SandboxMNKController::SandboxMNKController(float moveSpeed, float mouseSensitivity, PhysicsEngine* physics)
    : m_moveSpeed(moveSpeed) 
    , m_mouseSensitivity(mouseSensitivity)
    , m_yaw(-90.f), m_pitch(0.f)
    , m_physics(physics)
{
}

void SandboxMNKController::mouseCallback(glm::vec2 delta) {
    m_rawDelta += delta;
}


void SandboxMNKController::update(float dt, std::shared_ptr<IWindowInput> input, TransformComponent& transform) {
    if (!input || dt <= 0.0f) return;

    float deltaYaw = m_rawDelta.x * m_mouseSensitivity;
    float deltaPitch = -m_rawDelta.y * m_mouseSensitivity;


    m_yaw += deltaYaw;
    m_pitch += deltaPitch;
    m_pitch = glm::clamp(m_pitch, -89.f, 89.f);

    transform.rotation.x = glm::radians(m_pitch);
    transform.rotation.y = glm::radians(m_yaw);

    // Compute input direction
    glm::vec3 front{
        std::cos(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch)),
        std::sin(glm::radians(m_pitch)),
        std::sin(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch))
    };
    front = glm::normalize(front);
    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.f, 1.f, 0.f)));
    glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);

    glm::vec3 dir{ 0.f };
    if (input->isKeyPressed(SandboxKey::W)) dir += front;
    if (input->isKeyPressed(SandboxKey::S)) dir -= front;
    if (input->isKeyPressed(SandboxKey::A)) dir -= right;
    if (input->isKeyPressed(SandboxKey::D)) dir += right;
    if (input->isKeyPressed(SandboxKey::Q)) dir -= up;
    if (input->isKeyPressed(SandboxKey::E)) dir += up;

    if (glm::length(dir) > 1e-6f) {
        dir = glm::normalize(dir);
        float speed = m_moveSpeed * (input->isKeyPressed(SandboxKey::LEFT_SHIFT) ? 4.f : 1.f);
        glm::vec3 displacement = dir * speed * dt;

        if (m_controller) {
            // Move using PhysX capsule controller
            m_physics->moveFPSController(displacement, dt);
            transform.translation = m_physics->getFPScontrollerPosition();
        }
        else {
            // Fallback to simple movement
            transform.translation += displacement;
        }
    }

    m_rawDelta = glm::vec2(0.f);
}