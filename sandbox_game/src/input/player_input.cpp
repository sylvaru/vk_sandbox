
// player_input.cpp
#include "input/player_input.h"

SandboxMNKController::SandboxMNKController(float moveSpeed, float mouseSensitivity)
    : m_moveSpeed(moveSpeed), m_mouseSensitivity(mouseSensitivity), m_yaw(-90.f), m_pitch(0.f)
{
}

void SandboxMNKController::mouseCallback(glm::vec2 delta) {
    m_rawDelta = delta;
}
void SandboxMNKController::update(float dt, std::shared_ptr<IWindowInput> input, TransformComponent& transform) {
    if (!input || dt <= 0.0f) return;

    // --- 1) Smooth raw mouse delta ---
    float alpha = 1.0f - std::exp(-m_smoothing * dt);
    m_smoothDelta += (m_rawDelta - m_smoothDelta) * alpha;

    // convert pixels -> degrees
    float deltaYaw = m_smoothDelta.x * m_mouseSensitivity;
    float deltaPitch = -m_smoothDelta.y * m_mouseSensitivity; // invert Y for typical FPS

    // --- 2) Update camera rotation ---
    m_yaw += deltaYaw;
    m_pitch += deltaPitch;
    m_pitch = glm::clamp(m_pitch, -89.f, 89.f);

    transform.rotation.x = glm::radians(m_pitch);
    transform.rotation.y = glm::radians(m_yaw);

    // --- 3) Compute movement basis ---
    glm::vec3 front{
        std::cos(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch)),
        std::sin(glm::radians(m_pitch)),
        std::sin(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch))
    };
    front = glm::normalize(front);
    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.f, 1.f, 0.f)));
    glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);

    // --- 4) Apply WASDQE movement ---
    glm::vec3 dir{ 0.f };
    if (input->isKeyPressed(SandboxKey::W)) dir += front;
    if (input->isKeyPressed(SandboxKey::S)) dir -= front;
    if (input->isKeyPressed(SandboxKey::A)) dir -= right;
    if (input->isKeyPressed(SandboxKey::D)) dir += right;
    if (input->isKeyPressed(SandboxKey::Q)) dir -= up;
    if (input->isKeyPressed(SandboxKey::E)) dir += up;

    if (glm::length(dir) > 1e-6f) {
        dir = glm::normalize(dir);
        float speed = m_moveSpeed * (input->isKeyPressed(SandboxKey::LEFT_SHIFT) ? 3.f : 1.f);
        transform.translation += dir * speed * dt;
    }

    // --- 5) Reset raw delta for next frame ---
    m_rawDelta = glm::vec2(0.f);
}
