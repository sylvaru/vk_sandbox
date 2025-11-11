#pragma once

#include "interfaces/window_input_i.h"
#include "transform_component.h"
#include <glm/glm.hpp>
#include <memory>
#include <functional>

class SandboxMNKController {
public:
    SandboxMNKController(float moveSpeed = 7.f, float mouseSensitivity = 0.08f);

    void update(float dt, std::shared_ptr<IWindowInput> input, TransformComponent& transform);
    void mouseCallback(glm::vec2 delta);
    void setMoveSpeed(float s) { m_moveSpeed = s; }
    void setMouseSensitivity(float s) { m_mouseSensitivity = s; }
    void setOrientation(float yawDeg, float pitchDeg) { m_yaw = yawDeg; m_pitch = pitchDeg; }
    float getYaw() const { return m_yaw; }
    float getPitch() const { return m_pitch; }

private:
    float m_moveSpeed;
    float m_mouseSensitivity;
    float m_yaw;
    float m_pitch;

    float m_smoothing = 15.f;
    glm::vec2 m_rawDelta{ 0.f };
    glm::vec2 m_smoothDelta{ 0.f };

    glm::vec2 m_mouseDelta{ 0.f };
};