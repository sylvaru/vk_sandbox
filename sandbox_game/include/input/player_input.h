#pragma once
#include "interfaces/window_i.h"
#include <glm/glm.hpp>
#include <memory>
#include <functional>
#include <physics/physics_engine.h>
#include "camera/camera.h"

class SandboxMNKController {
public:
    SandboxMNKController(
        const glm::vec3& startPos,
        float yawDeg,
        float pitchDeg,
        float fov,
        float moveSpeed,
        float sensitivity,
        PhysicsEngine* physics);

    void update(float dt, IWindow& window);
    void mouseCallback(glm::vec2 delta);
    void setMoveSpeed(float s) { m_moveSpeed = s; }
    void setMouseSensitivity(float s) { m_mouseSensitivity = s; }
    void setOrientation(float yawDeg, float pitchDeg) { m_yaw = yawDeg; m_pitch = pitchDeg; }
    void setYawPitch(float yawDeg, float pitchDeg);

    float getYaw() const { return m_yaw; }
    float getPitch() const { return m_pitch; }

    void setPhysicsController(physx::PxCapsuleController* ctrl) { m_controller = ctrl; }
    SandboxCamera& getCamera() { return m_camera; }

private:
    float m_moveSpeed;
    float m_mouseSensitivity;
    float m_yaw;
    float m_pitch;

    float m_smoothing = 15.f;
    glm::vec2 m_rawDelta{ 0.f };
    glm::vec2 m_smoothDelta{ 0.f };

    glm::vec2 m_mouseDelta{ 0.f };
    PhysicsEngine* m_physics = nullptr;
    physx::PxCapsuleController* m_controller = nullptr;

    SandboxCamera m_camera;
};