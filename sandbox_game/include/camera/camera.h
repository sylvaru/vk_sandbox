#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include "interfaces/camera_i.h"

class SandboxCamera : public ICamera {
public:
    SandboxCamera() = default;
    SandboxCamera(glm::vec3 position, float yawDeg = -90.f, float pitchDeg = 0.f, float zoomDeg = 90.f);

    void updateView();
    void updateProjection(float aspect, float nearZ = 0.1f, float farZ = 100.f);

    void move(glm::vec3 delta);

    // quaternion-based rotate
    void rotate(float yawOffsetDeg, float pitchOffsetDeg);

    inline void setZoom(float zoom) { m_zoom = glm::clamp(zoom, 1.f, 120.f); }
    inline void setPosition(const glm::vec3& pos) { m_position = pos; }

    void setRotation(float pitchDeg, float yawDeg);

    glm::vec3 getForwardVector() const { return m_forward; }
    glm::vec3 getRightVector()   const { return m_right; }
    glm::vec3 getUpVector()      const { return m_up; }

    glm::mat4 getViewMatrix() const override { return m_viewMatrix; }
    glm::mat4 getProjectionMatrix() const override { return m_projMatrix; }

    glm::vec3 getPosition() const override { return m_position; }
    glm::mat4 getInverseViewMatrix() const override { return m_inverseViewMatrix; }
    void setOrientation(const glm::quat& q);
private:
    void updateVectors();

    glm::vec3 m_position{ 0.f };
    glm::quat m_orientation{ 1.f, 0.f, 0.f, 0.f };

    glm::vec3 m_forward{ 0.f, 0.f, -1.f };
    glm::vec3 m_right{ 1.f, 0.f, 0.f };
    glm::vec3 m_up{ 0.f, 1.f, 0.f };

    float m_zoom = 90.f;

    glm::mat4 m_viewMatrix{ 1.f };
    glm::mat4 m_projMatrix{ 1.f };
    glm::mat4 m_inverseViewMatrix{ 1.f };
};
