#include "common/game_pch.h"
#include "camera/camera.h"

SandboxCamera::SandboxCamera(glm::vec3 position, float yawDeg, float pitchDeg, float zoomDeg)
    : m_position(position), m_zoom(zoomDeg)
{
    // Convert yaw/pitch into quaternion
    glm::vec3 eulerRad = glm::radians(glm::vec3(pitchDeg, yawDeg, 0.f));
    m_orientation = glm::quat(eulerRad);

    updateVectors();
    updateView();
}

void SandboxCamera::updateVectors() {
    // Rotate basis vectors via quaternion
    m_forward = glm::normalize(m_orientation * glm::vec3(0.f, 0.f, -1.f));
    m_right = glm::normalize(m_orientation * glm::vec3(1.f, 0.f, 0.f));
    m_up = glm::normalize(glm::cross(m_right, m_forward * -1.f));
}

void SandboxCamera::updateView() {
    // Build view matrix from quaternion and position
    glm::mat4 rot = glm::toMat4(glm::conjugate(m_orientation));
    glm::mat4 trans = glm::translate(glm::mat4(1.f), -m_position);

    m_viewMatrix = rot * trans;
    m_inverseViewMatrix = glm::inverse(m_viewMatrix);
}

void SandboxCamera::updateProjection(float aspect, float nearZ, float farZ) {
    m_projMatrix = glm::perspective(glm::radians(m_zoom), aspect, nearZ, farZ);
    m_projMatrix[1][1] *= -1.f; // Vulkan Y flip
}

void SandboxCamera::move(glm::vec3 delta) {
    m_position += delta;
    updateView();
}

void SandboxCamera::rotate(float yawOffsetDeg, float pitchOffsetDeg) {

    float yawRad = glm::radians(yawOffsetDeg);
    float pitchRad = glm::radians(pitchOffsetDeg);

    glm::quat qYaw = glm::angleAxis(yawRad, glm::vec3(0.f, 1.f, 0.f));
    glm::quat qPitch = glm::angleAxis(pitchRad, m_right);

    m_orientation = glm::normalize(qYaw * qPitch * m_orientation);

    updateVectors();
    updateView();
}



void SandboxCamera::setRotation(float pitchDeg, float yawDeg)
{
    glm::quat qYaw = glm::angleAxis(glm::radians(yawDeg), glm::vec3(0, 1, 0));
    glm::quat qPitch = glm::angleAxis(glm::radians(pitchDeg), glm::vec3(1, 0, 0));

    m_orientation = glm::normalize(qYaw * qPitch);

    updateVectors();
    updateView();
}

void SandboxCamera::setOrientation(const glm::quat& q) {
    m_orientation = q;
    updateVectors();
    updateView();
}

