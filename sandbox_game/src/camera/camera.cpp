#include "camera/camera.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

SandboxCamera::SandboxCamera(glm::vec3 position, float yawDeg, float pitchDeg, float zoomDeg)
    : m_position(position),
    m_worldUp(0.f, 1.f, 0.f),
    m_yaw(yawDeg),
    m_pitch(pitchDeg),
    m_zoom(zoomDeg)
{
    updateVectors();
    updateView();
}

void SandboxCamera::updateVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_front = glm::normalize(front);

    // Recalculate Right and Up vector
    m_right = glm::normalize(glm::cross(m_front, m_worldUp));  // Right vector
    m_up = glm::normalize(glm::cross(m_right, m_front));
}

void SandboxCamera::updateView() {
    m_viewMatrix = glm::lookAt(m_position, m_position + m_front, m_up);
    m_inverseViewMatrix = glm::inverse(m_viewMatrix);
}

void SandboxCamera::updateProjection(float aspect, float nearZ, float farZ) {
    m_projMatrix = glm::perspective(glm::radians(m_zoom), aspect, nearZ, farZ);
    m_projMatrix[1][1] *= -1; // Vulkan Y-flip
}

void SandboxCamera::move(glm::vec3 delta) {
    m_position += delta;
    updateView();
}

void SandboxCamera::rotate(float yawOffset, float pitchOffset) {
    m_yaw += yawOffset;
    m_pitch += pitchOffset;

    m_pitch = glm::clamp(m_pitch, -89.f, 89.f);
    updateVectors();
    updateView();
}

void SandboxCamera::setZoom(float zoom) {
    m_zoom = glm::clamp(zoom, 1.f, 120.f);
}

void SandboxCamera::setRotation(glm::vec3 eulerRad) {
    m_pitch = glm::degrees(eulerRad.x);
    m_yaw = glm::degrees(eulerRad.y);
    updateVectors();
    updateView();
}
