// player_input.cpp
#include "input/player_input.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <algorithm>

SandboxMNKController::SandboxMNKController(float moveSpeed, float mouseSensitivity)
    : m_moveSpeed(moveSpeed),
    m_mouseSensitivityX(mouseSensitivity),
    m_mouseSensitivityY(mouseSensitivity),
    m_mode(CameraSmoothingMode::Exponential),
    m_resp(18.0f),
    m_springOmega(45.0f),
    m_springZeta(1.0f),
    m_targetYaw(-90.0f),
    m_targetPitch(0.0f),
    m_currentYaw(-90.0f),
    m_currentPitch(0.0f)
{
}

void SandboxMNKController::mouseCallback(glm::vec2 delta) {
    // accumulate raw deltas in case multiple events happen between frames
    // clamp huge jumps which commonly occur on alt-tab or focus events
    if (std::abs(delta.x) > m_maxDeltaClamp || std::abs(delta.y) > m_maxDeltaClamp) {
        // ignore a single enormous sample — treat as lost focus event
        return;
    }
    m_rawDelta += delta;
}

float SandboxMNKController::calcExpAlpha(float omega, float dt) const {
    // alpha = 1 - exp(-omega * dt)
    // safe for dt small and large; frame-rate independent
    return 1.0f - std::expf(-omega * dt);
}

glm::vec2 SandboxMNKController::lowpassRawDelta(const glm::vec2& raw, float dt) {
    // Simple single-pole low-pass (stateful). Cutoff interpreted as "responsiveness".
    // alpha_lp = 1 - exp(-2 * PI * fc * dt)
    float fc = std::max(1e-3f, m_lowpassCutoff);
    float alpha = 1.0f - std::expf(-2.0f * glm::pi<float>() * fc * dt);

    m_lowpassState += (raw - m_lowpassState) * alpha;
    return m_lowpassState;
}

void SandboxMNKController::reset() {
    m_rawDelta = glm::vec2(0.f);
    m_lowpassState = glm::vec2(0.f);
    m_angleVel = glm::vec2(0.f);
    m_targetYaw = m_currentYaw;
    m_targetPitch = m_currentPitch;
    m_firstFrameAfterReset = true;
}

void SandboxMNKController::update(float dt, std::shared_ptr<IWindowInput> input, TransformComponent& transform) {
    if (!input || dt <= 0.0f) return;

    // ---------------------------------------------------------------------
    // 1) Preprocess raw input: clamp, lowpass, deadzone.
    // ---------------------------------------------------------------------
    glm::vec2 raw = m_rawDelta;

    // optional deadzone: ignore tiny movements
    if (glm::length(raw) < m_deadzone) {
        raw = glm::vec2(0.f);
    }

    // apply low-pass filtering to remove micro-jitter
    glm::vec2 filtered = lowpassRawDelta(raw, dt);

    // convert pixels -> degrees using per-axis sensitivity
    glm::vec2 scaledDelta;
    scaledDelta.x = filtered.x * m_mouseSensitivityX;
    scaledDelta.y = filtered.y * m_mouseSensitivityY;

    // optional: clamp per-frame delta magnitude to avoid teleporting camera on spurious events
    if (std::abs(scaledDelta.x) > m_maxDeltaClamp) scaledDelta.x = (scaledDelta.x > 0.0f) ? m_maxDeltaClamp : -m_maxDeltaClamp;
    if (std::abs(scaledDelta.y) > m_maxDeltaClamp) scaledDelta.y = (scaledDelta.y > 0.0f) ? m_maxDeltaClamp : -m_maxDeltaClamp;

    // Reset raw accumulation for next frame
    m_rawDelta = glm::vec2(0.f);

    // Update target angles immediately (so flicks are captured)
    m_targetYaw += scaledDelta.x;
    m_targetPitch += -scaledDelta.y; // invert Y by default (common FPS behavior)

    // Enforce hard clamp on pitch target to avoid flipping
    m_targetPitch = glm::clamp(m_targetPitch, -89.0f, 89.0f);

    // If we're the first frame after a reset (cursor lock toggle), snap targets so we don't jump
    if (m_firstFrameAfterReset) {
        m_currentYaw = m_targetYaw;
        m_currentPitch = m_targetPitch;
        m_angleVel = glm::vec2(0.f);
        m_firstFrameAfterReset = false;
    }

    // ---------------------------------------------------------------------
    // 2) Apply smoothing based on selected mode
    // ---------------------------------------------------------------------
    switch (m_mode) {
    case CameraSmoothingMode::None:
    {
        m_currentYaw = m_targetYaw;
        m_currentPitch = m_targetPitch;
    }
    break;

    case CameraSmoothingMode::Exponential:
    {
        float alpha = calcExpAlpha(m_resp, dt);
        m_currentYaw += (m_targetYaw - m_currentYaw) * alpha;
        m_currentPitch += (m_targetPitch - m_currentPitch) * alpha;
    }
    break;

    case CameraSmoothingMode::CriticallyDampedSpring:
    {
        // Using a second-order critical-damped spring:
        // x'' + 2*zeta*wn*x' + wn^2*(x - xTarget) = 0
        // Implement semi-implicit Euler: v += a*dt; x += v*dt
        float wn = m_springOmega;
        float z = m_springZeta;

        glm::vec2 x(m_currentYaw, m_currentPitch);
        glm::vec2 xT(m_targetYaw, m_targetPitch);
        glm::vec2 v = m_angleVel;

        // acceleration
        glm::vec2 a = -2.0f * z * wn * v - (wn * wn) * (x - xT);

        // integrate velocity & position
        v += a * dt;
        x += v * dt;

        m_angleVel = v;
        m_currentYaw = x.x;
        m_currentPitch = glm::clamp(x.y, -89.0f, 89.0f);
    }
    break;
    }

    // ---------------------------------------------------------------------
    // 3) Apply to transform and compute movement basis
    // ---------------------------------------------------------------------
    transform.rotation = glm::vec3(glm::radians(m_currentPitch), glm::radians(m_currentYaw), 0.0f);

    // Movement basis computed from current smoothed angles
    float pitchDeg = m_currentPitch;
    float yawDeg = m_currentYaw;
    glm::vec3 front{
        std::cos(glm::radians(yawDeg)) * std::cos(glm::radians(pitchDeg)),
        std::sin(glm::radians(pitchDeg)),
        std::sin(glm::radians(yawDeg)) * std::cos(glm::radians(pitchDeg))
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
        const float sprintMultiplier = input->isKeyPressed(SandboxKey::LEFT_SHIFT) ? 3.0f : 1.0f;
        float speed = m_moveSpeed * sprintMultiplier;
        transform.translation += dir * speed * dt;
    }

    // end update
}
