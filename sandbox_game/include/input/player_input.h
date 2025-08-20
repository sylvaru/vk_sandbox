#pragma once

#include "interfaces/window_input_i.h"
#include "transform_component.h"
#include <glm/glm.hpp>
#include <memory>
#include <functional>

enum class CameraSmoothingMode {
    None = 0,
    Exponential = 1,
    CriticallyDampedSpring = 2
};

class SandboxMNKController {
public:
    SandboxMNKController(float moveSpeed = 7.f, float mouseSensitivity = 0.08f);

    // Main per-frame update
    void update(float dt, std::shared_ptr<IWindowInput> input, TransformComponent& transform);

    // Feed raw mouse delta (pixels). Call this from your glfw cursor callback or poll loop.
    void mouseCallback(glm::vec2 delta);

    // Reset internal state (use when locking/unlocking cursor or teleporting camera)
    void reset();

    // Setters for runtime tuning
    void setMouseSensitivity(float sens) { m_mouseSensitivityX = m_mouseSensitivityY = sens; }
    void setMouseSensitivityXY(float sensX, float sensY) { m_mouseSensitivityX = sensX; m_mouseSensitivityY = sensY; }
    void setResponse(float resp) { m_resp = resp; } // for exponential
    void setSmoothingMode(CameraSmoothingMode mode) { m_mode = mode; }
    void setSpringParams(float omega, float zeta) { m_springOmega = omega; m_springZeta = zeta; }
    void setDeadzone(float dz) { m_deadzone = dz; }
    void setMaxDeltaClamp(float maxDelta) { m_maxDeltaClamp = maxDelta; }

    // Query angles (degrees)
    float getYaw() const { return m_currentYaw; }
    float getPitch() const { return m_currentPitch; }

private:
    // core movement
    float m_moveSpeed;

    // RAW -> angle conversion
    float m_mouseSensitivityX; // degrees per pixel X
    float m_mouseSensitivityY; // degrees per pixel Y

    // smoothing mode & parameters
    CameraSmoothingMode m_mode = CameraSmoothingMode::CriticallyDampedSpring;

    // Exponential smoothing (frame-rate independent)
    float m_resp = 18.0f; // responsiveness (omega) used in alpha = 1 - exp(-omega * dt)

    // Spring (2nd-order) parameters
    float m_springOmega = 45.0f; // natural frequency (rad/s) — higher => snappier
    float m_springZeta = 1.0f;   // damping ratio (~1.0 = critical damping)

    // angle state (degrees)
    float m_targetYaw = -90.0f;
    float m_targetPitch = 0.0f;
    float m_currentYaw = -90.0f;
    float m_currentPitch = 0.0f;

    // velocity for spring (degrees/sec)
    glm::vec2 m_angleVel{ 0.0f, 0.0f }; // (yawVel, pitchVel)

    // raw/prefiltering
    glm::vec2 m_rawDelta{ 0.f };     // <-- written by mouseCallback()
    glm::vec2 m_lowpassState{ 0.f }; // stateful lowpass on raw delta
    float m_lowpassCutoff = 60.0f;   // Hz-ish feel; higher => passes more high-frequency
    float m_deadzone = 0.0f;         // tiny deadzone to avoid tiny jitter
    float m_maxDeltaClamp = 1000.0f; // clamp giant deltas (alt-tab / focus changes)

    // misc internal bookkeeping
    bool m_firstFrameAfterReset = true;

    // small helpers
    float calcExpAlpha(float omega, float dt) const;
    glm::vec2 lowpassRawDelta(const glm::vec2& raw, float dt);
};
