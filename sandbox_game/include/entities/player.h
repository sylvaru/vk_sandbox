#pragma once
#include "interfaces/game_object_i.h"
#include "interfaces/window_input_i.h"
#include "input/player_input.h"
#include "interfaces/camera_i.h"
#include "transform_component.h"
#include "interfaces/renderer_i.h"
#include "physics/physics_engine.h"


#include <glm/glm.hpp>
#include <memory>

class SandboxPlayer : public IGameObject
{
public:
    SandboxPlayer(std::shared_ptr<IWindowInput> input,
        const glm::vec3& startPos,
        const glm::vec3& startRotRad,
        float fov,
        float sensitivity,
        float moveSpeed,
        PhysicsEngine* physics);

    void onInit() override;
    void onUpdate(float deltaTime) override;

    TransformComponent& getTransform() override;

    std::shared_ptr<IModel> getModel() const override;
    SandboxMNKController& getController() { return m_controller; }


private:
    std::shared_ptr<IWindowInput>       m_pInput;
    TransformComponent m_transform;
    SandboxMNKController m_controller;
    PhysicsEngine* m_physics;

    // Configuration
    float m_mouseSensitivity = 0.15f;
    float m_moveSpeed = 4.0f;

    glm::vec3 m_initialPosition;  // Initialized from startPos in constructor
    glm::vec3 m_initialRotation;  // Initialized from startRotRad in constructor (radians)

    // Optional: you can also store other defaults if needed
    float m_initialFov;
    float m_initialSensitivity;
    float m_initialMoveSpeed;

};