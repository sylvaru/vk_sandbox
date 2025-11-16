#pragma once
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <unordered_map>
#include <spdlog/spdlog.h>



class PhysicsEngine {
public:
    PhysicsEngine();

    void stepSimulation(float deltaTime);

};