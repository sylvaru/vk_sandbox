#pragma once
#include  <memory>
#include <glm/glm.hpp>
#include <json.hpp>

struct IGameObject;
struct IGameFactory {
    virtual std::shared_ptr<IGameObject> createGameObject(const nlohmann::json& objJson) = 0;
       virtual std::shared_ptr<IGameObject> makePointLight(
        float intensity,
        float radius,
        const glm::vec3& color
    ) = 0;


    virtual std::shared_ptr<IGameObject> createPlayer(
        std::shared_ptr<IWindowInput> input,
        const glm::vec3& startPos,
        const glm::vec3& startRot,
        float fov,
        float sensitivity,
        float moveSpeed) = 0;
};