// IGameObject.h
#pragma once
#include "transform_component.h"
#include "renderer_i.h"
#include "interfaces/model_i.h"
#include <memory>
#include <optional>
#include <glm/vec3.hpp>
#include <string>

struct IModel;


struct PointLightComponent
{
    float lightIntensity = 1.0f;
};


struct IGameObject {
    virtual ~IGameObject() = default;

    virtual void onInit() {}
    virtual void onUpdate(float deltaTime) {}

    virtual TransformComponent& getTransform() = 0;
    virtual std::shared_ptr<IModel> getModel() const = 0;

    virtual glm::vec3 getColor() const { return glm::vec3(1.f); }
    virtual const PointLightComponent* getPointLight() const { return nullptr; }
    virtual uint32_t getId() const { return 0; }
    virtual std::string getCubemapTextureName() const { return ""; }
};
