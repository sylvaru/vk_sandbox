// IGameObject.h
#pragma once
#include "global_common/transform_component.h"
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
enum class RenderTag : uint8_t {
    Auto = 0,     // let renderer infer (OBJ/glTF/PointLight/etc.)
    Skybox,       // explicit
    Obj,
    Gltf,
    PointLight,
    Scene         // <-- NEW: route to SceneRenderSystem
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

    virtual RenderTag getPreferredRenderTag() const { return RenderTag::Auto; }
};
