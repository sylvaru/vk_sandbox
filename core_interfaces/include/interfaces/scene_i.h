// IScene.h
#pragma once

#include <unordered_map>
#include <memory>
#include "interfaces/game_object_i.h"
#include "interfaces/camera_i.h"
#include "interfaces/renderer_i.h"   // for FrameContext
#include <optional>
#include <string>

class RenderableRegistry;
class ICamera;
struct IGameObject;

struct IScene {
    virtual ~IScene() = default;


    virtual void init() = 0;


    virtual void update(float deltaTime) = 0;
    
    virtual ICamera& getCamera() = 0;

    virtual std::unordered_map<unsigned int, std::shared_ptr<IGameObject>>&
        getGameObjects() = 0;

    virtual std::optional<std::reference_wrapper<IGameObject>>
        getSkyboxObject() const
    {
        return std::nullopt;
    }
    virtual const RenderableRegistry* getRenderableRegistry() const { return nullptr; }
    virtual std::string getSkyboxCubemapName() const = 0;
};
