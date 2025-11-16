#pragma once
#include "interfaces/window_input_i.h"

#include "interfaces/scene_i.h"
#include "interfaces/entity_i.h"
#include "entities/player.h"
#include "asset_manager.h"
#include "entities/game_object.h"
#include "vulkan_wrapper/core/renderable_registry.h"
#include "physics/physics_engine.h"
#include "base/engine_scene_base.h"


#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <optional>


// SandboxScene is a game level scene (loads JSON scene file, creates game objects, player spawn, etc)
// SandboxScene uses EngineSceneBase to register objects into the engine

class SandboxScene : public EngineSceneBase {
public:
	SandboxScene(std::shared_ptr<IWindowInput> input,
		Core::AssetManager& assetManager);

	void init() override;
	void update(float dt) override;

	void loadSceneFile(const std::string& fileName);

	// Camera
	SandboxCamera& getCamera();
	std::pair<glm::mat4, glm::mat4> getMainCameraMatrices() const;

	// Skybox name (configured when loading scenes)
	std::string getSkyboxCubemapName() const {
		return m_skyboxCubemapName;
	}

private:
	std::shared_ptr<IWindowInput> m_pInput;
	Core::AssetManager& m_assetManager;

	std::vector<std::shared_ptr<SandboxPlayer>> m_players;


	glm::vec3 m_initialCameraPosition{ 0.f };
	glm::vec3 m_initialCameraRotation{ 0.f };

	glm::vec3 m_initialPlayerPosition{ 0.f };
	glm::vec3 m_initialPlayerRotation{ 0.f };
	float m_initialPlayerFov = 80.f;
	float m_initialPlayerSensitivity = 0.15f;
	float m_initialPlayerMoveSpeed = 4.0f;

	std::optional<uint32_t> m_skyboxId;
	std::shared_ptr<IGameObject> m_skyboxObject;
	std::string m_skyboxCubemapName = "skybox_hdr";

	bool m_bIsObj = false;
	bool m_bIsGltf = false;

	void attachRenderable(
		std::shared_ptr<IGameObject> go,
		RenderableType overrideType = RenderableType::None
	);


};

