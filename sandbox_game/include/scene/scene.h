#pragma once
#include "interfaces/window_input_i.h"

#include "interfaces/scene_i.h"
#include "interfaces/entity_i.h"
#include "entities/player.h"
#include "asset_manager.h"
#include "entities/game_object.h"
#include "vulkan_wrapper/core/renderable_registry.h"

#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <optional>

class SandboxScene : public IScene {
public:
	SandboxScene(std::shared_ptr<IWindowInput> input, Core::AssetManager& assetManager);      // pass input so your Player can read it
	void init() override;                 // load models, spawn entities
	void update(float dt) override;        // advance all entities

	void loadSceneFile(const std::string& fileName);

	std::unordered_map<unsigned int, std::shared_ptr<IGameObject>>&
		getGameObjects() override { return m_gameObjects; }


	std::pair<glm::mat4, glm::mat4> getMainCameraMatrices()const;

	void setSkyboxObject(std::shared_ptr<IGameObject> obj) {
		m_skyboxId = obj->getId();
		m_skyboxObject = std::move(obj);
	}

	SandboxCamera& getCamera();

	void addGameObject(uint32_t id, SandboxGameObject obj);
	void removeGameObject(uint32_t id);

	std::optional<std::reference_wrapper<IGameObject>>
		getSkyboxObject() const override;

	std::optional<std::reference_wrapper<SandboxGameObject>>
		getSkyboxObject();

	std::string getSkyboxCubemapName() const {
		return m_skyboxCubemapName;
	}
	const RenderableRegistry* getRenderableRegistry() const override {
		return &m_renderRegistry;
	}

private:
	std::shared_ptr<IWindowInput> m_pInput;
	Core::AssetManager& m_assetManager;

	std::vector<std::shared_ptr<SandboxPlayer>> m_players;
	std::unordered_map<uint32_t, std::shared_ptr<IGameObject>>  m_gameObjects;

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

	RenderableRegistry m_renderRegistry;
	std::unordered_map<uint32_t, RenderableID> m_goRenderable;
	RenderableID createRenderableForGameObject(
		uint32_t gameObjectId,
		uint32_t meshIndex, uint32_t materialIndex,
		const TransformData& t,
		const glm::vec3& bsCenter, float bsRadius,
		RenderableType type);

	void removeRenderableForGameObject(uint32_t gameObjectId);
};