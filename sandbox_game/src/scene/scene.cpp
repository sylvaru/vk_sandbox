#include "scene/scene.h"
#include "entities/player.h"
#include "entities/game_object.h"

#include <json.hpp>


#include <glm/gtc/constants.hpp>
#include <spdlog/spdlog.h>


using json = nlohmann::json;

SandboxScene::SandboxScene(std::shared_ptr<IWindowInput> input, Core::AssetManager& assetManager)
	: m_pInput(std::move(input)), m_assetManager(assetManager) 
{
}

void SandboxScene::init() {
    auto player = std::make_shared<SandboxPlayer>(m_pInput);
    
    player->getTransform().translation = m_initialCameraPosition;
    player->getTransform().rotation = m_initialCameraRotation;
    player->onInit();

    m_players.push_back(player);

}

void SandboxScene::update(float dt) {
    for (auto& player : m_players) {
        player->onUpdate(dt);
    }

    for (auto& [id, obj] : m_gameObjects) {
        obj->onUpdate(dt);
    }
}
void SandboxScene::loadSceneFile(const std::string& fileName) {
    std::string path = std::string(PROJECT_ROOT_DIR) + "/sandbox_game/res/scenes/" + fileName + ".json";

    std::ifstream inFile(path);
    if (!inFile.is_open()) {
        throw std::runtime_error("Could not open scene file: " + path);
    }

    json sceneJson;
    inFile >> sceneJson;

    m_skyboxCubemapName.clear();

    spdlog::info("Loading scene file: {} ({})", fileName, path);

    // Camera setup
    if (sceneJson.contains("camera")) {
        const auto& camJson = sceneJson["camera"];
        auto pos = camJson.value("position", std::vector<float>{0.f, 0.f, 0.f});
        auto rot = camJson.value("rotation", std::vector<float>{0.f, 0.f, 0.f});

        m_initialCameraPosition = { pos[0], pos[1], pos[2] };
        m_initialCameraRotation = {
            glm::radians(rot[0]),
            glm::radians(rot[1]),
            glm::radians(rot[2])
        };

        spdlog::info("Camera position: ({}, {}, {}), rotation (deg): ({}, {}, {})",
            pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]);
    }

    // Clear previous objects before loading new
    m_gameObjects.clear();

    for (auto& objJson : sceneJson["objects"]) {

        // Special spinning lights case
        if (objJson.value("special", "") == "lights") {
            int count = objJson.value("count", 1);
            float radius = objJson.value("radius", 4.8f);
            float height = objJson.value("height", -2.5f);
            float intensity = objJson.value("intensity", 15.8f);
            const auto& colorsJson = objJson["colors"];

            for (int i = 0; i < count; ++i) {
                float angle = i * glm::two_pi<float>() / count;
                glm::vec3 pos = {
                    radius * std::cos(angle),
                    height,
                    radius * std::sin(angle)
                };

                const auto& colorArray = colorsJson[i % colorsJson.size()];
                glm::vec3 color = {
                    colorArray[0].get<float>(),
                    colorArray[1].get<float>(),
                    colorArray[2].get<float>()
                };

                auto light = SandboxGameObject::makePointLight(intensity, 0.1f, color);
                light->setRenderTag(RenderTag::PointLight);
                light->getTransform().translation = pos;

                spdlog::info("Placed point light at ({}, {}, {})", pos.x, pos.y, pos.z);

                m_gameObjects.emplace(light->getId(), std::move(light));
            }

            continue; // skip normal parsing for this object
        }

        // Normal GameObject
        auto gameObject = SandboxGameObject::createGameObject();

        // Model loading
        if (auto it = objJson.find("model"); it != objJson.end()) {
            const std::string modelName = it->get<std::string>();

            // Try OBJ model first
            if (auto objModel = m_assetManager.getOBJModel(modelName)) {
                gameObject->setModel(objModel);
                m_bIsObj = true;
            }
            // Otherwise try glTF model
            else if (auto gltfModel = m_assetManager.getGLTFmodel(modelName)) {
                gameObject->setModel(gltfModel);
                m_bIsGltf = true;
            }
            else {
                throw std::runtime_error("Model not found in cache: " + modelName);
            }
        }

        // Transform
        auto pos = objJson.value("position", std::vector<float>{0.f, 0.f, 0.f});
        auto rot = objJson.value("rotation", std::vector<float>{0.f, 0.f, 0.f});
        auto scl = objJson.value("scale", std::vector<float>{1.f, 1.f, 1.f});

        gameObject->m_transform.translation = { pos[0], pos[1], pos[2] };
        gameObject->m_transform.rotation = { rot[0], rot[1], rot[2] };
        gameObject->m_transform.scale = { scl[0], scl[1], scl[2] };

        spdlog::info("Loaded GameObject '{}' - Pos: ({}, {}, {}), Rot: ({}, {}, {}), Scale: ({}, {}, {})",
            objJson.value("name", "unnamed"),
            pos[0], pos[1], pos[2],
            rot[0], rot[1], rot[2],
            scl[0], scl[1], scl[2]);

        // Check skybox usage
        bool isSkybox = false;
        if (auto usageIt = objJson.find("usage"); usageIt != objJson.end()) {
            if (usageIt->get<std::string>() == "skybox") {
                isSkybox = true;
            }
        }
        gameObject->m_bIsSkybox = isSkybox;

        // Optional cubemap texture name
        if (objJson.contains("cubemap")) {
            gameObject->m_cubemapTextureName = objJson["cubemap"].get<std::string>();
        }

        // If skybox, set as the scene's skybox object
        if (isSkybox) {
            setSkyboxObject(gameObject);
            spdlog::info("GameObject '{}' marked as skybox with cubemap '{}'", objJson.value("name", "unnamed"), gameObject->m_cubemapTextureName);
        }


        auto getStr = [&](const json& j, const char* k) -> std::optional<std::string> {
            if (auto it = j.find(k); it != j.end() && it->is_string()) return it->get<std::string>();
            return std::nullopt;
            };

        RenderTag tag = RenderTag::Auto;
        auto rs = getStr(objJson, "renderSystem");
        if (!rs) rs = getStr(objJson, "render_system"); // alias

        if (rs) {
            const std::string v = *rs;
            if (v == "scene")                                 tag = RenderTag::Scene;
            else if (v == "gltf")                             tag = RenderTag::Gltf;
            else if (v == "obj")                              tag = RenderTag::Obj;
            else if (v == "skybox")                           tag = RenderTag::Skybox;
            else                                              tag = RenderTag::Auto;

            spdlog::info("Render tag override for object '{}': {}",
                objJson.value("name", "unnamed"), v);
        }
        else {
            // --- No explicit tag: infer like the old behavior ---
            if (isSkybox)    tag = RenderTag::Skybox;
            else if (gameObject->getPointLight()) tag = RenderTag::PointLight; // for "special: lights"
            else if (m_bIsGltf) tag = RenderTag::Gltf;
            else if (m_bIsObj)  tag = RenderTag::Obj;
            // else Auto stays, if you have other systems
        }

        gameObject->setRenderTag(tag);


        // Insert into map (store as base interface)
        m_gameObjects.emplace(gameObject->getId(), std::static_pointer_cast<IGameObject>(gameObject));
    }

    spdlog::info("Scene '{}' loaded. Total objects: {}", fileName, m_gameObjects.size());
}


std::optional<std::reference_wrapper<SandboxGameObject>> SandboxScene::getSkyboxObject() {
    if (!m_skyboxId) return std::nullopt;
    auto it = m_gameObjects.find(*m_skyboxId);
    if (it != m_gameObjects.end()) {
        // cast back from IGameObject→SandboxGameObject
        return std::reference_wrapper(
            static_cast<SandboxGameObject&>(*it->second));
    }
    return std::nullopt;
}

// Implements the IScene interface:
std::optional<std::reference_wrapper<IGameObject>>
SandboxScene::getSkyboxObject() const {
    if (!m_skyboxId) {
        return std::nullopt;
    }
    auto it = m_gameObjects.find(*m_skyboxId);
    if (it == m_gameObjects.end()) {
        return std::nullopt;
    }
    // we know it really is a SandboxGameObject, but expose it as IGameObject
    return std::make_optional<std::reference_wrapper<IGameObject>>(
        *it->second
    );
}

SandboxCamera& SandboxScene::getCamera() {
    if (m_players.empty()) {
        throw std::runtime_error("no players available to get camera from");
    }

    auto* player = dynamic_cast<SandboxPlayer*>(m_players[0].get());
    if (!player) {
        throw std::runtime_error("first player is not a SandboxPlayer");
    }

    return player->getCamera();
}