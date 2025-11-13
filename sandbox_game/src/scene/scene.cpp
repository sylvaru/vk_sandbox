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
    spdlog::info("Initializing SandboxScene...");

    auto player = std::make_shared<SandboxPlayer>(
        m_pInput,
        m_initialPlayerPosition,
        m_initialPlayerRotation,
        m_initialPlayerFov,
        m_initialPlayerSensitivity,
        m_initialPlayerMoveSpeed
    );

    player->onInit();

    m_players.push_back(player);
    m_gameObjects[player->getId()] = player;

    spdlog::info("Spawned player at ({:.2f}, {:.2f}, {:.2f}), rot (deg) ({:.2f}, {:.2f}, {:.2f}), fov {:.1f}, sens {:.2f}, speed {:.2f}",
        m_initialPlayerPosition.x, m_initialPlayerPosition.y, m_initialPlayerPosition.z,
        glm::degrees(m_initialPlayerRotation.x),
        glm::degrees(m_initialPlayerRotation.y),
        glm::degrees(m_initialPlayerRotation.z),
        m_initialPlayerFov, m_initialPlayerSensitivity, m_initialPlayerMoveSpeed);

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

    // Player setup
    if (sceneJson.contains("player")) {
        const auto& playerJson = sceneJson["player"];

        auto pos = playerJson.value("position", std::vector<float>{0.f, 0.f, 0.f});
        auto rot = playerJson.value("rotation", std::vector<float>{0.f, 0.f, 0.f});

        m_initialPlayerPosition = { pos[0], pos[1], pos[2] };
        m_initialPlayerRotation = {
            glm::radians(rot[0]),
            glm::radians(rot[1]),
            glm::radians(rot[2])
        };

        m_initialPlayerFov = playerJson.value("fov", 75.f);
        m_initialPlayerSensitivity = playerJson.value("sensitivity", 0.09f);
        m_initialPlayerMoveSpeed = playerJson.value("move_speed", 1.f);

        spdlog::info(
            "Loaded player config: pos ({}, {}, {}), rot (deg) ({}, {}, {}), FOV {}, sens {}, speed {}",
            pos[0], pos[1], pos[2], rot[0], rot[1], rot[2],
            m_initialPlayerFov, m_initialPlayerSensitivity, m_initialPlayerMoveSpeed
        );
    }


    // Clear previous objects before loading new
    m_gameObjects.clear();

    for (auto& objJson : sceneJson["objects"]) {

        if (objJson.value("special", "") == "lights") {
            int count = objJson.value("count", 1);
            float orbitRadius = objJson.value("orbit_radius", 5.0f);
            float height = objJson.value("height", 3.5f);
            float intensity = objJson.value("intensity", 0.092f);
            float lightRadius = objJson.value("light_radius", 0.1f);
            const auto& colorsJson = objJson["colors"];

            for (int i = 0; i < count; ++i) {
                float angle = i * glm::two_pi<float>() / count;
                glm::vec3 pos = {
                    orbitRadius * std::cos(angle),
                    height,
                    orbitRadius * std::sin(angle)
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

                uint32_t id = light->getId();
                m_gameObjects.emplace(id, std::move(light));

                auto it = m_gameObjects.find(id);
                if (it == m_gameObjects.end()) continue;
                auto& goRef = *it->second;

                TransformData t{};
                t.model = goRef.getTransform().mat4();
                t.normalMat = glm::transpose(glm::inverse(t.model));

                RenderableID rid = m_renderRegistry.createInstance(
                    0, 0, t, glm::vec3(0.0f),
                    lightRadius, // visual size
                    RenderableType::Light
                );

                if (MeshInstance* inst = m_renderRegistry.getInstanceMutable(rid)) {
                    inst->emissiveColor = color;
                    inst->intensity = intensity;
                    inst->boundingSphereRadius = lightRadius;
                }
            }

            continue;
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
        gameObject->m_transform.rotation = {
            glm::radians(rot[0]),
            glm::radians(rot[1]),
            glm::radians(rot[2])
        };
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
        RenderableType rtype = RenderableType::None;

        // --- Read "renderSystem" or "render_system" from JSON ---
        auto rs = getStr(objJson, "renderSystem");
        if (!rs) rs = getStr(objJson, "render_system");

        if (rs) {
            const std::string v = *rs;

            if (v == "scene") { tag = RenderTag::Scene;    rtype = RenderableType::Scene; }
            else if (v == "gltf") { tag = RenderTag::Gltf;     rtype = RenderableType::Gltf; }
            else if (v == "obj") { tag = RenderTag::Obj;      rtype = RenderableType::Obj; }
            else if (v == "skybox") { tag = RenderTag::Skybox;   rtype = RenderableType::Skybox; }
            else if (v == "light") { tag = RenderTag::PointLight; rtype = RenderableType::Light; }
            else { tag = RenderTag::Auto;     rtype = RenderableType::None; }

            spdlog::info("Render tag override for object '{}': {}", objJson.value("name", "unnamed"), v);
        }
        else {
            if (isSkybox) {
                tag = RenderTag::Skybox;
                rtype = RenderableType::Skybox;
            }
            else if (gameObject->getPointLight()) {
                tag = RenderTag::PointLight;
                rtype = RenderableType::Light;
            }
            else if (m_bIsGltf) {
                tag = RenderTag::Gltf;
                rtype = RenderableType::Gltf;
            }
            else if (m_bIsObj) {
                tag = RenderTag::Obj;
                rtype = RenderableType::Obj;
            }
        }


        gameObject->setRenderTag(tag);


        // Insert into map (store as base interface)
        m_gameObjects.emplace(gameObject->getId(), std::static_pointer_cast<IGameObject>(gameObject));



        if (auto model = gameObject->getModel()) {
            TransformData t{};
            t.model = gameObject->m_transform.mat4();
            t.normalMat = glm::transpose(glm::inverse(t.model));

            uint32_t meshIndex = 0;
            uint32_t materialIndex = 0;

            createRenderableForGameObject(gameObject->getId(), meshIndex, materialIndex, t, glm::vec3{ 0.0f }, 1.0f, rtype);
        }
    }


    spdlog::info("Scene '{}' loaded. Total objects: {}", fileName, m_gameObjects.size());
}
void SandboxScene::addRigidBody(btRigidBody* body) {
	//m_physicsWorld->addRigidBody(body);
}

RenderableID SandboxScene::createRenderableForGameObject(
    uint32_t gameObjectId,
    uint32_t meshIndex,
    uint32_t materialIndex,
    const TransformData& transform,
    const glm::vec3& bsCenter,
    float bsRadius,
    RenderableType type)
{
    // Register a new renderable in the global registry
    RenderableID rid = m_renderRegistry.createInstance(
        meshIndex, materialIndex, transform, bsCenter, bsRadius, type);

    // Remember which renderable belongs to which object
    m_goRenderable[gameObjectId] = rid;

    // Assign the correct model pointer depending on renderable type
    if (auto go = m_gameObjects[gameObjectId]) {
        if (auto base = go->getModel()) {
            switch (type) {
            case RenderableType::Gltf:
            case RenderableType::Skybox:
            case RenderableType::Light:
            case RenderableType::Scene:

            {
                if (auto gltf = std::dynamic_pointer_cast<vkglTF::Model>(base)) {
                    m_renderRegistry.setModelPointer(rid, gltf.get());
                }
                break;
            }
            // (Optional: add more cases for Obj or custom model types later)
            default:
                break;
            }
        }
    }

    return rid;
}


void SandboxScene::removeRenderableForGameObject(uint32_t gameObjectId)
{
    auto it = m_goRenderable.find(gameObjectId);
    if (it != m_goRenderable.end()) {
        m_renderRegistry.removeInstance(it->second);
        m_goRenderable.erase(it);
    }
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