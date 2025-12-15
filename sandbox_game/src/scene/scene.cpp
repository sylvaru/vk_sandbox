#include "common/game_pch.h"
#include "scene/scene.h"
#include "entities/player.h"
#include "entities/game_object.h"


using json = nlohmann::json;

SandboxScene::SandboxScene(IWindow& window, Core::AssetManager& assetManager)
	: EngineSceneBase(&assetManager)
    , m_window(window)
    , m_assetManager(assetManager) 
{
}

void SandboxScene::initSceneData() {

    auto player = std::make_shared<SandboxPlayer>(
        m_window,
        m_initialPlayerPosition,
        m_initialPlayerRotation,
        m_initialPlayerFov,
        m_initialPlayerSensitivity,
        m_initialPlayerMoveSpeed,
        m_physicsEngine
    );


    m_players.push_back(player);
    m_gameObjects[player->getId()] = player;

    init();
}
void SandboxScene::update(float dt) {
    EngineSceneBase::update(dt);
}
void SandboxScene::loadSceneFile(const std::string& fileName)
{
    initPhysics();
    
    clearScene();

    const std::string path =
        std::string(PROJECT_ROOT_DIR) +
        "/sandbox_game/res/scenes/" +
        fileName + ".json";

    std::ifstream inFile(path);
    if (!inFile.is_open())
        throw std::runtime_error("Could not open scene file: " + path);

    json sceneJson;
    inFile >> sceneJson;

    spdlog::info("Loading scene '{}'", fileName);


    // Load player config
    if (sceneJson.contains("player"))
    {
        const auto& j = sceneJson["player"];

        auto pos = j.value("position", std::vector<float>{0, 0, 0});
        auto rot = j.value("rotation", std::vector<float>{0, 0, 0});

        m_initialPlayerPosition = { pos[0], pos[1], pos[2] };
        m_initialPlayerRotation =
        {
            glm::radians(rot[0]),
            glm::radians(rot[1]),
            glm::radians(rot[2])
        };

        m_initialPlayerFov = j.value("fov", 75.f);
        m_initialPlayerSensitivity = j.value("sensitivity", 0.09f);
        m_initialPlayerMoveSpeed = j.value("move_speed", 1.f);

        spdlog::info("Player settings updated from scene file.");
    }

    m_bIsObj = false;
    m_bIsGltf = false;


    // Load all scene objects
    for (auto& objJson : sceneJson["objects"])
    {
       
        // Normal GameObject
        auto go = SandboxGameObject::createGameObject();

        if (auto it = objJson.find("model"); it != objJson.end())
        {
            std::string name = it->get<std::string>();

            if (auto m = m_assetManager.getOBJModel(name))
            {
                go->setModel(m);
                m_bIsObj = true;
            }
            else if (auto m = m_assetManager.getGLTFmodel(name))
            {
                go->setModel(m);
                m_bIsGltf = true;
            }
        }

        // Transform
        auto pos = objJson.value("position", std::vector<float>{0, 0, 0});
        auto rot = objJson.value("rotation", std::vector<float>{0, 0, 0});
        auto scl = objJson.value("scale", std::vector<float>{1, 1, 1});

        go->m_transform.translation = { pos[0], pos[1], pos[2] };
        go->m_transform.rotation =
        {
            glm::radians(rot[0]),
            glm::radians(rot[1]),
            glm::radians(rot[2])
        };
        go->m_transform.scale = { scl[0], scl[1], scl[2] };


        // Special case: procedural lights
        if (objJson.value("special", "") == "lights")
        {
            int count = objJson.value("count", 1);
            float orbitRadius = objJson.value("orbit_radius", 5.f);
            float height = objJson.value("height", 3.5f);
            float intensity = objJson.value("intensity", 0.1f);
            float lightRadius = objJson.value("light_radius", 0.1f);

            const auto& colorsJson = objJson["colors"];

            for (int i = 0; i < count; ++i)
            {
                float angle = i * glm::two_pi<float>() / count;
                glm::vec3 pos =
                {
                    orbitRadius * std::cos(angle),
                    height,
                    orbitRadius * std::sin(angle)
                };

                const auto& c = colorsJson[i % colorsJson.size()];
                glm::vec3 color{ c[0].get<float>(), c[1].get<float>(), c[2].get<float>() };

                auto light = SandboxGameObject::makePointLight(intensity, lightRadius, color);
                light->getTransform().translation = pos;

                spdlog::info("Placed light at ({:.2f},{:.2f},{:.2f})", pos.x, pos.y, pos.z);

                addGameObject(light);
                attachRenderable(light, RenderableType::Light);
                RenderableID rid = m_goRenderable[light->getId()];

                if (MeshInstance* inst = m_renderRegistry.getInstanceMutable(rid))
                {
                    inst->emissiveColor = color;
                    inst->intensity = intensity;
                }
            }

            continue;
        }


        // Cubemap for this object?
        if (objJson.contains("cubemap"))
            go->m_cubemapTextureName = objJson["cubemap"].get<std::string>();

        // Is skybox?
        bool isSkybox = (objJson.value("usage", "") == "skybox");

        auto getStr = [&](const json& j, const char* k)
            -> std::optional<std::string>
            {
                if (auto it = j.find(k); it != j.end() && it->is_string())
                    return it->get<std::string>();
                return std::nullopt;
            };

        RenderTag tag = RenderTag::Auto;
        RenderableType rtype = RenderableType::None;

        auto rs = getStr(objJson, "renderSystem");
        if (!rs) rs = getStr(objJson, "render_system");

        if (rs)
        {
            const std::string v = *rs;

            if (v == "scene") { tag = RenderTag::Scene; rtype = RenderableType::Scene; }
            else if (v == "gltf") { tag = RenderTag::Gltf; rtype = RenderableType::Gltf; }
            else if (v == "obj") { tag = RenderTag::Obj; rtype = RenderableType::Obj; }
            else if (v == "skybox") { tag = RenderTag::Skybox; rtype = RenderableType::Skybox; }
            else if (v == "light") { tag = RenderTag::PointLight; rtype = RenderableType::Light; }
            else { tag = RenderTag::Auto; rtype = RenderableType::None; }

            spdlog::info("Render tag override for object '{}': {}",
                objJson.value("name", "unnamed"), v);
        }
        else
        {
            if (isSkybox) {
                tag = RenderTag::Skybox;
                rtype = RenderableType::Skybox;
            }
            else if (go->getPointLight()) {
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

        go->setRenderTag(tag);


        if (isSkybox)
            setSkyboxObject(go);

        addGameObject(go);

        if (go->getModel())
            attachRenderable(go, rtype);
    }
    initSceneData();
    spdlog::info("Scene loaded.");
}


void SandboxScene::attachRenderable(std::shared_ptr<IGameObject> go,
    RenderableType overrideType)
{
    TransformData t{};
    t.model = go->getTransform().mat4();
    t.normalMat = glm::transpose(glm::inverse(t.model));

    const RenderableType type =
        (overrideType != RenderableType::None)
        ? overrideType
        : (m_bIsGltf ? RenderableType::Gltf :
            m_bIsObj ? RenderableType::Obj :
            RenderableType::Scene);

    createRenderable(
        go->getId(),
        0,
        0,
        t,
        type
    );
}

std::pair<glm::mat4, glm::mat4>
SandboxScene::getMainCameraMatrices() const
{
    const auto& cam = m_players[0]->getController().getCamera();
    return { cam.getViewMatrix(), cam.getProjectionMatrix() };
}

SandboxCamera& SandboxScene::getCamera() {
    if (m_players.empty()) {
        throw std::runtime_error("no players available to get camera from");
    }

    auto* player = dynamic_cast<SandboxPlayer*>(m_players[0].get());
    if (!player) {
        throw std::runtime_error("first player is not a SandboxPlayer");
    }

    return player->getController().getCamera();
}