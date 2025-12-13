// sandbox_engine/src/engine.cpp
#include "common/engine_pch.h"
#include "engine.h"
#include "key_codes.h"
#include "frame_info.h"


namespace Core {

    SandboxEngine::SandboxEngine()
        : SandboxEngine(EngineSpecification{}) {
    }

	SandboxEngine::SandboxEngine(const EngineSpecification& engineSpec)
		: m_engineSpec(engineSpec)
		, m_window(std::make_unique<GLFWwindowAndInput>(
            m_engineSpec.windowSpec,
            m_engineSpec.name))
		, m_vkinstance()
		, m_device(m_vkinstance, *m_window)
		, m_assetManager(m_device)
		, m_renderer(m_device, *m_window)
		, m_physicsEngine(std::make_unique<PhysicsEngine>())
	{
		m_assetManager.preloadGlobalAssets();
        spdlog::info("Window: {}x{}, Name: \"{}\"",
            m_engineSpec.windowSpec.width,
            m_engineSpec.windowSpec.height,
            m_engineSpec.name);
	}
    SandboxEngine::~SandboxEngine() {}

	void SandboxEngine::initialize() {

        m_window->lockCursor(m_cursorLocked);
		setupInputCallbacks();

        for (auto& layer : m_layers) {
            layer->onInit();
            if (IScene* s = layer->getSceneInterface()) {
                m_renderer.initializeSystems(m_assetManager, *s);
            }
        }

		spdlog::info("Engine initialized");
	}

    void SandboxEngine::runApp()
    {
        if (m_layers.empty()) {
            spdlog::warn("SandboxEngine::runApp() called but no layers pushed");
            return;
        }

        // Fallback scanning function used only when cache is empty/invalid
        auto scanForTopScene = [&]() -> IScene* {
            for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
                if (IScene* s = (*it)->getSceneInterface()) return s;
            }
            return nullptr;
            };

        using clock = std::chrono::high_resolution_clock;
        using duration_t = std::chrono::duration<double>;
        constexpr double TARGET_FPS = 300.0;
        constexpr double TARGET_FRAME_TIME = 1.0 / TARGET_FPS;
        auto lastTime = clock::now();

        while (!m_window->isWindowShouldClose()) {

            m_window->pollEvents();
            processInput();

            // Don't render when minimized
            int fbWidth = 0, fbHeight = 0;
            m_window->getFramebufferSize(fbWidth, fbHeight);

            if (fbWidth == 0 || fbHeight == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                continue;
            }
            // Compute delta time
            auto now = clock::now();
            double deltaTime = duration_t(now - lastTime).count();
            lastTime = now;

            // Update all layers
            for (auto& layer : m_layers) {
                layer->onUpdate(static_cast<float>(deltaTime));
            }

            // Determine active scene: prefer cached pointer, fallback to scan
            IScene* scene = m_activeScene;
            if (!scene && !m_activeSceneOwner) {
                scene = scanForTopScene();
            }

            if (scene) {
                ISandboxRenderer::FrameContext frame = m_renderer.beginFrame();
                if (!frame.isValid()) break;

                const ICamera& cam = scene->getCamera();

                int idx = m_renderer.getFrameIndex();

                FrameInfo info{};
                info.frameIndex = idx;
				info.frameTime = static_cast<float>(deltaTime);
				info.commandBuffer = frame.primaryGraphicsCommandBuffer;
				info.camera = &cam;
				info.globalDescriptorSet = m_renderer.getGlobalDescriptorSet()[idx];
				info.renderRegistry = scene->getRenderableRegistry();

                GlobalUbo ubo{};
                ubo.projection = cam.getProjectionMatrix();
                ubo.view = cam.getViewMatrix();
                ubo.viewPos = glm::vec4(cam.getPosition(), 1.0f);

                m_renderer.updateSystems(info, ubo, static_cast<float>(deltaTime));

                auto& uboBuffer = m_renderer.getUboBuffers()[idx];
                uboBuffer->writeToBuffer(&ubo);
                uboBuffer->flush();

                for (auto& layer : m_layers) {
                    layer->onRender(frame);
                }

                // Main render graph sequence
                m_renderer.renderSystems(info, frame);

                if (m_renderer.isImGuiInitialized()) {
                    ImGui::UpdatePlatformWindows();
                    ImGui::RenderPlatformWindowsDefault();
                }

                m_renderer.endFrame(frame);
            }
            else {
                // No scene: still allow layers to render Ui only apps
                ISandboxRenderer::FrameContext frame = m_renderer.beginFrame();
                if (!frame.isValid()) break;
                m_renderer.beginSwapChainRenderPass(frame);
                for (auto& layer : m_layers) {
                    layer->onRender(frame);
                }
                m_renderer.endSwapChainRenderPass(frame);
                m_renderer.endFrame(frame);
            }

            // Frame cap sleep
            auto frameEnd = clock::now();
            double frameTime = duration_t(frameEnd - now).count();
            if (frameTime < TARGET_FRAME_TIME) {
                std::this_thread::sleep_for(std::chrono::duration<double>(TARGET_FRAME_TIME - frameTime));
            }
        }

        m_renderer.waitDeviceIdle();

        // Cleanup layers
        for (auto& layer : m_layers) {
            layer->onDetach();
        }
    }
    void SandboxEngine::setupInputCallbacks() {
        m_window->setKeyCallback([this](SandboxKey key, int scancode, KeyAction action, int mods) {
            if (key == SandboxKey::LEFT_ALT && action == KeyAction::PRESS) {
                toggleCursorLock();
            }
            });
    }

    void SandboxEngine::processInput() {

        if (m_window && m_window->isKeyPressed(SandboxKey::ESCAPE)) {
            m_window->requestWindowClose();
        }
    }

	void SandboxEngine::toggleCursorLock() {
		m_cursorLocked = !m_cursorLocked;
        m_window->lockCursor(m_cursorLocked);
	}


    void SandboxEngine::setActiveScene(IScene* scene, ILayer* owner) {
        if (scene) {
            // Set/claim an active scene
            m_activeScene = scene;
            m_activeSceneOwner = owner;
            spdlog::debug("[Engine] setActiveScene: scene={} owner={} (claim)", fmt::ptr(scene), fmt::ptr(owner));
        }
        else {
            // Clearing the active scene: only allow if the caller is the owner (or if there is no owner)
            if (m_activeSceneOwner == nullptr || owner == m_activeSceneOwner) {
                spdlog::debug("[Engine] setActiveScene: scene=nullptr owner={} (cleared)", fmt::ptr(owner));
                m_activeScene = nullptr;
                m_activeSceneOwner = nullptr;
            }
            else {
                // Deny clears from non-owners and log it so we can find the caller
                spdlog::warn("[Engine] Ignored setActiveScene(nullptr) from owner={} while current owner={} remains. (ptrs)",
                    fmt::ptr(owner), fmt::ptr(m_activeSceneOwner));
            }
        }
    }
}
