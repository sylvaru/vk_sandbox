#include "engine.h"
#include "key_codes.h"
#include "spdlog/spdlog.h"
#include "frame_info.h"
#include <thread>
#include <chrono>



namespace Core {

	SandboxEngine::SandboxEngine(const EngineSpecification& engineSpec)
		: m_engineSpec(engineSpec)
		, m_window(m_engineSpec.windowSpec.Width, m_engineSpec.windowSpec.Height, m_engineSpec.Name)
		, m_vkinstance()
		, m_device(m_vkinstance, m_window)
		, m_assetManager(m_device)
		, m_renderer(m_device, m_window)
		, m_windowInput(std::make_shared<GLFWWindowInput>(m_window.getGLFWwindow()))
	{
		m_assetManager.preloadGlobalAssets();
		initialize();
	}
	SandboxEngine::~SandboxEngine() {

	}
	void SandboxEngine::initialize() {
		if (auto* userData = static_cast<WindowUserData*>(m_windowInput->getWindowUserPointer())) {
			userData->input = m_windowInput.get();
		}
		m_windowInput->lockCursor(m_cursorLocked);
		setupInputCallbacks();

 
		spdlog::info("Engine initialized");
	}

    void SandboxEngine::runApp()
    {
        if (m_layers.empty()) {
            spdlog::warn("SandboxEngine::runApp() called but no layers pushed");
            return;
        }

        // Initialize all layers
        for (auto& layer : m_layers) {
            layer->onInit();
            // If layer has a scene, initialize renderer systems for it
            if (IScene* s = layer->getSceneInterface()) {
                m_renderer.initializeSystems(m_assetManager, *s);
            }
        }

        using clock = std::chrono::high_resolution_clock;
        using duration_t = std::chrono::duration<double>;
        constexpr double TARGET_FPS = 144.0;
        constexpr double TARGET_FRAME_TIME = 1.0 / TARGET_FPS;
        auto lastTime = clock::now();

        auto pickTopScene = [&]() -> IScene* {
            for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
                if (IScene* s = (*it)->getSceneInterface()) return s;
            }
            return nullptr;
            };

        while (!m_windowInput->isWindowShouldClose()) {

            m_windowInput->pollEvents();
            processInput();

            auto now = clock::now();
            double deltaTime = duration_t(now - lastTime).count();
            lastTime = now;

            // Update all layers (game logic). Layers mutate scene/state.
            for (auto& layer : m_layers) {
                layer->onUpdate(static_cast<float>(deltaTime));
            }

            // Select scene & camera to render from
            IScene* scene = pickTopScene();
            const ICamera& cam = scene->getCamera();

            if (scene) {
                ISandboxRenderer::FrameContext frame = m_renderer.beginFrame();
                if (!frame.isValid()) break;

                int idx = m_renderer.getFrameIndex();

                FrameInfo info{
                    idx,
                    static_cast<float>(deltaTime),
                    frame.primaryGraphicsCommandBuffer,
                    cam,
                    m_renderer.getGlobalDescriptorSet()[idx],
                    scene->getGameObjects(),
                    *scene,
                    scene->getRenderableRegistry()
                };

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
                // No scene: still allow layers to render (UI-only apps)
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
            //layer->onDetach();
        }
    }


	inline void SandboxEngine::toggleCursorLock() {
		m_cursorLocked = !m_cursorLocked;
		m_windowInput->lockCursor(m_cursorLocked);
	}

	inline void SandboxEngine::setupInputCallbacks() {
		m_windowInput->setKeyCallback([this](SandboxKey key, int scancode, KeyAction action, int mods) {
			if (key == SandboxKey::LEFT_ALT && action == KeyAction::PRESS) {
				toggleCursorLock();
			}
			});
	}

	inline void SandboxEngine::processInput() {

		if (m_windowInput && m_windowInput->isKeyPressed(SandboxKey::ESCAPE)) {
			m_windowInput->requestWindowClose();
		}
	}
}
