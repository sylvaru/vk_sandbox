#include "engine.h"
#include "key_codes.h"
#include "spdlog/spdlog.h"
#include "frame_info.h"
#include <thread>
#include <chrono>


SandboxEngine::SandboxEngine() {
	m_assetManager.preloadGlobalAssets();
	initialize();
}
SandboxEngine::~SandboxEngine() {
	if (m_pPhysics) {
		m_pPhysics->shutdown();
	}
}
void SandboxEngine::initialize() {
	m_windowInput = std::make_shared<GLFWWindowInput>(m_window.getGLFWwindow());
	if (auto* userData = static_cast<WindowUserData*>(m_windowInput->getWindowUserPointer())) {
		userData->input = m_windowInput.get();
	}
	m_windowInput->lockCursor(m_cursorLocked);
	setupInputCallbacks();


	m_pPhysics = std::make_unique<SandboxPhysics>();
	if (!m_pPhysics->init(/*workerThreads=*/2)) {
		spdlog::error("SandboxEngine: PhysX initialize failed");
		// decide policy: continue without physics or abort. Here we continue.
		m_pPhysics.reset();
	}
	else {
		if (!m_pPhysics->createDefaultScene()) {
			spdlog::error("SandboxEngine: createDefaultScene failed");
			m_pPhysics->shutdown();
			m_pPhysics.reset();
		}
	}
	spdlog::info("Engine initialized");
}
void SandboxEngine::initLayer(IGameLayer* game) {


	game->onInit();

	if (m_pPhysics) {
		//game->onPhysicsInit(m_pPhysics.get());
	}

	m_renderer.initializeSystems(m_assetManager, game->getSceneInterface());
	spdlog::info("Game initialized");
}

void SandboxEngine::run(std::unique_ptr<IGameLayer> game) {
	using clock = std::chrono::high_resolution_clock;
	using duration_t = std::chrono::duration<double>;

	constexpr double TARGET_FPS = 60.0;
	constexpr double TARGET_FRAME_TIME = 1.0 / TARGET_FPS;

	auto lastTime = clock::now();

	IScene& scene = game->getSceneInterface();
	ICamera& cam = scene.getCamera();


	while (!m_windowInput->isWindowShouldClose()) {
		// Poll events / process input
		m_windowInput->pollEvents();
		processInput();

		// Compute delta time
		auto now = clock::now();
		double deltaTime = duration_t(now - lastTime).count();
		lastTime = now;

		// Begin recording command buffer
		ISandboxRenderer::FrameContext frame = m_renderer.beginFrame();
		if (!frame.isValid()) break;

		int idx = m_renderer.getFrameIndex();

		FrameInfo info{
		 idx,
		 static_cast<float>(deltaTime),
		 frame.primaryGraphicsCommandBuffer,
		 cam,
		 m_renderer.getGlobalDescriptorSet()[idx],
		 scene.getGameObjects(),
		 scene
		};

		// Update game and subsystems 
		game->onUpdate(static_cast<float>(deltaTime));

		GlobalUbo ubo{};
		ubo.projection = cam.getProjectionMatrix();
		ubo.view = cam.getViewMatrix();
		ubo.viewPos = glm::vec4(cam.getPosition(), 1.0f);

		m_renderer.updateSystems(info, ubo, static_cast<float>(deltaTime));

		auto& uboBuffer = m_renderer.getUboBuffers()[idx];
		uboBuffer->writeToBuffer(&ubo);
		uboBuffer->flush();
	
		// Render Game
		m_renderer.beginSwapChainRenderPass(frame);
		m_renderer.renderSystems(info);
		m_renderer.endSwapChainRenderPass(frame);
		m_renderer.endFrame(frame);

		// Frame cap sleep
		auto frameEnd = clock::now();
		double frameTime = duration_t(frameEnd - now).count();
		if (frameTime < TARGET_FRAME_TIME) {
			auto sleepDuration = std::chrono::duration<double>(TARGET_FRAME_TIME - frameTime);
			std::this_thread::sleep_for(sleepDuration);
		}
	}
	m_renderer.waitDeviceIdle();

}

void SandboxEngine::toggleCursorLock() {
	m_cursorLocked = !m_cursorLocked;
	m_windowInput->lockCursor(m_cursorLocked);
}

void SandboxEngine::setupInputCallbacks() {
	m_windowInput->setKeyCallback([this](SandboxKey key, int scancode, KeyAction action, int mods) {
		if (key == SandboxKey::LEFT_ALT && action == KeyAction::PRESS) {
			toggleCursorLock();
		}
		});
}

void SandboxEngine::processInput() {

	if (m_windowInput && m_windowInput->isKeyPressed(SandboxKey::ESCAPE)) {
		m_windowInput->requestWindowClose();
	}
}