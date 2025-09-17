#pragma once
#include "renderer_i.h"
#include "scene_i.h"

namespace Core { class SandboxEngine; }
struct IGameLayer {

	virtual void onAttach(Core::SandboxEngine* engine) { /* optional */ }
	virtual void onInit() = 0;
	virtual void onUpdate(float deltaTime) = 0;
	virtual void onRender(ISandboxRenderer::FrameContext& frame) {};
	virtual IScene* getSceneInterface() = 0;
	virtual ~IGameLayer() = default;
};

