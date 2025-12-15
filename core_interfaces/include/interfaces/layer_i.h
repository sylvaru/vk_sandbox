#pragma once
#include "renderer_i.h"
#include "scene_i.h"

namespace Core { class SandboxEngine; }
struct ILayer {

	virtual void onAttach(Core::SandboxEngine* engine) { /* optional */ }
	virtual void onInit() = 0;
	virtual void onUpdate(float deltaTime) = 0;
	virtual void onRender(ISandboxRenderer::FrameContext& frame) {};
	virtual void onDetach() = 0;
	virtual bool isAttached() = 0;
	virtual IScene* getSceneInterface() = 0;
	virtual ~ILayer() = default;
};

