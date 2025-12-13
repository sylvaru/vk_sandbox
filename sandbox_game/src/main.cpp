// sandbox_game/main.cpp
#include "engine.h"
#include "layers/game_layer.h"
#include "layers/imgui_layer.h"


int main()
{
    Core::EngineSpecification engineSpec;
    engineSpec.name = "A Space In Time";
    engineSpec.windowSpec.mode = WindowMode::BorderlessFullscreen;
    Core::SandboxEngine engine(engineSpec);

    engine.pushLayer<MyGameLayer>();
    engine.pushLayer<ImGuiLayer>();
    engine.initialize();
    engine.runApp();

    return 0;
}

