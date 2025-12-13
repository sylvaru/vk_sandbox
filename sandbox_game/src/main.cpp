// sandbox_game/main.cpp
#include "engine.h"
#include "layers/game_layer.h"
#include "layers/imgui_layer.h"


int main()
{
    Core::EngineSpecification engineSpec;
    engineSpec.name = "A Space In Time";
    engineSpec.windowSpec.width = 1920;
    engineSpec.windowSpec.height = 1080;
    Core::SandboxEngine engine(engineSpec);

    engine.pushLayer<MyGameLayer>();
    engine.pushLayer<ImGuiLayer>();
    engine.initialize();
    engine.runApp();

    return 0;
}

