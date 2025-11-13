// sandbox_game/main.cpp
#include <memory>
#include "engine.h"
#include "layers/game_layer.h"
#include "layers/imgui_layer.h"


int main()
{
    Core::EngineSpecification engineSpec;
    engineSpec.Name = "A Space In Time";
    engineSpec.windowSpec.Width = 1720;
    engineSpec.windowSpec.Height = 1000;

    Core::SandboxEngine engine(engineSpec);

    engine.pushLayer<MyGameLayer>();
    engine.pushLayer<ImGuiLayer>();

    engine.runApp();


    return 0;
}

