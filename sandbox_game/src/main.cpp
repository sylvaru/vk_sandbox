// main.cpp
#include <memory>
#include "engine.h"
#include "game/game_layer.h"

int main()
{
    Core::EngineSpecification engineSpec;
    engineSpec.Name = "A Space In Time";
    engineSpec.windowSpec.Width = 1920;
    engineSpec.windowSpec.Height = 1080;

    Core::SandboxEngine engine(engineSpec);

    engine.pushLayer<MyGameLayer>();

    engine.runApp();

    return 0;
}

