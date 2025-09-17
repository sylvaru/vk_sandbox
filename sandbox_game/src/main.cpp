// main.cpp
#include <memory>
#include "engine.h"
#include "game/game_layer.h"

int main()
{
    Core::AppSpecification appSpec;
    appSpec.Name = "A Space In Time";
    appSpec.windowSpec.Width = 1080;
    appSpec.windowSpec.Height = 1080;

    Core::SandboxEngine engine(appSpec);

    engine.pushLayer<MyGameLayer>();

    engine.runApp();

    return 0;
}

