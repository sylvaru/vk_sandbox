# sandbox_game/CMakeLists.txt
add_executable(sandbox_game)

target_sources(sandbox_game
    PRIVATE
        src/main.cpp
        include/layers/game_layer.h
        src/layers/game_layer.cpp
        include/layers/imgui_layer.h
        src/layers/imgui_layer.cpp
        include/input/player_input.h
        src/input/player_input.cpp
        include/entities/player.h
        src/entities/player.cpp
        include/entities/game_object.h
        src/entities/game_object.cpp
        include/scene/scene.h
        src/scene/scene.cpp
        include/camera/camera.h
        src/camera/camera.cpp
)

target_include_directories(sandbox_game
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sandbox_game
    PRIVATE
        sandbox_engine
        core_interfaces
        glm
        json
)
