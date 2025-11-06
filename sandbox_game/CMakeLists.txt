# sandbox_game/CMakeLists.txt
add_executable(sandbox_game)

target_sources(sandbox_game
    PRIVATE
        src/main.cpp
        src/layers/game_layer.cpp
        src/layers/imgui_layer.cpp
        src/input/player_input.cpp
        src/entities/player.cpp 
        src/entities/game_object.cpp 
        src/scene/scene.cpp 
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
