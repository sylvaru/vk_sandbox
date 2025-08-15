# sandbox_game/CMakeLists.txt
add_executable(sandbox_game)

target_sources(sandbox_game
    PRIVATE
        src/main.cpp
        src/game/game_layer.cpp
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

)

target_link_libraries(sandbox_game
    PRIVATE
        sandbox_engine
        PhysX       # link PhysX here
        core_interfaces
        glm
        json
)

# Copy runtime DLLs from PhysX
get_target_property(PHYSX_DLL_DIR PhysX PHYSX_DLL_DIR)
set(PHYSX_DLLS
    PhysXFoundation_64.dll
    PhysXCommon_64.dll
    PhysX_64.dll
)

register_runtime_dlls(sandbox_game "${PHYSX_DLL_DIR}" ${PHYSX_DLLS})