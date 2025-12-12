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

target_precompile_headers(sandbox_game
    PRIVATE
    include/common/game_pch.h
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

# ---------------------------------------------------------
# PhysX library + DLL root
# ---------------------------------------------------------
set(PHYSX_DLL_ROOT
    ${CMAKE_SOURCE_DIR}/external/cmake_build/physx/physx/bin/win.x86_64.vc143.md
)

target_link_directories(sandbox_game PRIVATE
    $<$<CONFIG:Debug>:${PHYSX_DLL_ROOT}/debug>
    $<$<CONFIG:Release>:${PHYSX_DLL_ROOT}/release>
)

# ---------------------------------------------------------
# Copy PhysX runtime DLLs
# ---------------------------------------------------------
function(copy_physx_dlls TARGET)
    foreach(dll
        PhysX_64.dll
        PhysXCommon_64.dll
        PhysXCooking_64.dll
        PhysXFoundation_64.dll
    )
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<$<CONFIG:Debug>:"${PHYSX_DLL_ROOT}/debug/${dll}">
                $<$<CONFIG:Release>:"${PHYSX_DLL_ROOT}/release/${dll}">
                $<TARGET_FILE_DIR:${TARGET}>
            COMMENT "Copying PhysX DLL: ${dll}"
        )
    endforeach()
endfunction()

copy_physx_dlls(sandbox_game)
