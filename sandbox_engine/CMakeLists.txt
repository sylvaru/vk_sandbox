# sandbox_engine/CMakeLists.txt

cmake_minimum_required(VERSION 3.24)

project(sandbox_engine)

add_library(sandbox_engine STATIC
    include/common/glfw_input.h
    src/common/glfw_input.cpp 
    include/engine.h
    src/engine.cpp
    include/asset_manager.h
    src/asset_manager.cpp
    include/physics/physics_engine.h
    src/physics/physics_engine.cpp
    include/base/engine_scene_base.h 
    src/base/engine_scene_base.cpp 
)

target_include_directories(sandbox_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external/cmake_build/physx/physx/include
)

target_link_libraries(sandbox_engine
    PUBLIC core_interfaces
    PUBLIC sandbox_renderer
    PUBLIC spdlog::spdlog
    PUBLIC glfw
    PUBLIC json

    # PhysX libs
    PUBLIC PhysX_64
    PUBLIC PhysXCommon_64
    PUBLIC PhysXCooking_64
    PUBLIC PhysXExtensions_static_64
    PUBLIC PhysXFoundation_64
    PUBLIC PhysXCharacterKinematic_static_64
)


