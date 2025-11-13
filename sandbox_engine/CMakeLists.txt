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
)

target_include_directories(sandbox_engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../External/cmake_build/bullet3/src
)

# Link against required libraries
target_link_libraries(sandbox_engine
    PUBLIC core_interfaces
    PUBLIC sandbox_renderer
    PRIVATE spdlog::spdlog
    PRIVATE glfw
    PRIVATE json
    PRIVATE BulletDynamics
    PRIVATE BulletCollision
    PRIVATE LinearMath
    PRIVATE Bullet3Common
)