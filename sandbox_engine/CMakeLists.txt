# sandbox_engine/CMakeLists.txt

cmake_minimum_required(VERSION 3.24)

project(sandbox_engine)

add_library(sandbox_engine STATIC
    src/engine.cpp
    src/asset_manager.cpp
    src/common/glfw_input.cpp 
)

target_include_directories(sandbox_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against required libraries
target_link_libraries(sandbox_engine
    PUBLIC
        sandbox_renderer
        spdlog::spdlog
        glfw
        json
        core_interfaces
)