add_library(sandbox_physics STATIC
    src/physics.cpp
)

target_include_directories(sandbox_physics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link PhysX
target_link_libraries(sandbox_physics
    PUBLIC
        PhysX
        glm
        spdlog::spdlog
        core_interfaces
)


