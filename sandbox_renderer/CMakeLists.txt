# sandbox_renderer/CMakeLists.txt
add_library(sandbox_renderer STATIC)

target_sources(sandbox_renderer
    PRIVATE
         include/vulkan_wrapper/core/window.h
         src/vulkan_wrapper/core/window.cpp
         include/vulkan_wrapper/core/vulkan_renderer.h
         src/vulkan_wrapper/core/vulkan_renderer.cpp
         include/vk_tools/vk_tools.h
         src/vk_tools/vk_tools.cpp 
         include/vulkan_wrapper/vulkan_instance.h
         src/vulkan_wrapper/vulkan_instance.cpp
         include/vulkan_wrapper/vulkan_device.h
         src/vulkan_wrapper/vulkan_device.cpp
         include/vulkan_wrapper/vulkan_swapchain.h
         src/vulkan_wrapper/vulkan_swapchain.cpp
         include/vulkan_wrapper/vulkan_buffer.h
         src/vulkan_wrapper/vulkan_buffer.cpp
         include/vulkan_wrapper/vulkan_pipeline.h
         src/vulkan_wrapper/vulkan_pipeline.cpp
         include/vulkan_wrapper/vulkan_descriptor.h
         src/vulkan_wrapper/vulkan_descriptor.cpp
         include/vulkan_wrapper/vulkan_obj.h
         src/vulkan_wrapper/vulkan_obj.cpp
         include/vulkan_wrapper/vulkan_gltf.h
         src/vulkan_wrapper/vulkan_gltf.cpp
         include/vulkan_wrapper/vulkan_texture.h
         src/vulkan_wrapper/vulkan_texture.cpp
         include/vulkan_wrapper/render_systems/gltf_render_system.h
         src/vulkan_wrapper/render_systems/gltf_render_system.cpp
         "include/vulkan_wrapper/render_systems/pointlight_render_system.h"
         src/vulkan_wrapper/render_systems/pointlight_render_system.cpp
         "include/vulkan_wrapper/render_systems/skybox_render_system.h"
         "src/vulkan_wrapper/render_systems/skybox_render_system.cpp"
         "include/vulkan_wrapper/render_systems/scene_render_system.h"
         "src/vulkan_wrapper/render_systems/scene_render_system.cpp"
         include/vulkan_wrapper/core/render_graph.h
         src/vulkan_wrapper/core/render_graph.cpp
         include/vulkan_wrapper/core/renderable_registry.h
         src/vulkan_wrapper/core/renderable_registry.cpp
 )


target_include_directories(sandbox_renderer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

find_package(Vulkan REQUIRED)

# Link against required libraries
target_link_libraries(sandbox_renderer
    PUBLIC
        glfw
		glm
        spdlog
        Vulkan::Vulkan
        tinyobjloader
        tinygltf
        ktx
        imgui
        core_interfaces
)