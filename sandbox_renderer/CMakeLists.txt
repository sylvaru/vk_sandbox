# sandbox_renderer/CMakeLists.txt
add_library(sandbox_renderer STATIC)

target_sources(sandbox_renderer
    PRIVATE
         src/window.cpp
         src/vk_tools/vk_tools.cpp 
         src/vulkan_wrapper/vulkan_instance.cpp
         src/vulkan_wrapper/vulkan_device.cpp
         src/vulkan_wrapper/vulkan_renderer.cpp
         src/vulkan_wrapper/vulkan_swapchain.cpp
         src/vulkan_wrapper/vulkan_buffer.cpp
         src/vulkan_wrapper/vulkan_pipeline.cpp
         src/vulkan_wrapper/vulkan_descriptor.cpp
         src/vulkan_wrapper/vulkan_obj.cpp
         src/vulkan_wrapper/vulkan_gltf.cpp
         src/vulkan_wrapper/vulkan_texture.cpp
         src/vulkan_wrapper/render_systems/obj_render_system.cpp
         src/vulkan_wrapper/render_systems/gltf_render_system.cpp
         src/vulkan_wrapper/render_systems/point_light_rs.cpp
         src/vulkan_wrapper/render_systems/skybox_ibl_rs.cpp
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
        core_interfaces
)