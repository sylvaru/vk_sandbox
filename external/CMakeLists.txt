# external/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(external)

find_package(Vulkan REQUIRED)


# ==========================================================
# spdlog
# ==========================================================
add_subdirectory(spdlog)

# ==========================================================
# GLM (header-only)
# ==========================================================
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")

# ==========================================================
# JSON (header-only)
# ==========================================================
add_library(json INTERFACE)
target_include_directories(json INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/json")

# ==========================================================
# GLFW
# ==========================================================
add_subdirectory(glfw)

# ==========================================================
# stb_image (header-only)
# ==========================================================
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/stb")

# ==========================================================
# tinygltf
# ==========================================================
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/tinygltf")
target_link_libraries(tinygltf INTERFACE json stb)

# ==========================================================
# tinyobjloader
# ==========================================================
add_subdirectory(tinyobjloader)

# ----------------------------
# Bullet3
# ----------------------------

# Turn off demos/tests
set(BUILD_BULLET2_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_CPU_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_OPENGL3_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_BULLET3 ON CACHE BOOL "" FORCE)

# Make Bullet use the same MSVC runtime flags as the main project
if(MSVC)
    # Ensure global and cached variable are both set
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC runtime for Bullet" FORCE)
    # Also force propagate it down into subdirectories
    set(BULLET_MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}" CACHE STRING "" FORCE)
endif()

add_subdirectory(bullet3)

# ==========================================================
# dear imgui
# ==========================================================


add_library(imgui
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_vulkan.cpp
    imgui/backends/imgui_impl_glfw.cpp
)

target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
)

target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)


# ==========================================================
# KTX
# ==========================================================
set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ktx)
set(KTX_SOURCES
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c
    ${KTX_DIR}/lib/vkloader.c
)

add_library(ktx STATIC ${KTX_SOURCES})
target_compile_definitions(ktx PUBLIC KTX_FEATURE_VULKAN=1 KTX_FEATURE_LOAD=1)
target_include_directories(ktx PUBLIC
    "${KTX_DIR}/include"
    "${KTX_DIR}/other_include"
    "${KTX_DIR}/lib"
)


target_link_libraries(ktx PUBLIC Vulkan::Vulkan)