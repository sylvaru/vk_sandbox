# external/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(external)



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
find_package(Vulkan REQUIRED)
target_link_libraries(ktx PUBLIC Vulkan::Vulkan)

# ==========================================================
# PhysX
# ==========================================================
set(PHYSX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/physx/physx")
set(PHYSX_INCLUDE_DIR "${PHYSX_DIR}/include")
set(PHYSX_BIN_DIR "${PHYSX_DIR}/bin/win.x86_64.vc143.mt")

set(PHYSX_DLLS
    PhysXFoundation_64.dll
    PhysXCommon_64.dll
    PhysX_64.dll
)

set(PHYSX_STATIC_LIBS
    PhysXExtensions_static_64
    PhysXCharacterKinematic_static_64
)

set(PHYSX_DYNAMIC_LIBS
    PhysXFoundation_64
    PhysXCommon_64
    PhysX_64
)

function(get_physx_config_folder out_var)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(${out_var} "debug" PARENT_SCOPE)
    else()
        set(${out_var} "release" PARENT_SCOPE)
    endif()
endfunction()
get_physx_config_folder(PHYSX_CFG)

# Static libs
foreach(lib IN LISTS PHYSX_STATIC_LIBS)
    if(NOT TARGET ${lib})
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_LOCATION "${PHYSX_BIN_DIR}/${PHYSX_CFG}/${lib}.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_INCLUDE_DIR}"
        )
    endif()
endforeach()

# Dynamic libs
foreach(lib IN LISTS PHYSX_DYNAMIC_LIBS)
    if(NOT TARGET ${lib})
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_IMPLIB "${PHYSX_BIN_DIR}/${PHYSX_CFG}/${lib}.lib"
            IMPORTED_LOCATION "${PHYSX_BIN_DIR}/${PHYSX_CFG}/${lib}.dll"
            INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_INCLUDE_DIR}"
        )
    endif()
endforeach()

# PhysX interface target
add_library(PhysX INTERFACE)
target_include_directories(PhysX INTERFACE "${PHYSX_INCLUDE_DIR}")
target_link_libraries(PhysX INTERFACE ${PHYSX_STATIC_LIBS} ${PHYSX_DYNAMIC_LIBS})

# Expose DLL directory as property
set_property(TARGET PhysX PROPERTY PHYSX_DLL_DIR "${PHYSX_BIN_DIR}")