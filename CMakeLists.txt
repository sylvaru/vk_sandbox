cmake_minimum_required(VERSION 3.24)
project(vk_sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------
# Force dynamic CRT for MSVC (required for hot reload / edit & continue)
# ----------------------------
if(MSVC)
    # Use dynamic runtime in all configurations
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# ----------------------------
# Enable hot reloading
# ----------------------------
if (MSVC AND WIN32 AND NOT MSVC_VERSION VERSION_LESS 142)
    add_compile_options($<$<CONFIG:Debug>:/ZI>)
    add_link_options($<$<CONFIG:Debug>:/INCREMENTAL>)
endif()

# ----------------------------
# Enable warnings
# ----------------------------
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ----------------------------
# Global definitions
# ----------------------------
add_compile_definitions(
    GLM_ENABLE_EXPERIMENTAL
    PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ----------------------------
# Output directories
# ----------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)



# ----------------------------
# External libraries
# ----------------------------
add_subdirectory(external/cmake_build/)


# ----------------------------
# Function to register runtime DLLs dynamically
# ----------------------------
function(register_runtime_dlls TARGET DLL_DIR)
    foreach(dll ${ARGN})
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<$<CONFIG:Debug>:${DLL_DIR}/debug/${dll}>
                $<$<CONFIG:Release>:${DLL_DIR}/release/${dll}>
                $<TARGET_FILE_DIR:${TARGET}>
            COMMENT "Copying ${dll} for $<CONFIG> to output dir"
        )
    endforeach()
endfunction()


# ----------------------------
# Core targets
# ----------------------------
add_subdirectory(sandbox_renderer)
add_subdirectory(sandbox_engine)
add_subdirectory(sandbox_game)
add_subdirectory(core_interfaces)


#----------------------------
# Shader compilation using shaderc (glslc)
#----------------------------

# Find all shader source files
file(GLOB SHADER_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.frag"
)

# Output directory for compiled SPIR-V
set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/spirV")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})

# ----------------------------
# Prebuilt Shaderc paths
# ----------------------------
set(SHADERC_BINARY "${CMAKE_CURRENT_SOURCE_DIR}/external/standalone/shaderc/build/glslc/glslc.exe")

# ----------------------------
# Compile shaders into SPIR-V
# ----------------------------
file(GLOB SHADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.frag"
)

set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/spirV")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})

set(SPIRV_BINARIES)
foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(FILE_NAME ${SHADER_FILE} NAME)
    set(OUTPUT_SPV "${SPIRV_OUTPUT_DIR}/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT "${SPIRV_OUTPUT_DIR}/${FILE_NAME}.spv"
        COMMAND ${CMAKE_COMMAND} -E echo "Compiling shader via shaderc: ${FILE_NAME}"
        COMMAND "${SHADERC_BINARY}"
           "${SHADER_FILE}"
           --target-env=vulkan1.3
           -I "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders"
            -o "${SPIRV_OUTPUT_DIR}/${FILE_NAME}.spv"
        DEPENDS "${SHADER_FILE}"
        COMMENT "ShaderC compiling ${FILE_NAME}"
    )
    list(APPEND SPIRV_BINARIES ${OUTPUT_SPV})
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${SPIRV_BINARIES})
add_dependencies(sandbox_game CompileShaders)