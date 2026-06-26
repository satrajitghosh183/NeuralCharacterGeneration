# Standalone Vulkan game — "Rock Walker". Built on a machine WITH a display + Vulkan SDK
# (your M4 Pro via MoltenVK, or a Linux box with a GPU + Xorg/Wayland), NOT the headless H100.
#
#   brew install glfw vulkan-headers vulkan-loader molten-vk glslang   # macOS prereqs
#   (or install the LunarG Vulkan SDK, which provides all of the above)
#
#   cmake -S apps/ncg_game -B build/game -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/game
#   ./build/game/ncg_game --glb rock_char.glb        # play
#
# Controls: WASD walk · mouse-drag orbit · scroll zoom · Space jump · Esc quit.
cmake_minimum_required(VERSION 3.24)
project(ncg_game LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(vkb
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git GIT_TAG v1.3.290)
FetchContent_Declare(vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git GIT_TAG v3.1.0)
FetchContent_Declare(tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git GIT_TAG v2.9.3)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1)
FetchContent_MakeAvailable(glfw vkb vma tinygltf glm)

# Compile GLSL -> SPIR-V at build time.
find_program(GLSLANG glslangValidator HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} ENV VULKAN_SDK
             PATH_SUFFIXES bin)
if(NOT GLSLANG)
  message(FATAL_ERROR "glslangValidator not found (install the Vulkan SDK / 'brew install glslang')")
endif()
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPV_DIR})
set(SPV_FILES)
foreach(s mesh.vert mesh.frag floor.vert floor.frag)
  add_custom_command(OUTPUT ${SPV_DIR}/${s}.spv
    COMMAND ${GLSLANG} -V ${SHADER_DIR}/${s} -o ${SPV_DIR}/${s}.spv
    DEPENDS ${SHADER_DIR}/${s} VERBATIM)
  list(APPEND SPV_FILES ${SPV_DIR}/${s}.spv)
endforeach()
add_custom_target(ncg_game_shaders DEPENDS ${SPV_FILES})

add_executable(ncg_game main.cpp vk_mem_alloc_impl.cpp tiny_gltf_impl.cpp)
add_dependencies(ncg_game ncg_game_shaders)
target_link_libraries(ncg_game PRIVATE Vulkan::Vulkan glfw vk-bootstrap::vk-bootstrap
                                       VulkanMemoryAllocator tinygltf glm::glm)
target_compile_definitions(ncg_game PRIVATE SPV_DIR="${SPV_DIR}")
