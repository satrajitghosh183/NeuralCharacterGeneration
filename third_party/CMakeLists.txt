# Third-party dependencies via FetchContent. All pinned to explicit tags for reproducible
# remote builds. Heavy deps (LibTorch, CUDA) are found system-side, not fetched.
include(FetchContent)

# Be quiet and don't let sub-projects clobber our options.
set(FETCHCONTENT_QUIET OFF)

# --- spdlog (logging) ------------------------------------------------------
FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.15.0
  GIT_SHALLOW    TRUE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL       OFF CACHE BOOL "" FORCE)

# --- nlohmann_json (safetensors header + manifest parse) -------------------
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE)
set(JSON_BuildTests OFF CACHE INTERNAL "")

# --- Catch2 v3 (tests) -----------------------------------------------------
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(spdlog nlohmann_json Catch2)

# Catch2's CTest integration module (catch_discover_tests) lives in extras/.
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

# --- stb (header-only image I/O) -------------------------------------------
# stb has no release tags; track master (shallow). Headers are API-stable.
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
add_library(ncg::stb ALIAS stb)
# SYSTEM so stb's own (very noisy) -Wconversion/-Wsign-conversion warnings are suppressed.
target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

# --- Eigen (header-only linear algebra; used from Phase 1 on) --------------
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE)
# Use header-only; do not configure Eigen's own build/tests.
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(eigen INTERFACE)
add_library(ncg::eigen ALIAS eigen)
target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})
