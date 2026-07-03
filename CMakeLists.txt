cmake_minimum_required(VERSION 3.27)

# CUDA is enabled conditionally below so this also configures host-only on a non-CUDA
# machine (Mac) to produce compile_commands.json for clang-tidy. The real build is the
# H100 (sm_90), where CUDA is always present.
project(NeuralCharGen LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# The presets request ccache as the compiler launcher; if it isn't installed, drop it
# rather than failing every compile with "ccache: not found".
foreach(_lang CXX CUDA)
  if(CMAKE_${_lang}_COMPILER_LAUNCHER)
    find_program(_ncg_launcher NAMES "${CMAKE_${_lang}_COMPILER_LAUNCHER}")
    if(NOT _ncg_launcher)
      message(WARNING "NCG: '${CMAKE_${_lang}_COMPILER_LAUNCHER}' not found — disabling ${_lang} compiler launcher.")
      unset(CMAKE_${_lang}_COMPILER_LAUNCHER CACHE)
      set(CMAKE_${_lang}_COMPILER_LAUNCHER "")
    endif()
    unset(_ncg_launcher CACHE)
  endif()
endforeach()

# --- Options ---------------------------------------------------------------
option(NCG_BUILD_TESTS "Build tests" ON)
option(NCG_BUILD_APPS  "Build apps"  ON)
option(NCG_LINT_ONLY   "Host-only configure for linting (no CUDA/apps)" OFF)

# --- CUDA (conditional) ----------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NOT NCG_LINT_ONLY)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(NCG_WITH_CUDA ON)
  include(NcgCudaArch)   # sets CMAKE_CUDA_ARCHITECTURES (default 90-real) if unset
  message(STATUS "NCG: CUDA enabled, architectures = ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(NCG_WITH_CUDA OFF)
  message(WARNING "NCG: CUDA compiler not found (or NCG_LINT_ONLY) — host-only configure. "
                  "CUDA targets and apps are skipped. The real build target is the H100.")
endif()

# --- Big dependencies ------------------------------------------------------
find_package(Torch REQUIRED)
if(NCG_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# CRITICAL: every target (including FetchContent spdlog/Catch2 which don't link Torch) must
# use the SAME libstdc++ ABI as libtorch, or std::string-bearing symbols mismatch at link
# time. TorchConfig sets TORCH_CXX_FLAGS to the right -D_GLIBCXX_USE_CXX11_ABI=N; apply it
# globally. Must be set BEFORE add_subdirectory(third_party).
if(DEFINED TORCH_CXX_FLAGS AND NOT "${TORCH_CXX_FLAGS}" STREQUAL "")
  message(STATUS "NCG: applying Torch ABI flags to all targets: ${TORCH_CXX_FLAGS}")
  separate_arguments(_ncg_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${_ncg_torch_flags}>
                      $<$<COMPILE_LANGUAGE:CUDA>:${_ncg_torch_flags}>)
endif()

include(NcgWarnings)   # defines interface target ncg::warnings

# --- Modules ---------------------------------------------------------------
# (Apple clang's incompatibility with spdlog's bundled-fmt consteval check is handled by an
# APPLE-only PATCH_COMMAND in third_party/CMakeLists.txt — see the note there.)
add_subdirectory(third_party)
add_subdirectory(ncg-core)
add_subdirectory(ncg-io)
add_subdirectory(ncg-select)
add_subdirectory(ncg-body)
add_subdirectory(ncg-recon)
add_subdirectory(ncg-geom)
add_subdirectory(ncg-diffuse)
add_subdirectory(ncg-material)
add_subdirectory(ncg-rig)
add_subdirectory(ncg-mesh)
add_subdirectory(ncg-record)

# ncg-runtime's only CUDA piece is the fast splat kernel (auto-dropped without CUDA; the
# soft/aniso renderers + camera are pure LibTorch). fit/nerf/apps run on CPU/MPS via those,
# so the whole pipeline builds on the Mac — the fast kernel path throws when !NCG_WITH_CUDA.
add_subdirectory(ncg-runtime)
add_subdirectory(ncg-fit)
add_subdirectory(ncg-nerf)

if(NCG_BUILD_APPS AND NOT NCG_LINT_ONLY)
  add_subdirectory(apps)
endif()

if(NCG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "NCG: configured ${PROJECT_NAME} ${PROJECT_VERSION} "
               "(tests=${NCG_BUILD_TESTS} apps=${NCG_BUILD_APPS} cuda=${NCG_WITH_CUDA})")
