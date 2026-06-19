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

include(NcgWarnings)   # defines interface target ncg::warnings

# --- Modules ---------------------------------------------------------------
add_subdirectory(third_party)
add_subdirectory(ncg-core)
add_subdirectory(ncg-io)

# Phase 1 modules are added as they come online:
# add_subdirectory(ncg-select)
# add_subdirectory(ncg-body)
# add_subdirectory(ncg-recon)
# add_subdirectory(ncg-runtime)

if(NCG_BUILD_APPS AND NCG_WITH_CUDA)
  # apps depend on CUDA render/runtime; skip in host-only lint configs.
  # add_subdirectory(apps)
endif()

if(NCG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "NCG: configured ${PROJECT_NAME} ${PROJECT_VERSION} "
               "(tests=${NCG_BUILD_TESTS} apps=${NCG_BUILD_APPS} cuda=${NCG_WITH_CUDA})")
