include(NcgAddLibrary)

set(_core_private "")
if(NCG_WITH_CUDA)
  list(APPEND _core_private CUDA::cudart)
endif()

ncg_add_library(ncg-core
  SOURCES
    src/error.cpp
    src/logging.cpp
    src/device.cpp
  CUDA_SOURCES
    src/kernels/elementwise.cu
  PUBLIC_DEPS
    torch
    spdlog::spdlog
  PRIVATE_DEPS
    ${_core_private})

# Generate config.hpp (version + NCG_WITH_CUDA) into a build-tree include dir.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ncg/core/config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/gen/ncg/core/config.hpp
  @ONLY)
target_include_directories(ncg-core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gen>)
