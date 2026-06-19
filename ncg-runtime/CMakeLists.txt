include(NcgAddLibrary)

set(_rt_private "")
if(NCG_WITH_CUDA)
  list(APPEND _rt_private CUDA::cudart)
endif()

ncg_add_library(ncg-runtime
  SOURCES
    src/camera.cpp
    src/renderer.cpp
  CUDA_SOURCES
    src/kernels/splat_forward.cu
  PUBLIC_DEPS
    ncg::core
    ncg::recon
    torch
  PRIVATE_DEPS
    ${_rt_private})
