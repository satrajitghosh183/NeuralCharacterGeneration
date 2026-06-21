include(NcgAddLibrary)

# Uses runtime::Camera/RenderOutput, so built only under CUDA (see top-level).
ncg_add_library(ncg-nerf
  SOURCES
    src/nerf.cpp
  PUBLIC_DEPS
    ncg::core
    ncg::runtime
    ncg::record
    torch)
