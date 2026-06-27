include(NcgAddLibrary)

# Depends on ncg-runtime (differentiable renderer), so built only under CUDA (see top-level).
ncg_add_library(ncg-fit
  SOURCES
    src/fit_image.cpp
    src/fit_adaptive.cpp
    src/fit_avatar.cpp
    src/splat_bind.cpp
  PUBLIC_DEPS
    ncg::core
    ncg::recon
    ncg::runtime
    ncg::record
    ncg::body
    torch)
