include(NcgAddLibrary)

ncg_add_library(ncg-recon
  SOURCES
    src/gaussian_model.cpp
    src/init_from_body.cpp
    src/animate.cpp
    src/appearance.cpp
    src/inverse_render.cpp
    src/motion_style.cpp
  PUBLIC_DEPS
    ncg::core
    torch)
