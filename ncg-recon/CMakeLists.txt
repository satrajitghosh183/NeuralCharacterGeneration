include(NcgAddLibrary)

ncg_add_library(ncg-recon
  SOURCES
    src/gaussian_model.cpp
    src/init_from_body.cpp
    src/animate.cpp
    src/appearance.cpp
  PUBLIC_DEPS
    ncg::core
    torch)
