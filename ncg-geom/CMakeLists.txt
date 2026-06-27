include(NcgAddLibrary)

ncg_add_library(ncg-geom
  SOURCES
    src/solve_geometry.cpp
  PUBLIC_DEPS
    ncg::core
    torch)
