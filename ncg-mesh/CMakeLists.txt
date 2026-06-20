include(NcgAddLibrary)

ncg_add_library(ncg-mesh
  SOURCES
    src/extract.cpp
    src/marching_cubes.cpp
    src/mesh_io.cpp
  PUBLIC_DEPS ncg::core ncg::recon torch)
