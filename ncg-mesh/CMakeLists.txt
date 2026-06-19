include(NcgAddLibrary)

# SKELETON (Phase 4). Compiles; entry points throw NCG_NOT_IMPLEMENTED.
ncg_add_library(ncg-mesh
  SOURCES src/extract.cpp
  PUBLIC_DEPS ncg::core ncg::recon torch)
