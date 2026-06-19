include(NcgAddLibrary)

# SKELETON (Phase 4). Compiles; entry points throw NCG_NOT_IMPLEMENTED.
ncg_add_library(ncg-material
  SOURCES src/intrinsic.cpp
  PUBLIC_DEPS ncg::core torch)
