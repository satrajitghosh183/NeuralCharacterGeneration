include(NcgAddLibrary)

# SKELETON (later phase). Compiles; entry points throw NCG_NOT_IMPLEMENTED.
ncg_add_library(ncg-rig
  SOURCES src/rig.cpp
  PUBLIC_DEPS ncg::core torch)
