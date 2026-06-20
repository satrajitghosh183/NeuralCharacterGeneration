include(NcgAddLibrary)

# relight.cpp is a real analytical shader; intrinsic.cpp (learned decomposition) stays gated.
ncg_add_library(ncg-material
  SOURCES
    src/intrinsic.cpp
    src/relight.cpp
  PUBLIC_DEPS ncg::core torch)
