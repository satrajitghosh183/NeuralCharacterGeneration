include(NcgAddLibrary)

ncg_add_library(ncg-body
  SOURCES
    src/smplx.cpp
    src/nlf.cpp
    src/face_models.cpp
    src/album_gate.cpp
  PUBLIC_DEPS
    ncg::core
    torch
  PRIVATE_DEPS
    ncg::io
    ncg::select)
