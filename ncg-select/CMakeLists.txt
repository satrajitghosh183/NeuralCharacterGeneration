include(NcgAddLibrary)

ncg_add_library(ncg-select
  SOURCES
    src/selector.cpp
    src/subject_gate.cpp
  PUBLIC_DEPS
    ncg::core
    torch
  PRIVATE_DEPS
    ncg::io)
