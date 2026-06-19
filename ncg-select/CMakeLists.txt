include(NcgAddLibrary)

ncg_add_library(ncg-select
  SOURCES
    src/selector.cpp
  PUBLIC_DEPS
    ncg::core
    torch
  PRIVATE_DEPS
    ncg::io)
