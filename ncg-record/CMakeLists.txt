include(NcgAddLibrary)

ncg_add_library(ncg-record
  SOURCES
    src/metrics.cpp
    src/recorder.cpp
  PUBLIC_DEPS
    ncg::core
    torch
  PRIVATE_DEPS
    ncg::io
    nlohmann_json::nlohmann_json)
