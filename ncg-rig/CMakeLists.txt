include(NcgAddLibrary)

# make_rigged + export_rigged are real; autorig (UniRig, learned) stays gated.
ncg_add_library(ncg-rig
  SOURCES src/rig.cpp
  PUBLIC_DEPS ncg::core torch
  PRIVATE_DEPS nlohmann_json::nlohmann_json)
