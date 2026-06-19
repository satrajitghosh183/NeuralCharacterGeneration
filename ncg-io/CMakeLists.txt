include(NcgAddLibrary)

ncg_add_library(ncg-io
  SOURCES
    src/image.cpp
    src/stb_impl.cpp
    src/safetensors.cpp
    src/weight_loader.cpp
    src/npy.cpp
  PUBLIC_DEPS
    ncg::core
    torch
  PRIVATE_DEPS
    nlohmann_json::nlohmann_json
    ncg::stb)
