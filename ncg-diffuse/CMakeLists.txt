include(NcgAddLibrary)

# Phase E: native diffusion port for SDS-as-loss completion. Scheduler + SDS math first (pure
# tensor ops, CPU-tested); the UNet port (TORCH_LIBRARY / nn::Module + safetensors weights) lands
# as subsequent translation units.
ncg_add_library(ncg-diffuse
  SOURCES
    src/scheduler.cpp
    src/sds.cpp
    src/completion.cpp
    src/sd_guidance.cpp
  PUBLIC_DEPS
    ncg::core
    ncg::io
    torch)
