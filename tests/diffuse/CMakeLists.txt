# Phase E: diffusion schedule + SDS-as-loss math, proven on CPU against closed-form identities and
# an analytic stand-in denoiser (no UNet weights needed yet).
ncg_add_test(test_sds
  SOURCES test_sds.cpp
  DEPS    ncg::diffuse
  LABELS  "diffuse")

ncg_add_test(test_completion
  SOURCES test_completion.cpp
  DEPS    ncg::diffuse
  LABELS  "diffuse")

# SD TorchScript contract smoke test (CUDA; skips unless NCG_SD_DIR points at the exported assets).
ncg_add_test(test_sd_smoke
  SOURCES test_sd_smoke.cpp
  DEPS    ncg::diffuse
  LABELS  "cuda;diffuse")
