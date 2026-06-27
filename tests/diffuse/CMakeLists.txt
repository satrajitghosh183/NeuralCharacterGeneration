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
