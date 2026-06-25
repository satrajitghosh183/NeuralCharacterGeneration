# Gaussian deformation / animation tests (CPU).
ncg_add_test(test_deform
  SOURCES test_deform.cpp
  DEPS    ncg::recon
  LABELS  "recon")

# Single-photo appearance capture (CPU).
ncg_add_test(test_appearance
  SOURCES test_appearance.cpp
  DEPS    ncg::recon
  LABELS  "recon")

# Body Gaussian initializer + adaptive per-vertex scale (CPU).
ncg_add_test(test_init
  SOURCES test_init.cpp
  DEPS    ncg::recon
  LABELS  "recon")

# Multi-illumination inverse rendering — the paper's core math + C1 proof (CPU).
ncg_add_test(test_inverse_render
  SOURCES test_inverse_render.cpp
  DEPS    ncg::recon
  LABELS  "recon")

# C4 — robust cross-action motion-style factorization (CPU).
ncg_add_test(test_motion_style
  SOURCES test_motion_style.cpp
  DEPS    ncg::recon
  LABELS  "recon")
