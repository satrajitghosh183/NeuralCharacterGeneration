# Gaussian deformation / animation tests (CPU).
ncg_add_test(test_deform
  SOURCES test_deform.cpp
  DEPS    ncg::recon
  LABELS  "recon")
