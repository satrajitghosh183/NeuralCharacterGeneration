# Analytical relighting tests (CPU).
ncg_add_test(test_relight
  SOURCES test_relight.cpp
  DEPS    ncg::material
  LABELS  "material")
