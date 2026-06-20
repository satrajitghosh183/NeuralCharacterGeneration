# Rig packaging + export tests (CPU).
ncg_add_test(test_rig
  SOURCES test_rig.cpp
  DEPS    ncg::rig
  LABELS  "rig")
