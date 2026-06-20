# Marching cubes + mesh export tests (CPU).
ncg_add_test(test_mesh
  SOURCES test_mesh.cpp
  DEPS    ncg::mesh ncg::recon
  LABELS  "mesh")
