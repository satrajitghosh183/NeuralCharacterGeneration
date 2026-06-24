# Marching cubes + mesh export tests (CPU).
ncg_add_test(test_mesh
  SOURCES test_mesh.cpp
  DEPS    ncg::mesh ncg::recon
  LABELS  "mesh")

# glTF (.glb) export — static + skinned validity (CPU).
ncg_add_test(test_glb
  SOURCES test_glb.cpp
  DEPS    ncg::mesh
  LABELS  "mesh")
