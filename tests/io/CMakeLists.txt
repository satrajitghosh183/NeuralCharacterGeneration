# CPU-only IO tests (safetensors, npy, image round-trips).
ncg_add_test(test_io
  SOURCES
    test_safetensors.cpp
    test_npy.cpp
    test_image.cpp
  DEPS   ncg::io
  LABELS "io")
