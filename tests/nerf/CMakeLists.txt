# NeRF volume renderer + hybrid composite test (CUDA).
if(NCG_WITH_CUDA)
  ncg_add_test(test_nerf
    SOURCES test_nerf.cpp
    DEPS    ncg::nerf ncg::runtime ncg::record
    LABELS  "cuda;nerf")
endif()
