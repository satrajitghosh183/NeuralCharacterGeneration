# Renderer smoke test (CUDA).
if(NCG_WITH_CUDA)
  ncg_add_test(test_render
    SOURCES test_render.cpp
    DEPS    ncg::runtime ncg::recon
    LABELS  "cuda;runtime")
endif()
