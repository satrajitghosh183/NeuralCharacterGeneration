# Gaussian fitting / optimization test (CUDA).
if(NCG_WITH_CUDA)
  ncg_add_test(test_fit
    SOURCES test_fit.cpp
    DEPS    ncg::fit ncg::runtime ncg::recon ncg::record
    LABELS  "cuda;fit")
endif()
