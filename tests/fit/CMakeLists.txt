# Gaussian fitting / optimization test (CUDA).
if(NCG_WITH_CUDA)
  ncg_add_test(test_fit
    SOURCES test_fit.cpp
    DEPS    ncg::fit ncg::runtime ncg::recon ncg::record
    LABELS  "cuda;fit")

  ncg_add_test(test_fit_adaptive
    SOURCES test_fit_adaptive.cpp
    DEPS    ncg::fit ncg::runtime ncg::recon ncg::record
    LABELS  "cuda;fit")

  ncg_add_test(test_avatar
    SOURCES test_avatar.cpp
    DEPS    ncg::fit ncg::recon
    LABELS  "cuda;fit")
endif()
