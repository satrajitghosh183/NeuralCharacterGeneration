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

  ncg_add_test(test_gs_skinning
    SOURCES test_gs_skinning.cpp
    DEPS    ncg::fit ncg::recon
    LABELS  "cuda;fit")

  # Phase C: k-NN free-splat binding + anti-swim metric.
  ncg_add_test(test_swim
    SOURCES test_swim.cpp
    DEPS    ncg::fit
    LABELS  "cuda;fit")
endif()
