# CUDA kernel correctness test: custom saxpy vs LibTorch reference. Only meaningful with CUDA.
if(NCG_WITH_CUDA)
  ncg_add_test(test_elementwise_kernel
    SOURCES test_elementwise_kernel.cpp
    DEPS    ncg::core
    LABELS  "cuda;core")
endif()
