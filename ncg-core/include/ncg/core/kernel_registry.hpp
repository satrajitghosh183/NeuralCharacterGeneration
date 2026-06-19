#pragma once

#include <cstdint>

// Custom-kernel conventions for NeuralCharGen. Two tiers:
//
//  Tier 1 (DEFAULT — use for everything that doesn't need autograd):
//    A plain C++ free function declared in a module header, implemented in a .cu:
//        Tensor saxpy(const Tensor& x, const Tensor& y, double a);
//    The .cu validates inputs, allocates outputs via at::empty_like, computes a launch
//    grid, launches the __global__, calls NCG_CUDA_KERNEL_CHECK(), returns. Trivially
//    unit-testable against a LibTorch reference op; no dispatcher ceremony.
//
//  Tier 2 (only when autograd through a custom kernel is required, e.g. the Phase 2
//          Gaussian rasterizer backward):
//    Register a LibTorch custom op + a torch::autograd::Function:
//        TORCH_LIBRARY(ncg, m)      { m.def("splat_forward(...) -> ..."); }
//        TORCH_LIBRARY_IMPL(ncg, CUDA, m) { m.impl("splat_forward", splat_forward_cuda); }
//    and wrap fwd/bwd in a torch::autograd::Function<SplatRaster>.
//
// This header only holds shared launch helpers; the registration macros come from
// <torch/library.h> at the call site.

namespace ncg {

/// Number of blocks to cover `n` elements with the given block size (ceil-div).
inline unsigned grid_1d(int64_t n, unsigned block) {
  return static_cast<unsigned>((n + static_cast<int64_t>(block) - 1) / static_cast<int64_t>(block));
}

}  // namespace ncg
