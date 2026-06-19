#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg {

/// Computes `a * x + y` out-of-place on CUDA. Inputs must be float32, contiguous, CUDA,
/// and the same shape. Exists primarily to validate the CUDA + LibTorch + CMake toolchain
/// and the "custom kernel vs LibTorch reference" test pattern.
Tensor saxpy(const Tensor& x, const Tensor& y, double a);

}  // namespace ncg
