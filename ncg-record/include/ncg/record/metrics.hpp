#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::record {

/// Mean absolute error between two same-shaped tensors.
double mae(const Tensor& a, const Tensor& b);

/// Peak signal-to-noise ratio (dB) for images in [0, max_val]. Returns a large finite value
/// when the inputs are identical.
double psnr(const Tensor& a, const Tensor& b, double max_val = 1.0);

/// Structural similarity (mean over channels) using an 11x11 Gaussian window. Inputs are
/// CHW or [B,C,H,W] float tensors in [0, max_val].
double ssim(const Tensor& a, const Tensor& b, double max_val = 1.0);

}  // namespace ncg::record
