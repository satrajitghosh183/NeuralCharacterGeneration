#pragma once

#include <ncg/core/tensor.hpp>

#include <array>
#include <utility>

namespace ncg::runtime {

/// Phase-1 forward splatting (CUDA). One thread per pixel composites all Gaussians
/// front-to-back; no tiling, no backward pass. Inputs are expected pre-sorted near->far.
///   u, v       : [N] projected pixel centers
///   inv_sigma2 : [N] 1 / sigma_px^2 (screen-space)
///   opacity    : [N] in [0,1]
///   colors     : [N,3] linear RGB
/// Returns {image [3,H,W], alpha [1,H,W]} on the same CUDA device.
std::pair<Tensor, Tensor> splat_render_cuda(const Tensor& u, const Tensor& v,
                                            const Tensor& inv_sigma2, const Tensor& opacity,
                                            const Tensor& colors, int height, int width,
                                            const std::array<float, 3>& background);

}  // namespace ncg::runtime
