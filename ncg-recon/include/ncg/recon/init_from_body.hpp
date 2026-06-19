#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>

namespace ncg::recon {

/// Initializes one Gaussian per body vertex (the Phase-1 slice initializer — no optimization,
/// no fusion). Scale is isotropic; rotation identity; opacity 1.
///   vertices : [V,3] world-space SMPL-X vertices
///   scale    : isotropic world-space std-dev per Gaussian
///   colors   : [V,3] linear RGB in [0,1], or an undefined tensor for a default gray
GaussianCloud gaussians_on_body(const Tensor& vertices, float scale,
                                const Tensor& colors = {});

}  // namespace ncg::recon
