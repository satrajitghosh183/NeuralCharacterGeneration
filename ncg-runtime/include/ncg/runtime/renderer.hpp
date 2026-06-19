#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/runtime/camera.hpp>

#include <array>

namespace ncg::runtime {

struct RenderOutput {
  Tensor image;  // [3,H,W] in [0,1]
  Tensor alpha;  // [1,H,W] in [0,1]
};

/// Renders a Gaussian cloud from a camera via the Phase-1 forward splatter. The cloud must be
/// on a CUDA device.
RenderOutput render_gaussians(const recon::GaussianCloud& gaussians, const Camera& camera,
                              std::array<float, 3> background = {0.0F, 0.0F, 0.0F});

}  // namespace ncg::runtime
