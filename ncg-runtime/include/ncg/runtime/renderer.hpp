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

/// Renders a Gaussian cloud from a camera via the Phase-1 forward splatter (custom CUDA
/// kernel, no gradients). The cloud must be on a CUDA device.
RenderOutput render_gaussians(const recon::GaussianCloud& gaussians, const Camera& camera,
                              std::array<float, 3> background = {0.0F, 0.0F, 0.0F});

/// Fully-differentiable soft (normalized additive) splatter built from LibTorch ops — gradients
/// flow to positions/scales/opacity/colors via autograd. The correctness reference for the
/// custom rasterizer and the renderer used by the Gaussian fitting loop (ncg-fit). Slower than
/// render_gaussians; `chunk` bounds the per-step [chunk,H,W] memory. Output image is graph-connected.
RenderOutput render_soft(const recon::GaussianCloud& gaussians, const Camera& camera,
                         std::array<float, 3> background = {0.0F, 0.0F, 0.0F}, int64_t chunk = 256);

}  // namespace ncg::runtime
