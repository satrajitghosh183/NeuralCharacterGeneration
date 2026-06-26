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

/// Fully-differentiable *anisotropic* EWA splatter. Unlike render_soft (which collapses each
/// Gaussian to an isotropic screen disk from the mean scale and ignores rotation), this projects
/// the full 3D covariance Σ = R diag(s)² Rᵀ through the perspective Jacobian to a 2D conic, so
/// Gaussians can be oriented and elongated and gradients flow to the rotation quaternions. This is
/// the reconstruction-quality renderer used by the adaptive fitter (oriented splats resolve hair,
/// edges and the face that isotropic disks blur). `dilation` is the screen-space low-pass added to
/// the 2D covariance diagonal (3DGS uses ~0.3 px). Output image is graph-connected. CUDA or CPU.
RenderOutput render_soft_aniso(const recon::GaussianCloud& gaussians, const Camera& camera,
                               std::array<float, 3> background = {0.0F, 0.0F, 0.0F},
                               int64_t chunk = 256, float dilation = 0.3F);

}  // namespace ncg::runtime
