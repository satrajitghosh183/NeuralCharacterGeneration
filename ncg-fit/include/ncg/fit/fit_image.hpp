#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/runtime/camera.hpp>

#include <cstdint>

namespace ncg::fit {

struct FitConfig {
  int iterations = 300;
  double lr = 0.02;
  int64_t num_gaussians = 2000;
  float init_scale = 0.03F;
  float init_spread = 0.3F;  // std-dev of initial Gaussian positions around the origin
  int log_every = 25;        // record loss/psnr cadence
};

/// Optimizes a Gaussian cloud (positions, scales, opacities, colors) to reproduce `target_chw`
/// rendered from `camera`, using the differentiable soft renderer + Adam. This is the Phase-2
/// 3DGS optimization core, correct-by-construction via autograd. Records per-iteration loss /
/// PSNR / SSIM (and periodic image dumps) to `recorder` when provided. Returns the fitted cloud.
recon::GaussianCloud fit_gaussians_to_image(const Tensor& target_chw,
                                            const runtime::Camera& camera, const FitConfig& cfg,
                                            record::Recorder* recorder = nullptr);

}  // namespace ncg::fit
