#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/runtime/camera.hpp>

#include <cstdint>
#include <vector>

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

struct RefineConfig {
  int iterations = 200;
  double lr = 0.01;
  float mask_threshold = 0.1F;  // body silhouette taken from the initial render's alpha
  int log_every = 25;
};

/// Per-subject 3DGS refinement: starting from an initial cloud (e.g. the photo-colored body in
/// its source-photo frame), optimize colors/opacity/scale/position to match `target_chw` from
/// `camera` via the differentiable renderer + Adam. The loss is confined to the body silhouette
/// (the init render's alpha) so the background doesn't pull the fit. This sharpens the
/// single-sample-per-vertex appearance into photographic detail. Returns the refined cloud.
recon::GaussianCloud refine_gaussians_to_image(const recon::GaussianCloud& init,
                                               const Tensor& target_chw,
                                               const runtime::Camera& camera,
                                               const RefineConfig& cfg,
                                               record::Recorder* recorder = nullptr);

/// Multi-view variant: optimizes a single Gaussian cloud to reproduce several target images
/// from their respective cameras (Phase-3 multi-view reconstruction). `targets[i]` is rendered
/// from `cameras[i]`; the per-iteration loss is summed across views. Sizes must match.
recon::GaussianCloud fit_gaussians_to_views(const std::vector<Tensor>& targets,
                                            const std::vector<runtime::Camera>& cameras,
                                            const FitConfig& cfg,
                                            record::Recorder* recorder = nullptr);

}  // namespace ncg::fit
