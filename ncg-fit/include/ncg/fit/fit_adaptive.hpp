#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/runtime/camera.hpp>

#include <cstdint>
#include <vector>

namespace ncg::fit {

/// Configuration for the adaptive (full 3DGS-style) optimizer. Defaults are tuned for a
/// human-scale scene (extent ~1–2 m) and a few hundred to a few thousand source views.
struct AdaptiveFitConfig {
  int iterations = 3000;
  // Per-parameter learning rates (Adam). Positions move slowly; opacity fast.
  double lr_position = 4e-4;
  double lr_scale = 5e-3;
  double lr_rotation = 1e-3;
  double lr_color = 2.5e-3;
  double lr_opacity = 5e-2;

  double lambda_dssim = 0.2;  // loss = (1-λ)·L1 + λ·(1 - SSIM)

  // Adaptive density control (the 3DGS quality lever).
  int densify_from = 300;       // start densifying after this many iters (warm-up)
  int densify_until = 2200;     // stop densifying before final convergence
  int densify_every = 100;      // densify/prune cadence
  double grad_threshold = 3e-5;  // mean accumulated positional-gradient norm to densify a Gaussian
  double split_scale_frac = 0.01;  // scale (rel. to scene extent) above which we split, else clone
  double split_jitter = 1.6;       // child scale divisor when splitting
  int opacity_reset_every = 600;   // periodically knock opacity down to cull floaters
  double prune_opacity = 0.05;     // prune Gaussians dimmer than this
  int64_t max_gaussians = 400000;  // safety cap

  bool per_view_exposure = false;  // learn a per-view RGB gain+bias (casual-photo exposure/WB)
  bool use_mask = true;            // supervise only inside each view's body silhouette
  int log_every = 100;
  int dump_every = 0;  // if >0 and recorder set, dump a render of view 0 every N iters
};

/// Adaptive 3DGS fit: optimizes a Gaussian cloud (positions, anisotropic scales, rotations,
/// opacity, color) against several posed target images via the differentiable anisotropic EWA
/// splatter + Adam, with adaptive density control (clone/split high-gradient Gaussians, prune
/// transparent ones, periodic opacity reset). One random view is supervised per iteration. This
/// is the reconstruction-quality path: oriented splats + densification resolve the face, hair and
/// edges that the fixed-count isotropic fitter blurs. `init` seeds geometry (e.g. the SMPL-X body
/// surface) and is required — densification refines detail, it does not invent global structure.
recon::GaussianCloud fit_adaptive(const std::vector<Tensor>& targets,
                                  const std::vector<runtime::Camera>& cameras,
                                  const recon::GaussianCloud& init, const AdaptiveFitConfig& cfg,
                                  record::Recorder* recorder = nullptr);

/// Differentiable structural similarity between two [3,H,W] images in [0,1] (11×11 Gaussian
/// window). Exposed for the D-SSIM loss term and for tests. Returns a scalar in (−1,1].
Tensor ssim(const Tensor& a, const Tensor& b);

}  // namespace ncg::fit
