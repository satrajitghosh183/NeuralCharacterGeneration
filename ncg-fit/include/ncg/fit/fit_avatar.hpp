#pragma once

#include <ncg/body/smplx.hpp>
#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/runtime/camera.hpp>

#include <cstdint>
#include <vector>

namespace ncg::fit {

/// One supervised video frame: the subject's SMPL-X pose in that frame, the camera it was seen
/// from, and the image. `pose_aa` is [J,3] axis-angle (joint 0 = global orientation).
struct AvatarFrame {
  Tensor pose_aa;            // [J,3]
  runtime::Camera camera;
  Tensor target;            // [3,H,W] in [0,1]
  Tensor transl;            // [3] world translation (optional; empty -> zeros)
};

struct AvatarFitConfig {
  int iterations = 4000;
  double lr_color = 2.5e-3;
  double lr_opacity = 5e-2;
  double lr_scale = 4e-3;
  double lr_rotation = 1e-3;
  double lr_position = 0.0;  // canonical position offsets; 0 keeps Gaussians on their vertices
  double lambda_dssim = 0.2;
  float init_scale = 0.015F;        // canonical Gaussian std-dev (world units)
  bool use_mask = true;             // supervise inside each frame's posed body silhouette
  bool per_view_exposure = true;    // casual frames vary in exposure / white balance

  // C2-style robust consistency: down-weight pixels/frames whose appearance disagrees with the
  // converging canonical (different outfits, junk frames, bad cameras), so the consensus look of the
  // person wins instead of a blurry average. Auto-scaled Welsch weight (IRLS) per iteration. This is
  // what lets "upload whatever casual data exists and run" actually work on heterogeneous sources.
  bool robust = false;
  double robust_k = 3.0;            // Welsch scale = robust_k · median(residual), recomputed per step

  // Adaptive density control (off by default). Densified Gaussians inherit their parent's vertex
  // binding, so they still skin. Requires lr_position > 0 to produce a position-gradient signal.
  bool densify = false;
  // Deviation regularizer λ_dev: pull each (free) splat toward its bound vertex's REST position so
  // densified splats add surface detail without flying off into floaters. 0 = off (legacy).
  double position_reg = 0.0;
  double max_dev = 0.0;  // HARD cap on a splat's distance from its bound vertex rest pos (m); 0=off
  double min_scale = 1e-3;  // floor on rendered splat std-dev (m); raise so dense splats overlap
  double opacity_floor = 0.0;  // FIX2: splats bound to well-COVERED verts may not vanish (no dark holes)
  int densify_from = 500;
  int densify_until = 2500;
  int densify_every = 200;
  double densify_grad = 5e-5;      // mean accumulated position-gradient norm to densify
  double densify_scale_frac = 0.4;  // split (vs clone) when scale exceeds this fraction of init_scale
  double prune_opacity = 0.05;
  int64_t max_gaussians = 60000;  // bounded for the (non-tiled) soft renderer's per-iter cost

  int log_every = 100;
  int dump_every = 0;
};

/// Skins a canonical Gaussian cloud to a pose given that pose's per-vertex rest→posed transforms
/// `vertex_transforms` ([V,4,4]). Positions are rigidly transformed; orientations are rotated by
/// the transform's rotation (so the anisotropic splats follow the body); scales/opacity/color are
/// pose-invariant. `binding` ([N] int64) maps each Gaussian to the SMPL-X vertex whose transform
/// skins it — required once densification breaks the 1:1 correspondence; if empty, a 1:1 binding
/// (N == V) is assumed. Returns the posed cloud, ready to render. Deployable at training + runtime.
recon::GaussianCloud deform_avatar(const recon::GaussianCloud& canonical,
                                   const Tensor& vertex_transforms, const Tensor& binding = {});

/// Result of an avatar fit: the canonical (rest-pose) cloud and, when densification ran, the
/// per-Gaussian SMPL-X vertex binding needed to skin it (pass to deform_avatar). `binding` is
/// empty for a 1:1 (non-densified) fit, where deform_avatar needs no binding.
struct AvatarFitResult {
  recon::GaussianCloud canonical;
  Tensor binding;  // [N] int64, or empty for 1:1
};

/// Trains an animatable Gaussian avatar from posed video frames. Canonical Gaussians (one per
/// SMPL-X vertex, seeded at the rest body and colored by `init_colors` [V,3]) are optimized so
/// that, skinned to each frame's pose and rendered through the anisotropic splatter, they
/// reproduce the frames (L1 + D-SSIM, per-frame exposure, body-masked). Because every frame
/// constrains the same canonical appearance, multi-pose casual video becomes multi-view evidence
/// for one avatar — the route from a textured mannequin to a real likeness. Returns the canonical
/// (rest-pose) cloud + its vertex binding; animate it with deform_avatar or the runtime.
AvatarFitResult fit_avatar(const body::SmplxModel& model, const Tensor& betas,
                           const std::vector<AvatarFrame>& frames, const Tensor& init_colors,
                           const AvatarFitConfig& cfg, record::Recorder* recorder = nullptr,
                           const Tensor& coverage = {});  // [V] per-vertex photo coverage (FIX2 floor)

}  // namespace ncg::fit
