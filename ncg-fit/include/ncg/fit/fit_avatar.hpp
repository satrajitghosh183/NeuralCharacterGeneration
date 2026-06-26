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
  int log_every = 100;
  int dump_every = 0;
};

/// Skins a canonical Gaussian cloud (bound 1:1 to SMPL-X vertices, so N == V) to a pose given that
/// pose's per-vertex rest→posed transforms `vertex_transforms` ([V,4,4]). Positions are rigidly
/// transformed; orientations are rotated by the transform's rotation (so the anisotropic splats
/// follow the body); scales/opacity/color are pose-invariant. Returns the posed cloud, ready to
/// render. This is the deployable deformation used both in training and at runtime.
recon::GaussianCloud deform_avatar(const recon::GaussianCloud& canonical,
                                   const Tensor& vertex_transforms);

/// Trains an animatable Gaussian avatar from posed video frames. Canonical Gaussians (one per
/// SMPL-X vertex, seeded at the rest body and colored by `init_colors` [V,3]) are optimized so
/// that, skinned to each frame's pose and rendered through the anisotropic splatter, they
/// reproduce the frames (L1 + D-SSIM, per-frame exposure, body-masked). Because every frame
/// constrains the same canonical appearance, multi-pose casual video becomes multi-view evidence
/// for one avatar — the route from a textured mannequin to a real likeness. Returns the canonical
/// (rest-pose) cloud; animate it with deform_avatar or the runtime.
recon::GaussianCloud fit_avatar(const body::SmplxModel& model, const Tensor& betas,
                                const std::vector<AvatarFrame>& frames, const Tensor& init_colors,
                                const AvatarFitConfig& cfg, record::Recorder* recorder = nullptr);

}  // namespace ncg::fit
