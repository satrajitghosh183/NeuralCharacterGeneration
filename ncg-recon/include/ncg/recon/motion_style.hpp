#pragma once

#include <ncg/core/tensor.hpp>

#include <vector>

namespace ncg::recon {

/// C4 — personal motion-style recovery (docs/method.md §13): the pose-manifold analog of the
/// appearance solver. Given several casual motion clips of ONE person performing DIFFERENT actions
/// (each clip a pose-feature sequence [T_c, D], e.g. flattened SMPL-X joint rotations), recover the
/// person's **style** as the low-rank subspace shared across their clips, with per-clip **content**
/// coefficients. Robust frame weights reject cut frames / bad poses (the noisy-casual-video analog
/// of C2). Style is identifiable up to an R×R gauge given >=2 distinct actions; compare via the
/// row-space (projection) it spans, which is gauge-invariant.
struct MotionStyleConfig {
  int rank = 8;             // R: style/content latent dimension
  int iterations = 60;     // ALS sweeps
  bool robust = true;      // per-frame Welsch weights (reject cuts/outliers)
  float robust_scale = 0.0F;  // 0 => auto (1.4826 * MAD of residuals)
  float ridge = 1e-3F;
};

struct MotionStyleResult {
  Tensor style;                    // [R, D] shared style subspace (the personal signature)
  std::vector<Tensor> contents;    // per-clip content coefficients [T_c, R]
  std::vector<Tensor> consistency; // per-clip per-frame trust weights [T_c]
};

MotionStyleResult solve_motion_style(const std::vector<Tensor>& clips,
                                     const MotionStyleConfig& cfg);

/// Project pose features `poses` [T,D] onto a recovered style subspace `style` [R,D] — i.e. render
/// a (new, held-out) action *in this person's style*. Returns [T,D]. This is the transfer payoff:
/// content from anywhere, expressed through their motion signature.
Tensor apply_motion_style(const Tensor& poses, const Tensor& style);

}  // namespace ncg::recon
