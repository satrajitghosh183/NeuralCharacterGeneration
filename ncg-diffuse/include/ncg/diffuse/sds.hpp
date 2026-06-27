#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/diffuse/scheduler.hpp>

#include <functional>

namespace ncg::diffuse {

// ============================================================================================
// Score Distillation Sampling as a LOSS (docs/method.md §M10). The completion prior: an image (or
// latent) x being optimized — here our DIFFERENTIABLE RENDER of the avatar — is pushed toward the
// diffusion model's data manifold without ever sampling the diffusion chain. For a random timestep t
// and noise eps:
//     x_t = sqrt(abar_t) x + sqrt(1-abar_t) eps
//     grad_SDS = w(t) * (eps_theta(x_t, t, c) - eps)          [DreamFusion]
// This gradient is attached to x so backprop carries it into the renderer's parameters. We expose it
// two ways: the raw gradient (to add to an existing grad), and a SURROGATE SCALAR LOSS whose
// autograd gradient equals grad_SDS (so it composes with the photometric loss in one .backward()).
//
// `eps_theta` is the ported UNet's noise prediction with classifier-free guidance already folded in
// (eps = eps_uncond + g*(eps_cond - eps_uncond)); see cfg_eps(). Kept as a std::function so this math
// is UNIT-TESTABLE on CPU with an analytic stand-in denoiser before the real UNet port lands.
// ============================================================================================

// A denoiser: given the noised sample x_t [B,C,H,W] and timesteps t [B], predict the noise eps_hat.
using NoisePredictor = std::function<Tensor(const Tensor& x_t, const Tensor& t)>;

struct SdsConfig {
  int t_min = 20;            // sample t in [t_min, t_max) — avoid the degenerate endpoints
  int t_max = 980;
  float guidance = 100.0F;   // classifier-free guidance scale (DreamFusion uses large g ~100)
  bool clip_grad = true;     // clamp the per-element SDS grad for stability
  float grad_clip = 1.0F;
};

struct SdsResult {
  Tensor loss;     // [] surrogate scalar; .backward() yields grad_SDS on x (and its upstream params)
  Tensor grad;     // [B,C,H,W] the raw SDS gradient w(t)*(eps_hat - eps), detached
  Tensor t;        // [B] the sampled timesteps (for logging)
  double grad_norm;  // mean L2 of the per-sample gradient (a convergence signal)
};

// Classifier-free guidance combine: eps = eps_uncond + g*(eps_cond - eps_uncond).
Tensor cfg_eps(const Tensor& eps_uncond, const Tensor& eps_cond, float guidance);

// Compute the SDS loss/gradient for sample `x` (requires_grad recommended). `gen` supplies the
// random timestep+noise reproducibly in tests; pass an unseeded generator in production.
SdsResult sds_loss(const Tensor& x, const DdpmSchedule& schedule, const NoisePredictor& eps_theta,
                   const SdsConfig& cfg = {}, at::Generator gen = {});

}  // namespace ncg::diffuse
