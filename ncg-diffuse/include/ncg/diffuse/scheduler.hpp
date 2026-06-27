#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::diffuse {

// ============================================================================================
// PHASE E INTERFACE CONTRACT (frozen). Render-consistent completion via SDS-as-loss with a NATIVE
// C++/CUDA diffusion port (no Python ever — the user's "purest" choice). This file is the noise
// schedule: the diffusion forward process q(x_t | x_0) and the coefficients SDS needs. It is PURE
// tensor math (no network weights), so it is proven on CPU against the closed-form DDPM identities
// before the UNet port (the big sub-build) lands.
//
// Convention matches Stable-Diffusion / 🤗diffusers DDPMScheduler so ported SD weights are valid:
//   beta_schedule = "scaled_linear": beta_t = (sqrt(b0) + (t/(T-1))(sqrt(bT) - sqrt(b0)))^2
//   alpha_t = 1 - beta_t ;  alphabar_t = prod_{s<=t} alpha_s
//   x_t = sqrt(alphabar_t) x_0 + sqrt(1 - alphabar_t) eps,  eps ~ N(0, I)
// ============================================================================================

struct ScheduleConfig {
  int num_train_timesteps = 1000;
  float beta_start = 0.00085F;  // SD v1.x "scaled_linear" defaults
  float beta_end = 0.012F;
};

// Precomputed per-timestep coefficient tables (length num_train_timesteps), on the given device.
class DdpmSchedule {
 public:
  explicit DdpmSchedule(const ScheduleConfig& cfg = {}, at::Device device = at::kCPU);

  // Forward diffusion: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) noise. `t` is a [B] long tensor of
  // timestep indices in [0, T); x0/noise are [B, ...]. Broadcasts the per-sample coefficients.
  Tensor add_noise(const Tensor& x0, const Tensor& noise, const Tensor& t) const;

  // Per-timestep gathers ([B,1,1,...] broadcastable against an image/latent of `ndim` dims).
  Tensor sqrt_alpha_bar(const Tensor& t, int64_t ndim) const;        // sqrt(abar_t)
  Tensor sqrt_one_minus_alpha_bar(const Tensor& t, int64_t ndim) const;  // sqrt(1 - abar_t)

  // SDS weight w(t). Default DreamFusion uses w(t) = (1 - abar_t) (== sigma_t^2); this returns that.
  Tensor sds_weight(const Tensor& t, int64_t ndim) const;

  const Tensor& alphas_cumprod() const { return alphas_cumprod_; }
  const Tensor& betas() const { return betas_; }
  int num_timesteps() const { return cfg_.num_train_timesteps; }

 private:
  ScheduleConfig cfg_;
  Tensor betas_;           // [T]
  Tensor alphas_cumprod_;  // [T] abar_t
};

}  // namespace ncg::diffuse
