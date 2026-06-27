#include <ncg/diffuse/sds.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <optional>

namespace ncg::diffuse {

Tensor cfg_eps(const Tensor& eps_uncond, const Tensor& eps_cond, float guidance) {
  // eps = eps_uncond + g*(eps_cond - eps_uncond): push the prediction toward the conditioned score.
  return eps_uncond + guidance * (eps_cond - eps_uncond);
}

SdsResult sds_loss(const Tensor& x, const DdpmSchedule& schedule, const NoisePredictor& eps_theta,
                   const SdsConfig& cfg, at::Generator gen) {
  NCG_CHECK(x.dim() >= 2, "sds_loss: x must be batched [B, ...]");
  NCG_CHECK(cfg.t_min >= 0 && cfg.t_max <= schedule.num_timesteps() && cfg.t_min < cfg.t_max,
            "sds_loss: t range out of bounds");
  const int64_t B = x.size(0);
  const std::optional<at::Generator> g = gen.defined() ? std::optional<at::Generator>(gen)
                                                       : std::nullopt;

  // Random timestep per sample + fresh Gaussian noise (the forward-diffusion draw).
  const auto t = torch::randint(cfg.t_min, cfg.t_max, {B}, g,
                                x.options().dtype(at::kLong).device(x.device()));
  const auto noise = torch::randn(x.sizes(), g, x.options());

  // Noise prediction is OUTSIDE the avatar's autograd graph (DreamFusion: don't differentiate the
  // UNet — the SDS gradient already IS the useful direction). no_grad keeps it cheap + correct.
  Tensor eps_hat;
  {
    torch::NoGradGuard ng;
    const auto x_t = schedule.add_noise(x.detach(), noise, t);
    eps_hat = eps_theta(x_t, t);
    NCG_CHECK(eps_hat.sizes() == x.sizes(), "sds_loss: predictor output shape must match x");
  }

  // grad_SDS = w(t) * (eps_hat - eps), sanitized (a single NaN otherwise poisons clip_grad_norm).
  auto grad = schedule.sds_weight(t, x.dim()).to(x.dtype()) * (eps_hat - noise);
  grad = torch::nan_to_num(grad, 0.0, 0.0, 0.0);
  if (cfg.clip_grad) {
    grad = grad.clamp(-cfg.grad_clip, cfg.grad_clip);
  }
  grad = grad.detach();

  // Surrogate scalar (threestudio convention): target = (x - grad).detach(); 0.5*||x - target||^2/B.
  // Its autograd gradient w.r.t. x is exactly grad/B, so it composes in a single .backward() with the
  // photometric loss and carries SDS into the renderer's parameters.
  const auto target = (x - grad).detach();
  const auto loss = 0.5 * torch::nn::functional::mse_loss(
                              x, target, torch::nn::functional::MSELossFuncOptions().reduction(
                                             torch::kSum)) /
                    static_cast<double>(B);

  SdsResult r;
  r.loss = loss;
  r.grad = grad;
  r.t = t;
  r.grad_norm = grad.reshape({B, -1}).norm(2, 1).mean().item<double>();
  return r;
}

}  // namespace ncg::diffuse
