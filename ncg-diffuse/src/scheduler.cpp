#include <ncg/diffuse/scheduler.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <cmath>
#include <vector>

namespace ncg::diffuse {

namespace {
// Reshape a per-sample [B] vector to [B,1,1,...] so it broadcasts against an `ndim`-dim tensor.
Tensor to_broadcast(const Tensor& v, int64_t ndim) {
  std::vector<int64_t> shape(static_cast<size_t>(ndim), 1);
  shape[0] = v.size(0);
  return v.reshape(shape);
}
}  // namespace

DdpmSchedule::DdpmSchedule(const ScheduleConfig& cfg, at::Device device) : cfg_(cfg) {
  NCG_CHECK(cfg.num_train_timesteps > 0, "num_train_timesteps must be positive");
  const auto opts = at::TensorOptions().dtype(at::kDouble).device(device);
  // "scaled_linear": betas = linspace(sqrt(b0), sqrt(bT))^2 — SD's convention (more steps near 0).
  const auto beta_sqrt = torch::linspace(std::sqrt(static_cast<double>(cfg.beta_start)),
                                         std::sqrt(static_cast<double>(cfg.beta_end)),
                                         cfg.num_train_timesteps, opts);
  betas_ = beta_sqrt * beta_sqrt;                          // [T]
  const auto alphas = 1.0 - betas_;                        // [T]
  alphas_cumprod_ = torch::cumprod(alphas, /*dim=*/0);     // [T] abar_t in (0,1], decreasing
  betas_ = betas_.to(at::kFloat);
  alphas_cumprod_ = alphas_cumprod_.to(at::kFloat);
}

Tensor DdpmSchedule::sqrt_alpha_bar(const Tensor& t, int64_t ndim) const {
  const auto abar = alphas_cumprod_.to(t.device()).index_select(0, t.to(at::kLong));
  return to_broadcast(torch::sqrt(abar), ndim);
}

Tensor DdpmSchedule::sqrt_one_minus_alpha_bar(const Tensor& t, int64_t ndim) const {
  const auto abar = alphas_cumprod_.to(t.device()).index_select(0, t.to(at::kLong));
  return to_broadcast(torch::sqrt((1.0 - abar).clamp_min(0.0)), ndim);
}

Tensor DdpmSchedule::sds_weight(const Tensor& t, int64_t ndim) const {
  const auto abar = alphas_cumprod_.to(t.device()).index_select(0, t.to(at::kLong));
  return to_broadcast(1.0 - abar, ndim);  // w(t) = 1 - abar_t (DreamFusion default)
}

Tensor DdpmSchedule::add_noise(const Tensor& x0, const Tensor& noise, const Tensor& t) const {
  NCG_CHECK(x0.sizes() == noise.sizes(), "add_noise: x0 and noise must share shape");
  NCG_CHECK(t.size(0) == x0.size(0), "add_noise: t batch must match x0 batch");
  const auto sa = sqrt_alpha_bar(t, x0.dim()).to(x0.dtype());
  const auto so = sqrt_one_minus_alpha_bar(t, x0.dim()).to(x0.dtype());
  return sa * x0 + so * noise;
}

}  // namespace ncg::diffuse
