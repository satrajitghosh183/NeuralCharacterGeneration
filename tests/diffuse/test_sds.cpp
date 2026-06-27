#include <catch2/catch_test_macros.hpp>

#include <ncg/diffuse/scheduler.hpp>
#include <ncg/diffuse/sds.hpp>

#include <torch/torch.h>

// Phase E foundation (docs/method.md §M10). The diffusion schedule + SDS-as-loss math, proven on CPU
// against closed-form identities and an ANALYTIC stand-in denoiser — no UNet weights yet. Once green,
// swapping in the ported UNet's eps_theta is a drop-in: the gradient plumbing is already verified.

using namespace ncg::diffuse;

TEST_CASE("schedule: DDPM coefficients obey the closed-form identities", "[diffuse]") {
  DdpmSchedule sch;  // SD scaled_linear defaults, 1000 steps
  const auto abar = sch.alphas_cumprod();
  REQUIRE(abar.size(0) == 1000);

  // abar in (0,1], strictly decreasing (cumulative product of alphas<1).
  REQUIRE(abar.max().item<float>() <= 1.0F + 1e-6F);
  REQUIRE(abar.min().item<float>() > 0.0F);
  const auto diff = abar.slice(0, 1) - abar.slice(0, 0, 999);
  REQUIRE(diff.max().item<float>() < 0.0F);  // every step strictly smaller

  // Variance-preserving: sqrt(abar)^2 + sqrt(1-abar)^2 == 1 for every t.
  const auto t = torch::arange(1000, at::kLong);
  const auto sa = sch.sqrt_alpha_bar(t, 1).squeeze();
  const auto so = sch.sqrt_one_minus_alpha_bar(t, 1).squeeze();
  REQUIRE((sa * sa + so * so - 1.0).abs().max().item<float>() < 1e-5F);
}

TEST_CASE("schedule: add_noise matches sqrt(abar)x0 + sqrt(1-abar)eps", "[diffuse]") {
  DdpmSchedule sch;
  torch::manual_seed(0);
  const auto x0 = torch::randn({3, 4, 8, 8});
  const auto noise = torch::randn({3, 4, 8, 8});
  const auto t = torch::tensor({10, 500, 900}, at::kLong);
  const auto xt = sch.add_noise(x0, noise, t);
  const auto sa = sch.sqrt_alpha_bar(t, x0.dim());
  const auto so = sch.sqrt_one_minus_alpha_bar(t, x0.dim());
  REQUIRE((xt - (sa * x0 + so * noise)).abs().max().item<float>() < 1e-6F);
}

TEST_CASE("sds: a PERFECT denoiser yields zero SDS gradient", "[diffuse]") {
  DdpmSchedule sch;
  torch::manual_seed(1);
  const auto x = torch::randn({2, 4, 8, 8});

  // A denoiser that reconstructs the exact injected noise from x_t given x0:
  //   x_t = sa*x0 + so*eps  =>  eps = (x_t - sa*x0)/so. eps_hat == eps => grad_SDS == 0.
  const NoisePredictor perfect = [&](const Tensor& x_t, const Tensor& t) {
    const auto sa = sch.sqrt_alpha_bar(t, x.dim());
    const auto so = sch.sqrt_one_minus_alpha_bar(t, x.dim());
    return (x_t - sa * x.detach()) / so;
  };
  const auto r = sds_loss(x, sch, perfect, {});
  INFO("perfect-denoiser grad_norm = " << r.grad_norm);
  REQUIRE(r.grad_norm < 1e-3);
}

TEST_CASE("sds: surrogate loss autograd equals the SDS gradient (composes in one backward)",
          "[diffuse]") {
  DdpmSchedule sch;
  torch::manual_seed(2);
  auto x = torch::randn({2, 4, 8, 8}).requires_grad_(true);

  // A biased denoiser (returns zeros) => nonzero, well-defined gradient.
  const NoisePredictor zero_pred = [&](const Tensor& x_t, const Tensor& t) {
    return torch::zeros_like(x_t);
  };
  const auto r = sds_loss(x, sch, zero_pred, {});
  r.loss.backward();

  // d/dx [ 0.5||x - (x-grad).detach()||^2 / B ] = grad / B. Verify the plumbing exactly.
  const int64_t B = x.size(0);
  const auto expected = r.grad / static_cast<double>(B);
  REQUIRE(x.grad().defined());
  REQUIRE((x.grad() - expected).abs().max().item<float>() < 1e-6F);
  REQUIRE(r.grad_norm > 0.0);
}

TEST_CASE("sds: classifier-free guidance combines as eps_u + g(eps_c - eps_u)", "[diffuse]") {
  torch::manual_seed(3);
  const auto eu = torch::randn({2, 4, 8, 8});
  const auto ec = torch::randn({2, 4, 8, 8});
  const float g = 7.5F;
  const auto got = cfg_eps(eu, ec, g);
  REQUIRE((got - (eu + g * (ec - eu))).abs().max().item<float>() < 1e-6F);
}
