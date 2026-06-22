#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/inverse_render.hpp>

#include <torch/torch.h>

#include <tuple>

using ncg::recon::InverseRenderConfig;
using ncg::recon::shade_sh;
using ncg::recon::sh_basis;
using ncg::recon::solve_inverse_render;

// Ambient-only SH lighting => uniform irradiance over all normals (sanity for the basis/shading).
TEST_CASE("SH ambient lighting shades uniformly", "[recon][inverse]") {
  auto normals = torch::randn({16, 3});
  auto sh = torch::zeros({3, 9});
  sh.select(1, 0).fill_(2.0F);  // only the l=0 (ambient) term, all channels
  const auto shaded = shade_sh(torch::ones({16, 3}), sh, normals);
  REQUIRE(torch::allclose(shaded, torch::full({16, 3}, 2.0F * 0.886227F), 1e-4, 1e-4));
}

// C1 (docs/method.md §4): a single shared albedo observed under SEVERAL different unknown SH
// lights is recoverable (up to a global per-channel scale), and recovery IMPROVES with lighting
// diversity — the empirical proof that casual multi-illumination makes delighting identifiable.
TEST_CASE("multi-illumination recovers albedo; diversity helps (C1)", "[recon][inverse]") {
  torch::manual_seed(0);
  const int V = 300;
  auto normals = torch::randn({V, 3});
  normals = normals / normals.norm(2, -1, true);
  const auto a_true = torch::rand({V, 3}) * 0.7F + 0.2F;  // [0.2, 0.9]
  const auto b = sh_basis(normals);                       // [V,9]

  auto make_obs = [&](int N) {
    auto L = torch::randn({N, 3, 9}) * 0.2F;
    L.select(2, 0) += 1.2F;  // positive ambient so irradiance stays > 0
    const auto E = torch::einsum("nck,vk->nvc", {L, b});  // [N,V,3]
    const auto obs = (a_true.unsqueeze(0) * E).clamp_min(0.0);
    const auto nv = normals.unsqueeze(0).expand({N, V, 3}).contiguous();
    const auto w = torch::ones({N, V});
    return std::make_tuple(obs, nv, w);
  };
  // Compare albedo up to the global per-channel gauge (§4): best per-channel scale, then error.
  auto scaled_err = [&](const torch::Tensor& a_rec) {
    const auto scale = (a_rec * a_true).sum(0) / (a_rec * a_rec).sum(0).clamp_min(1e-8);  // [3]
    return (a_true - a_rec * scale).abs().mean().item<double>();
  };

  InverseRenderConfig cfg;
  cfg.iterations = 80;

  auto [o5, n5, w5] = make_obs(5);
  const double e5 = scaled_err(solve_inverse_render(o5, n5, w5, cfg).albedo);
  auto [o1, n1, w1] = make_obs(1);
  const double e1 = scaled_err(solve_inverse_render(o1, n1, w1, cfg).albedo);

  INFO("scaled albedo error: N=1 " << e1 << "  N=5 " << e5);
  REQUIRE(e5 < 0.05);  // diverse multi-illumination recovers albedo well
  REQUIRE(e5 < e1);    // lighting diversity strictly helps (the C1 effect)
}
