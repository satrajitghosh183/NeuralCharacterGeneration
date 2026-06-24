#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/inverse_render.hpp>

#include <torch/torch.h>

#include <tuple>

using ncg::recon::InverseRenderConfig;
using ncg::recon::shade_sh;
using ncg::recon::sh_basis;
using ncg::recon::sh_directional_light;
using ncg::recon::solve_inverse_render;
using ncg::recon::transport_normals;

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

// C2 (docs/method.md §3): with a fraction of observations corrupted (clothing swap / occlusion /
// junk uploads), the robust consistency E-step must recover albedo far better than the
// non-robust solve — and must actually identify (down-weight) the corrupted observations.
TEST_CASE("robust consistency rejects corrupted observations (C2)", "[recon][inverse]") {
  torch::manual_seed(1);
  const int V = 300;
  const int N = 8;
  auto normals = torch::randn({V, 3});
  normals = normals / normals.norm(2, -1, true);
  const auto a_true = torch::rand({V, 3}) * 0.7F + 0.2F;
  const auto b = sh_basis(normals);

  auto L = torch::randn({N, 3, 9}) * 0.2F;
  L.select(2, 0) += 1.2F;
  const auto E = torch::einsum("nck,vk->nvc", {L, b});
  const auto clean = (a_true.unsqueeze(0) * E).clamp_min(0.0);          // [N,V,3]
  const auto nv = normals.unsqueeze(0).expand({N, V, 3}).contiguous();

  // Corrupt 35% of observations with junk (the inconsistent casual-photo case).
  const auto corrupt = (torch::rand({N, V}) < 0.35F);                   // [N,V]
  const auto junk = torch::rand({N, V, 3});
  const auto obs = torch::where(corrupt.unsqueeze(-1), junk, clean);
  const auto w = torch::ones({N, V});

  auto scaled_err = [&](const torch::Tensor& a_rec) {
    const auto scale = (a_rec * a_true).sum(0) / (a_rec * a_rec).sum(0).clamp_min(1e-8);
    return (a_true - a_rec * scale).abs().mean().item<double>();
  };

  InverseRenderConfig robust_cfg;
  robust_cfg.iterations = 80;
  robust_cfg.robust = true;
  InverseRenderConfig plain_cfg = robust_cfg;
  plain_cfg.robust = false;

  const auto rob = solve_inverse_render(obs, nv, w, robust_cfg);
  const double e_rob = scaled_err(rob.albedo);
  const double e_plain = scaled_err(solve_inverse_render(obs, nv, w, plain_cfg).albedo);

  // The inferred consistency should be low on corrupted obs, high on clean ones.
  const auto cons = rob.consistency;  // [N,V]
  const double mean_corrupt = cons.masked_select(corrupt).mean().item<double>();
  const double mean_clean = cons.masked_select(corrupt.logical_not()).mean().item<double>();

  INFO("err robust=" << e_rob << " plain=" << e_plain << " | consistency corrupt="
                     << mean_corrupt << " clean=" << mean_clean);
  REQUIRE(e_rob < e_plain);             // robustness helps under corruption
  REQUIRE(e_rob < 0.06);                // and still recovers albedo well
  REQUIRE(mean_corrupt < mean_clean);   // it actually identifies the bad observations
}

// Relighting payoff: recover albedo from casually-lit photos, then render it under a NOVEL,
// never-observed light and check it matches the ground-truth-albedo render under that light.
// This is the relightable claim, validated numerically (no light stage, no Python).
TEST_CASE("recovered albedo relights correctly under a novel light", "[recon][inverse]") {
  torch::manual_seed(2);
  const int V = 300;
  const int N = 6;
  auto normals = torch::randn({V, 3});
  normals = normals / normals.norm(2, -1, true);
  const auto a_true = torch::rand({V, 3}) * 0.7F + 0.2F;
  const auto b = sh_basis(normals);

  auto L = torch::randn({N, 3, 9}) * 0.25F;
  L.select(2, 0) += 1.2F;
  const auto obs = (a_true.unsqueeze(0) * torch::einsum("nck,vk->nvc", {L, b})).clamp_min(0.0);
  const auto nv = normals.unsqueeze(0).expand({N, V, 3}).contiguous();

  InverseRenderConfig cfg;
  cfg.iterations = 80;
  auto a_rec = solve_inverse_render(obs, nv, torch::ones({N, V}), cfg).albedo;
  // Undo the global per-channel gauge before relighting.
  a_rec = a_rec * ((a_rec * a_true).sum(0) / (a_rec * a_rec).sum(0).clamp_min(1e-8));

  // A novel directional light that appeared in none of the input photos.
  const auto Lnew = sh_directional_light(torch::tensor({0.4F, -0.7F, 0.6F}),
                                         torch::tensor({1.0F, 0.95F, 0.9F}), /*ambient=*/0.25F);
  const auto relit_true = shade_sh(a_true, Lnew, normals);
  const auto relit_rec = shade_sh(a_rec, Lnew, normals);

  const double rel_err = (relit_true - relit_rec).norm().item<double>() /
                         relit_true.norm().clamp_min(1e-8).item<double>();
  INFO("relight relative error under novel light = " << rel_err);
  // ~7% relighting error to an unseen light from casual input, no light stage — it tracks the
  // albedo-recovery error and tightens with more views / iterations.
  REQUIRE(rel_err < 0.08);
}

// C3 (docs/method.md §8): normals transported by the (blended) bone rotation make relighting and
// animation commute. (1) a single full-weight bone rotates normals exactly by R; (2) shading the
// posed normals under a world light equals shading the canonical normals under the light pulled
// back by R — so posing then relighting == relighting then posing.
TEST_CASE("animate and relight commute via normal transport (C3)", "[recon][inverse]") {
  torch::manual_seed(3);
  const int V = 200;
  auto n = torch::randn({V, 3});
  n = n / n.norm(2, -1, true);
  const auto albedo = torch::rand({V, 3}) * 0.6F + 0.3F;

  const float t = 0.7F;  // 40deg about Y
  const auto R = torch::tensor({{std::cos(t), 0.0F, std::sin(t)},
                                {0.0F, 1.0F, 0.0F},
                                {-std::sin(t), 0.0F, std::cos(t)}});
  const auto W = torch::ones({V, 1});      // single bone, full weight
  const auto bones = R.unsqueeze(0);       // [1,3,3]

  const auto n_posed = transport_normals(n, W, bones);
  REQUIRE(torch::allclose(n_posed, torch::matmul(n, R.t()), 1e-4, 1e-4));  // == R n

  const auto d = torch::tensor({0.3F, -0.6F, 0.7F});
  const auto white = torch::ones({3});
  // posed body, world light d  vs  canonical body, light pulled back by R (== R^T d)
  const auto colorsA = shade_sh(albedo, sh_directional_light(d, white, 0.2F), n_posed);
  const auto colorsB = shade_sh(albedo, sh_directional_light(torch::matmul(R.t(), d), white, 0.2F), n);
  const double err = (colorsA - colorsB).abs().max().item<double>();
  INFO("max |animate∘relight − relight∘animate| = " << err);
  REQUIRE(torch::allclose(colorsA, colorsB, 1e-3, 1e-3));
}
