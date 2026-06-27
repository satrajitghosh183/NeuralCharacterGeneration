#include <catch2/catch_test_macros.hpp>

#include <ncg/diffuse/completion.hpp>
#include <ncg/diffuse/scheduler.hpp>
#include <ncg/diffuse/sds.hpp>

#include <torch/torch.h>

#include <cmath>

// Phase E novelty (docs/method.md §M10/§M11): observability-gated render-consistent completion.
// SDS is applied ONLY on unobserved surface and is EXACTLY ZERO on observed surface (identity
// protection). These are exact algebraic properties of the gate g(o) — proven on CPU, no weights.

using namespace ncg::diffuse;

TEST_CASE("completion: identity-protection — gate is BIT-EXACT zero on observed surface", "[diffuse]") {
  CompletionConfig cfg;  // obs_lo=0.15, obs_hi=0.35
  // Observed pixels: o >= obs_hi.
  const auto obs = torch::tensor({0.35F, 0.5F, 0.8F, 1.0F});
  const auto g = completion_gate(obs, cfg);
  // Theorem M11: g == 0 EXACTLY (not merely < eps) so no diffusion signal reaches observed identity.
  REQUIRE(g.abs().max().item<float>() == 0.0F);

  // And the applied completion gradient is therefore identically zero there.
  const auto grad = torch::randn({1, 4, 2, 2});
  const auto obs_map = torch::full({1, 1, 2, 2}, 0.6F);  // all observed
  const auto gated = apply_completion_gate(grad, obs_map, cfg);
  REQUIRE(gated.abs().max().item<float>() == 0.0F);
}

TEST_CASE("completion: full SDS on fully-unobserved surface", "[diffuse]") {
  CompletionConfig cfg;
  const auto obs = torch::tensor({0.0F, 0.05F, 0.15F});  // o <= obs_lo
  const auto g = completion_gate(obs, cfg);
  REQUIRE((g - 1.0F).abs().max().item<float>() == 0.0F);  // gate exactly 1 → SDS passes unchanged

  const auto grad = torch::randn({1, 4, 2, 2});
  const auto obs_map = torch::zeros({1, 1, 2, 2});  // fully unobserved
  const auto gated = apply_completion_gate(grad, obs_map, cfg);
  REQUIRE((gated - grad).abs().max().item<float>() < 1e-6F);
}

TEST_CASE("completion: gate is monotone non-increasing and C¹ (smoothstep)", "[diffuse]") {
  CompletionConfig cfg;
  const auto o = torch::linspace(0.0, 0.5, 1001);
  const auto g = completion_gate(o, cfg);

  // Monotone non-increasing in observability (more observed ⇒ less completion).
  const auto dg = g.slice(0, 1) - g.slice(0, 0, 1000);
  REQUIRE(dg.max().item<float>() <= 1e-6F);

  // C¹: derivative continuous and ≈0 at both seams (smoothstep), no step-function spike.
  const double h = 0.5 / 1000.0;
  const auto gp = dg / h;                          // ~g'(o)
  const auto gpp = (gp.slice(0, 1) - gp.slice(0, 0, 999)).abs();  // curvature proxy
  // A hard step would put a huge spike in gpp at the seam; smoothstep keeps it bounded.
  REQUIRE(gpp.max().item<float>() < 50.0F);
  // Derivative vanishes at the seams o=obs_lo (idx~300) and o=obs_hi (idx~700).
  REQUIRE(std::abs(gp[300].item<float>()) < 0.2F);
  REQUIRE(std::abs(gp[700].item<float>()) < 0.2F);
}

TEST_CASE("completion: end-to-end loop COMPLETES unobserved while freezing observed identity",
          "[diffuse]") {
  // The full render-consistent completion dynamics on CPU, no SD weights. A denoiser tuned to a
  // TARGET image x_tgt makes the SDS gradient reduce exactly to w(t)*(sa/so)*(x - x_tgt) (the eps
  // cancels), so SDS pulls x toward the prior's manifold (= x_tgt here). Gate half the pixels as
  // OBSERVED (o=1 → gate 0) and half UNOBSERVED (o=0 → gate 1), then run gated SDS descent.
  DdpmSchedule sch;
  torch::manual_seed(7);
  CompletionConfig ccfg;

  const auto x_tgt = torch::rand({1, 4, 8, 8});         // the "completed" manifold sample
  auto x = torch::rand({1, 4, 8, 8});                   // current render (starts far from x_tgt)
  const auto x0 = x.clone();

  // Observability: left half observed (1.0), right half unobserved (0.0).
  auto obs = torch::zeros({1, 1, 8, 8});
  obs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(), torch::indexing::Slice(0, 4)}, 1.0);
  const auto obs_bchw = obs.expand({1, 4, 8, 8});
  const auto observed = obs_bchw > 0.5;
  const auto unobserved = obs_bchw < 0.5;

  const NoisePredictor target_denoiser = [&](const at::Tensor& x_t, const at::Tensor& t) {
    const auto sa = sch.sqrt_alpha_bar(t, x_t.dim());
    const auto so = sch.sqrt_one_minus_alpha_bar(t, x_t.dim());
    return (x_t - sa * x_tgt) / so;  // perfect denoiser FOR x_tgt
  };

  SdsConfig scfg;
  scfg.clip_grad = false;  // clean dynamics for the test
  const float lr = 6.0F;
  for (int i = 0; i < 400; ++i) {
    const auto r = sds_loss(x, sch, target_denoiser, scfg);
    const auto gated = apply_completion_gate(r.grad, obs, ccfg);  // zero on observed half
    x = x - lr * gated;
  }

  // Identity protection through the WHOLE loop: observed pixels are bit-exactly unchanged.
  REQUIRE((x.index({observed}) - x0.index({observed})).abs().max().item<float>() == 0.0F);

  // Completion: unobserved pixels converged toward the prior manifold x_tgt.
  const auto err0 = (x0.index({unobserved}) - x_tgt.index({unobserved})).norm().item<float>();
  const auto err1 = (x.index({unobserved}) - x_tgt.index({unobserved})).norm().item<float>();
  INFO("unobserved err: " << err0 << " -> " << err1);
  REQUIRE(err1 < 0.2F * err0);
}

TEST_CASE("completion: provenance mask labels synthesized vs photo-observed", "[diffuse]") {
  CompletionConfig cfg;
  const auto obs = torch::tensor({{0.0F, 0.2F}, {0.34F, 0.9F}});  // last two: 0.34 synth, 0.9 observed
  const auto m = provenance_mask(obs, cfg);
  // synthesized (g>0) where o < obs_hi → 1; observed → 0.
  REQUIRE(m[0][0].item<float>() == 1.0F);  // o=0.0
  REQUIRE(m[0][1].item<float>() == 1.0F);  // o=0.2
  REQUIRE(m[1][0].item<float>() == 1.0F);  // o=0.34 (< 0.35)
  REQUIRE(m[1][1].item<float>() == 0.0F);  // o=0.9 (observed)
}
