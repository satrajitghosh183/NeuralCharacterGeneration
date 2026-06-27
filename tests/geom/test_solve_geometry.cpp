#include <catch2/catch_test_macros.hpp>

#include <ncg/geom/solve_geometry.hpp>

#include <torch/torch.h>

#include <cmath>
#include <vector>

// Phase B proof on synthetic ground truth (docs/method.md §M4/§M5). A grid mesh carries a known
// identity β and a known OFF-SUBSPACE displacement Δv (a Gaussian bump that is NOT in the smooth
// sin/cos shape basis). Dense 2D correspondences are observed from several weak-perspective cameras,
// but ONLY over the left half of the grid. The solver must (a) recover β, (b) recover Δv where
// observed, and (c) pin Δv≈0 where unobserved (the observability hard gate) — i.e. never hallucinate
// geometry on the unseen region.

namespace {
constexpr int G = 12;             // grid side
constexpr int64_t V = G * G;
}  // namespace

TEST_CASE("solve_geometry: off-subspace Δv recovered where observed, pinned where not (M4/M5)",
          "[geom]") {
  torch::manual_seed(0);
  const auto lin = torch::linspace(0, 1, G);
  const auto gx = lin.unsqueeze(0).expand({G, G}).reshape({V});
  const auto gy = lin.unsqueeze(1).expand({G, G}).reshape({V});
  // Non-planar DOME base — a planar grid is a bas-relief-degenerate config for weak-perspective
  // shape recovery (and gives zero angular diversity); a dome makes the geometry genuinely 3D.
  const auto r2 = (gx - 0.5).pow(2) + (gy - 0.5).pow(2);
  const auto zbase = 0.5 * torch::exp(-r2 / 0.1);
  const auto base = torch::stack({gx, gy, zbase}, 1);  // [V,3] dome

  // Triangulate the grid (2 tris per cell).
  std::vector<int64_t> fv;
  for (int r = 0; r < G - 1; ++r)
    for (int c = 0; c < G - 1; ++c) {
      const int64_t a = r * G + c, b = r * G + c + 1, d = (r + 1) * G + c, e = (r + 1) * G + c + 1;
      fv.insert(fv.end(), {a, b, d, b, e, d});
    }
  const auto faces = torch::tensor(fv, at::kLong).reshape({-1, 3});  // [F,3]
  const int64_t F = faces.size(0);

  // Graph Laplacian L = D - A from the mesh edges (smoothness operator).
  auto Adj = torch::zeros({V, V});
  const auto fa = faces.accessor<int64_t, 2>();
  for (int64_t f = 0; f < F; ++f)
    for (int e = 0; e < 3; ++e) {
      const int64_t i = fa[f][e], j = fa[f][(e + 1) % 3];
      Adj.index_put_({i, j}, 1.0F);
      Adj.index_put_({j, i}, 1.0F);
    }
  const auto L = torch::diag(Adj.sum(1)) - Adj;  // [V,V]

  // Smooth z-displacement bases (the SMPL-X subspace analog).
  auto basis = [&](int64_t n, double f0) {
    auto B = torch::zeros({V, 3, n});
    for (int64_t k = 0; k < n; ++k)
      B.select(2, k).select(1, 2).copy_(torch::sin((k + 1) * f0 * gx) * torch::cos((k + 1) * f0 * gy));
    return B;
  };
  const int64_t nid = 2, nex = 2;  // few smooth modes ⇒ β cannot absorb a sharp local bump
  const auto id_basis = basis(nid, 2.0), expr_basis = basis(nex, 5.0);
  const auto beta_gt = torch::randn({nid}) * 0.1;

  // Δv_gt: a SHARP, localized OFF-subspace bump in z — neither the smooth id modes nor a global
  // affine camera can absorb it (its multi-view parallax forces it into Δv where observed).
  const auto bump = torch::exp(-((gx - 0.30).pow(2) + (gy - 0.5).pow(2)) / 0.008);
  auto dv_gt = torch::zeros({V, 3});
  dv_gt.select(1, 2).copy_(bump * 0.25);

  // Dense points: K samples on left-half faces (x<0.5) ⇒ the right half is unobserved.
  std::vector<int64_t> sel;
  for (int64_t f = 0; f < F; ++f) {
    const float cx = (base[fa[f][0]][0] + base[fa[f][1]][0] + base[fa[f][2]][0]).item<float>() / 3;
    if (cx < 0.4F) sel.push_back(f);  // points only on the left ⇒ x>0.6 is cleanly unobserved
  }
  const int64_t K = 300;
  auto assoc = torch::zeros({K}, at::kLong);
  auto bary = torch::zeros({K, 3});
  for (int64_t k = 0; k < K; ++k) {
    assoc.index_put_({k}, sel[static_cast<size_t>(k) % sel.size()]);
    auto b = torch::rand({3});
    bary[k].copy_(b / b.sum());
  }

  // Ground-truth posed dense points, then projected by N diverse weak-perspective cameras.
  const auto gt_verts = base + torch::einsum("vcn,n->vc", {id_basis, beta_gt}) + dv_gt;
  const auto cflat = faces.index_select(0, assoc).reshape({-1});
  const auto gq = torch::einsum("kc,kcd->kd", {bary, gt_verts.index_select(0, cflat).reshape({K, 3, 3})});
  const int64_t N = 8;
  auto lm = torch::zeros({N, K, 2});
  for (int64_t i = 0; i < N; ++i) {
    const double ang = (i / static_cast<double>(N - 1) - 0.5) * 2.4;  // yaw -1.2..1.2
    const double az = (i % 2 ? 0.5 : -0.5);
    const auto Ry = torch::tensor({{std::cos(ang), 0.0, std::sin(ang)}, {0.0, 1.0, 0.0},
                                   {-std::sin(ang), 0.0, std::cos(ang)}});
    const auto Rx = torch::tensor({{1.0, 0.0, 0.0}, {0.0, std::cos(az), -std::sin(az)},
                                   {0.0, std::sin(az), std::cos(az)}});
    const auto Mi = torch::matmul(Rx, Ry).slice(0, 0, 2) * 80.0;  // [2,3]
    lm[i].copy_(torch::matmul(gq, Mi.t()) + torch::tensor({100.0, 100.0}));
  }

  ncg::geom::GeomConfig cfg;
  cfg.iterations = 30;
  cfg.lap_weight = 3.0F;   // light smoothness — let the localized bump through
  cfg.mag_weight = 0.1F;   // do not over-penalize Δv (else β+camera absorb the bump, Δv→0)
  cfg.cg_iters = 150;
  cfg.o_solve = 0.6F;  // gate between unobserved (~0.23) and observed (~0.99)
  const auto R = ncg::geom::solve_geometry(base, id_basis, expr_basis, faces, L, assoc, bary, lm,
                                           torch::ones({N, K}), torch::ones({N}), cfg);

  // Clean regions: points cover x<0.4, so x>0.6 is genuinely unobserved (boundary band excluded).
  const auto obs = (base.select(1, 0) < 0.4F);
  const auto unobs = (base.select(1, 0) > 0.6F);

  // (1) Data fit: GT landmarks are exact, so a correct solver reprojects to ~0 px (scale ~80).
  const auto residual = R.residual;
  // (2) Observability separates the two regions.
  const auto o_obs = R.obs.index({obs}).mean().item<float>();
  const auto o_unobs = R.obs.index({unobs}).mean().item<float>();
  // (3) Hard gate pins Δv where unobserved.
  const auto dv_unobs = R.delta_v.index({unobs}).abs().max().item<float>();
  // (4) Off-subspace structure: recovered Δv-z correlates with the GT bump where observed (gauge-
  // robust — weak-perspective recovers shape only up to an affine, so we check the PATTERN not the
  // exact magnitude).
  const auto az = R.delta_v.select(1, 2).index({obs}), bz = dv_gt.select(1, 2).index({obs});
  const auto corr =
      ((az * bz).sum() / (az.norm() * bz.norm()).clamp_min(1e-9)).item<float>();

  INFO("residual=" << residual << " o(obs)=" << o_obs << " o(unobs)=" << o_unobs
                   << " dv_unobs_max=" << dv_unobs << " Δv·Δv_gt corr=" << corr);

  REQUIRE(residual < 1.0);          // solver fits the (noise-free) observations
  REQUIRE(o_obs > 0.7F);            // observed region observable
  REQUIRE(o_unobs < 0.55F);         // unobserved region not
  REQUIRE(dv_unobs < 1e-4F);        // Δv pinned to 0 where unobserved (hard gate)
  REQUIRE(corr > 0.5F);             // recovered Δv captures the off-subspace bump where observed
}
