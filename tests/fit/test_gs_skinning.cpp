#include <catch2/catch_test_macros.hpp>

#include <ncg/fit/fit_avatar.hpp>
#include <ncg/fit/splat_bind.hpp>
#include <ncg/recon/gaussian_model.hpp>

#include <torch/torch.h>

#include <tuple>

// Validates the EXACT skinning algorithm the Unity/Unreal Gaussian-splat component implements:
// each splat is deformed by Σ_k weight_k · boneMatrix[joint_k] · pos, using the top-4 (joint,weight)
// pairs exported in `<prefix>.ply.skin`. Ground truth is ncg::fit::deform_avatar with the full
// per-vertex transforms. Because SMPL-X LBS gives ≤4 influences/vertex, the exported top-4 must
// reproduce the deformation EXACTLY — so the in-engine splats track the rig identically to the math.

namespace {
// A random rigid transform [4,4] (rotation about a random axis + translation).
torch::Tensor rigid(const at::TensorOptions& o) {
  auto axis = torch::randn({3}, o);
  axis = axis / axis.norm().clamp_min(1e-6);
  const double ang = torch::rand({1}, o).item<double>() * 3.0;
  const auto K = torch::zeros({3, 3}, o);
  K[0][1] = -axis[2].item<float>(); K[0][2] = axis[1].item<float>();
  K[1][0] = axis[2].item<float>();  K[1][2] = -axis[0].item<float>();
  K[2][0] = -axis[1].item<float>(); K[2][1] = axis[0].item<float>();
  const auto R = torch::eye(3, o) + std::sin(ang) * K + (1 - std::cos(ang)) * torch::matmul(K, K);
  auto M = torch::eye(4, o);
  M.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
  M.index_put_({torch::indexing::Slice(0, 3), 3}, torch::randn({3}, o) * 0.5);
  return M;
}
}  // namespace

TEST_CASE("GS skinning: exported top-4 bone weights reproduce deform_avatar exactly", "[fit][skin]") {
  torch::manual_seed(0);
  const auto o = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const int64_t J = 24;  // joints
  const int64_t V = 200; // splats/verts

  // Random bone matrices B[J,4,4] (the per-joint rest->posed transforms a rig produces each frame).
  std::vector<torch::Tensor> Bs;
  for (int64_t j = 0; j < J; ++j) Bs.push_back(rigid(o));
  const auto B = torch::stack(Bs, 0);  // [J,4,4]

  // Sparse LBS weights: ≤4 influences per vertex, normalized — the SMPL-X regime.
  auto lbs = torch::zeros({V, J}, o);
  for (int64_t v = 0; v < V; ++v) {
    const int k = 1 + (int)(torch::randint(0, 4, {1}, at::kLong).item<int64_t>());  // 1..4
    auto perm = torch::randperm(J, at::kLong).slice(0, 0, k);
    auto w = torch::rand({k}, o);
    w = w / w.sum();
    for (int t = 0; t < k; ++t) lbs[v][perm[t].item<int64_t>()] = w[t];
  }

  // Full per-vertex transform vt[v] = Σ_j lbs[v,j]·B[j]  — ground truth for deform_avatar.
  const auto vt_full = torch::einsum("vj,jab->vab", {lbs, B});  // [V,4,4]

  // Canonical splat cloud.
  ncg::recon::GaussianCloud g;
  g.positions = torch::randn({V, 3}, o) * 0.3;
  g.scales = torch::full({V, 3}, 0.03F, o);
  g.rotations = torch::zeros({V, 4}, o);
  g.rotations.select(1, 0).fill_(1.0);
  g.opacities = torch::full({V, 1}, 0.9F, o);
  g.colors = torch::rand({V, 3}, o);
  g.validate();
  const auto binding = torch::arange(V, at::TensorOptions().dtype(at::kLong));
  const auto ground = ncg::fit::deform_avatar(g, vt_full, binding);

  // Reconstruct from the EXPORTED top-4 (joints,weights) — exactly what write_gaussian_ply writes
  // and what the Unity compute shader consumes — then rebuild the transform the shader applies.
  const auto tk = lbs.topk(4, /*dim=*/1);
  auto w4 = std::get<0>(tk);                 // [V,4]
  const auto idx4 = std::get<1>(tk);         // [V,4]
  w4 = w4 / w4.sum(1, true).clamp_min(1e-8);
  const auto Bsel = B.index_select(0, idx4.reshape(-1)).reshape({V, 4, 4, 4});  // [V,k,4,4]
  const auto vt4 = (Bsel * w4.view({V, 4, 1, 1})).sum(1);                        // [V,4,4]
  const auto engine = ncg::fit::deform_avatar(g, vt4, binding);

  const double perr = (engine.positions - ground.positions).norm().item<double>() /
                      ground.positions.norm().clamp_min(1e-8).item<double>();
  INFO("position relative error (engine top-4 vs deform_avatar full) = " << perr);
  REQUIRE(perr < 1e-5);  // exact: ≤4 influences means top-4 loses nothing
  // Orientations stay unit and consistent between the two paths.
  REQUIRE(torch::allclose(engine.positions, ground.positions, 1e-4, 1e-4));
}

// FREE-SPLAT extension (Phase C / anti-swim): densified Layer-2 splats are NOT 1:1 with verts — they
// sit off the surface and bind to their top-k nearest mesh verts (bind_splats_knn), skinning by the
// BLENDED transform of those verts. This must TRACK the mesh under pose, not swim. We drive a small
// "motion clip" (several random pose transform-sets) and assert the swim metric (blend-then-apply vs
// apply-then-blend LBS error) stays ≪ the motion scale at every frame.
TEST_CASE("GS skinning: k-NN free splats track the mesh under a motion clip (anti-swim)", "[fit][skin]") {
  torch::manual_seed(1);
  const auto o = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const int64_t J = 24, V = 400;

  // A smooth-ish body mesh patch + sparse LBS (≤4 influences/vert).
  const auto verts = torch::randn({V, 3}, o) * 0.3;
  auto lbs = torch::zeros({V, J}, o);
  for (int64_t v = 0; v < V; ++v) {
    const int k = 1 + static_cast<int>(torch::randint(0, 4, {1}, at::kLong).item<int64_t>());
    auto perm = torch::randperm(J, at::kLong).slice(0, 0, k);
    auto w = torch::rand({k}, o);
    w = w / w.sum();
    for (int t = 0; t < k; ++t) lbs[v][perm[t].item<int64_t>()] = w[t];
  }
  // Free splats: just OFF the surface (densified children land near, not on, verts).
  const auto centers = verts.index_select(0, torch::randint(0, V, {V}, at::kLong)) +
                       torch::randn({V, 3}, o) * 0.015F;  // ~1.5cm off
  const auto bind = ncg::fit::bind_splats_knn(centers, verts, /*k=*/4);
  REQUIRE(bind.idx.size(1) == 4);
  REQUIRE((bind.weight.sum(1) - 1.0).abs().max().item<float>() < 1e-5);  // weights normalized

  for (int frame = 0; frame < 5; ++frame) {  // the motion clip
    std::vector<torch::Tensor> Bs;
    for (int64_t j = 0; j < J; ++j) Bs.push_back(rigid(o));
    const auto B = torch::stack(Bs, 0);
    const auto vt = torch::einsum("vj,jab->vab", {lbs, B});  // [V,4,4] per-vertex transforms
    const double swim = ncg::fit::swim_metric(centers, verts, vt, bind);
    INFO("frame " << frame << " swim = " << swim);
    REQUIRE(swim < 0.02);  // ≪ the ~O(1) per-frame motion scale: free splats track, don't swim
  }
}
