#include <catch2/catch_test_macros.hpp>

#include <ncg/fit/splat_bind.hpp>

#include <torch/torch.h>

// Phase C anti-swim proof (docs/method.md §M9). Free render-only splats bound to the Layer-1 mesh
// by a k-NN SOFT binding must track the surface under pose — NOT swim. This EXTENDS
// test_gs_skinning from 1:1 vertex binding (<1e-5) to k-NN free binding: under a smoothly-varying
// per-vertex deformation, a splat near the surface, skinned by the blended transform of its k
// nearest verts, stays attached (swim ≪ motion scale).

namespace {
constexpr int G = 10;
constexpr int64_t V = G * G;

// Smoothly-varying per-vertex transform: rotate about Y by angle a·x, translate in z by b·y. So
// nearby verts have similar-but-different transforms (the regime where LBS blending must hold).
torch::Tensor make_transforms(const torch::Tensor& gx, const torch::Tensor& gy) {
  const auto ang = 1.2 * gx;
  const auto c = torch::cos(ang), s = torch::sin(ang), z = torch::zeros({V}), o = torch::ones({V});
  const auto R = torch::stack({c, z, s, z, o, z, -s, z, c}, 1).reshape({V, 3, 3});
  auto vt = torch::eye(4).unsqueeze(0).repeat({V, 1, 1});  // [V,4,4]
  vt.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
  vt.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 3},
                torch::stack({z, z, 0.3 * gy}, 1));
  return vt;
}
}  // namespace

TEST_CASE("swim: k-NN free-splat binding tracks the surface under pose (M9)", "[cuda;fit]") {
  torch::manual_seed(0);
  const auto lin = torch::linspace(0, 1, G);
  const auto gx = lin.unsqueeze(0).expand({G, G}).reshape({V});
  const auto gy = lin.unsqueeze(1).expand({G, G}).reshape({V});
  const auto verts = torch::stack({gx, gy, torch::zeros({V})}, 1);  // [V,3]
  const auto vt = make_transforms(gx, gy);

  // (1) 1:1 sanity — a splat exactly on a vertex, k=1, must not swim at all (matches gs_skinning).
  const auto b1 = ncg::fit::bind_splats_knn(verts, verts, /*k=*/1);
  const auto swim_11 = ncg::fit::swim_metric(verts, verts, vt, b1);
  INFO("swim (1:1, k=1) = " << swim_11);
  REQUIRE(swim_11 < 1e-5);

  // (2) k-NN free splats near the surface — bound to 4 verts, blended transform; swim ≪ motion.
  const auto centers = verts + torch::randn({V, 3}) * 0.02;  // free splats off the surface
  const auto b4 = ncg::fit::bind_splats_knn(centers, verts, /*k=*/4);
  REQUIRE(b4.idx.size(1) == 4);
  const auto wsum = b4.weight.sum(1);
  REQUIRE((wsum - 1.0).abs().max().item<float>() < 1e-5);  // weights normalized
  const auto swim_knn = ncg::fit::swim_metric(centers, verts, vt, b4);
  // motion scale ~ O(1) (rotation up to 1.2 rad over unit grid); require swim under ~1% of that.
  INFO("swim (k-NN, k=4) = " << swim_knn);
  REQUIRE(swim_knn < 0.01);
}
