#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/uv_texture.hpp>

#include <torch/torch.h>

#include <cmath>

// UV rasterizer + bake: a single right triangle covering the lower-left half of the texture. Texels
// inside it must carry valid barycentric weights (summing to 1) and the correct geometry face;
// texels outside stay empty. Baking the three vertex one-hot colors must reproduce barycentric
// coordinates as RGB. Deterministic, no assets — the core of the per-texel albedo lift.

TEST_CASE("uv_rasterize: barycentric coverage of a UV triangle", "[recon][uv]") {
  const auto o = at::TensorOptions().dtype(at::kFloat);
  const auto uv = torch::tensor({{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}}, o);  // right triangle
  const auto uvf = torch::tensor({{0, 1, 2}}, at::TensorOptions().dtype(at::kLong));
  const int res = 16;
  const auto ras = ncg::recon::uv_rasterize(uv, uvf, res);

  REQUIRE(ras.res == res);
  REQUIRE(ras.face.sizes() == (std::vector<int64_t>{res * res}));

  // A texel near the (0,0) corner is inside; weights sum to 1 and face index is 0.
  const int64_t t_in = 0 * res + 0;  // row 0, col 0 -> center (0.5/16, 0.5/16)
  REQUIRE(ras.face[t_in].item<int64_t>() == 0);
  REQUIRE(std::abs(ras.bary[t_in].sum().item<double>() - 1.0) < 1e-4);
  // A texel deep in the upper-right is outside the lower-left triangle -> empty.
  const int64_t t_out = (res - 1) * res + (res - 1);
  REQUIRE(ras.face[t_out].item<int64_t>() == -1);

  // At least a third of the grid is covered (triangle is ~half).
  const double covered = (ras.face >= 0).to(at::kFloat).mean().item<double>();
  REQUIRE(covered > 0.3);
  REQUIRE(covered < 0.7);

  // Bake one-hot vertex colors -> texel color == its barycentric weights.
  const auto faces = torch::tensor({{0, 1, 2}}, at::TensorOptions().dtype(at::kLong));
  const auto vcol = torch::eye(3, o);  // vert k -> color e_k
  torch::Tensor mask;
  const auto tex = ncg::recon::bake_to_uv(ras, vcol, faces, mask);  // [res,res,3]
  REQUIRE(tex.sizes() == (std::vector<int64_t>{res, res, 3}));
  const auto tex_flat = tex.view({res * res, 3});
  // On a covered texel, the baked color equals its barycentric weights.
  REQUIRE(torch::allclose(tex_flat[t_in], ras.bary[t_in], 1e-4, 1e-4));
  REQUIRE(mask.view({res * res})[t_in].item<double>() > 0.5);
  REQUIRE(mask.view({res * res})[t_out].item<double>() < 0.5);
}
