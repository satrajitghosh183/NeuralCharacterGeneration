#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/appearance.hpp>

#include <torch/torch.h>

// Build a 4x4 image where pixel (row r, col c) encodes color (c/3, r/3, 0.5), then check that
// sampling at integer pixel centers returns exactly that pixel (align_corners math is correct).
TEST_CASE("sample_vertex_colors reads the right pixel at integer coords", "[recon][appearance]") {
  const int H = 4;
  const int W = 4;
  auto img = torch::zeros({3, H, W});
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      img[0][r][c] = static_cast<float>(c) / (W - 1);
      img[1][r][c] = static_cast<float>(r) / (H - 1);
      img[2][r][c] = 0.5F;
    }
  }

  // verts2d are (x=col, y=row).
  const auto verts2d = torch::tensor({{0.0F, 0.0F},
                                      {3.0F, 0.0F},
                                      {0.0F, 3.0F},
                                      {3.0F, 3.0F},
                                      {1.0F, 2.0F}});
  const auto colors = ncg::recon::sample_vertex_colors(img, verts2d);

  REQUIRE(colors.sizes() == (std::vector<int64_t>{5, 3}));
  const auto expected = torch::tensor({{0.0F, 0.0F, 0.5F},
                                       {1.0F, 0.0F, 0.5F},
                                       {0.0F, 1.0F, 0.5F},
                                       {1.0F, 1.0F, 0.5F},
                                       {1.0F / 3.0F, 2.0F / 3.0F, 0.5F}});
  REQUIRE(torch::allclose(colors, expected, 1e-4, 1e-4));
}

TEST_CASE("sample_vertex_colors clamps out-of-bounds samples to the border", "[recon][appearance]") {
  auto img = torch::ones({3, 4, 4}) * 0.7F;
  const auto verts2d = torch::tensor({{-50.0F, -50.0F}, {999.0F, 999.0F}});  // far outside
  const auto colors = ncg::recon::sample_vertex_colors(img, verts2d);
  REQUIRE(torch::allclose(colors, torch::full({2, 3}, 0.7F), 1e-4, 1e-4));
}

TEST_CASE("vertex_visibility keeps the frontmost vertex per pixel", "[recon][appearance]") {
  // Two verts on the same pixel (0,0) at depths 1 and 2; one lone vert at (3,3).
  const auto verts2d = torch::tensor({{0.0F, 0.0F}, {0.0F, 0.0F}, {3.0F, 3.0F}});
  const auto depth = torch::tensor({1.0F, 2.0F, 5.0F});
  const auto vis = ncg::recon::vertex_visibility(verts2d, depth, /*H=*/4, /*W=*/4, /*tol=*/0.05);
  REQUIRE(torch::allclose(vis, torch::tensor({1.0F, 0.0F, 1.0F}), 1e-5, 1e-5));
}

TEST_CASE("fuse_vertex_colors is a confidence-weighted mean with neutral fill", "[recon][appearance]") {
  const std::vector<torch::Tensor> colors = {
      torch::tensor({{1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {9.0F, 9.0F, 9.0F}}),
      torch::tensor({{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, {9.0F, 9.0F, 9.0F}})};
  const std::vector<torch::Tensor> weights = {torch::tensor({1.0F, 0.0F, 0.0F}),
                                              torch::tensor({1.0F, 1.0F, 0.0F})};
  const auto fused = ncg::recon::fuse_vertex_colors(colors, weights);

  const auto expected = torch::tensor({{0.5F, 0.5F, 0.0F},   // seen in both, weight 1+1
                                       {0.0F, 0.0F, 1.0F},   // seen only in view 1
                                       {0.5F, 0.5F, 0.5F}});  // never seen -> neutral gray
  REQUIRE(torch::allclose(fused.colors, expected, 1e-5, 1e-5));
  REQUIRE(torch::allclose(fused.coverage, torch::tensor({2.0F, 1.0F, 0.0F}), 1e-5, 1e-5));
}
