#include <catch2/catch_test_macros.hpp>

#include <ncg/rig/rig.hpp>

#include <torch/torch.h>

#include <filesystem>

TEST_CASE("make_rigged validates and packages a rig", "[rig]") {
  const int64_t V = 12;
  const int64_t J = 3;
  const int64_t F = 4;
  auto verts = torch::randn({V, 3});
  auto faces = torch::randint(0, V, {F, 3}, torch::kLong);
  auto joints = torch::randn({J, 3});
  auto parents = torch::zeros({J}, torch::kLong);
  auto weights = torch::softmax(torch::randn({V, J}), 1);

  const auto m = ncg::rig::make_rigged(verts, faces, joints, parents, weights);
  REQUIRE(m.vertices.size(0) == V);
  REQUIRE(m.skin_weights.sizes() == (std::vector<int64_t>{V, J}));

  REQUIRE_THROWS(ncg::rig::make_rigged(verts, faces, joints, parents, torch::randn({V, J + 1})));
}

TEST_CASE("export_rigged writes OBJ + rig JSON", "[rig]") {
  const int64_t V = 6;
  const int64_t J = 2;
  auto m = ncg::rig::make_rigged(torch::randn({V, 3}), torch::randint(0, V, {2, 3}, torch::kLong),
                                 torch::randn({J, 3}), torch::zeros({J}, torch::kLong),
                                 torch::softmax(torch::randn({V, J}), 1));

  const auto base = (std::filesystem::temp_directory_path() / "ncg_rig").string();
  ncg::rig::export_rigged(m, base);
  REQUIRE(std::filesystem::file_size(base + ".obj") > 0);
  REQUIRE(std::filesystem::file_size(base + ".rig.json") > 0);
  std::filesystem::remove(base + ".obj");
  std::filesystem::remove(base + ".rig.json");
}

TEST_CASE("autorig (UniRig) is gated", "[rig]") {
  REQUIRE_THROWS(ncg::rig::autorig(torch::randn({4, 3}), torch::zeros({2, 3}, torch::kLong)));
}
