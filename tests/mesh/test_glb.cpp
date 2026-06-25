#include <catch2/catch_test_macros.hpp>

#include <ncg/mesh/extract.hpp>

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
std::vector<char> read_file(const std::string& p) {
  std::ifstream is(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
}
uint32_t u32_at(const std::vector<char>& b, size_t o) {
  return static_cast<uint8_t>(b[o]) | (static_cast<uint8_t>(b[o + 1]) << 8) |
         (static_cast<uint8_t>(b[o + 2]) << 16) | (static_cast<uint8_t>(b[o + 3]) << 24);
}
// A unit tetrahedron mesh.
std::pair<torch::Tensor, torch::Tensor> tetra() {
  auto v = torch::tensor({{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F},
                          {0.0F, 0.0F, 1.0F}});
  auto f = torch::tensor({{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}, torch::kLong);
  return {v, f};
}
}  // namespace

TEST_CASE("write_glb produces a valid binary glTF", "[mesh][glb]") {
  auto [v, f] = tetra();
  const auto colors = torch::rand({4, 3});
  const auto path = (std::filesystem::temp_directory_path() / "ncg_test.glb").string();
  ncg::mesh::write_glb(v, f, /*normals=*/{}, colors, path);

  const auto b = read_file(path);
  REQUIRE(b.size() > 12);
  REQUIRE(std::string(b.data(), 4) == "glTF");
  REQUIRE(u32_at(b, 4) == 2);             // version
  REQUIRE(u32_at(b, 8) == b.size());      // total length matches file
  const uint32_t json_len = u32_at(b, 12);
  REQUIRE(std::string(b.data() + 16, 4) == "JSON");
  const std::string json(b.data() + 20, json_len);
  REQUIRE(json.find("\"POSITION\":0") != std::string::npos);
  REQUIRE(json.find("COLOR_0") != std::string::npos);
}

TEST_CASE("write_glb_skinned embeds a skin + joints", "[mesh][glb]") {
  auto [v, f] = tetra();
  const auto joints = torch::tensor({{0.0F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F}});  // 2 joints
  const auto parents = torch::tensor({0, 0}, torch::kLong);            // joint1 child of root
  const auto skin = torch::rand({4, 2});                               // [V,J] weights
  const auto path = (std::filesystem::temp_directory_path() / "ncg_test_skinned.glb").string();
  ncg::mesh::write_glb_skinned(v, f, /*normals=*/{}, /*colors=*/{}, joints, parents, skin, path);

  const auto b = read_file(path);
  REQUIRE(std::string(b.data(), 4) == "glTF");
  REQUIRE(u32_at(b, 8) == b.size());
  const std::string json(b.data() + 20, u32_at(b, 12));
  REQUIRE(json.find("\"skins\"") != std::string::npos);
  REQUIRE(json.find("JOINTS_0") != std::string::npos);
  REQUIRE(json.find("WEIGHTS_0") != std::string::npos);
  REQUIRE(json.find("inverseBindMatrices") != std::string::npos);
}

TEST_CASE("write_glb_animated embeds a skeletal animation", "[mesh][glb]") {
  auto [v, f] = tetra();
  const auto joints = torch::tensor({{0.0F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F}});
  const auto parents = torch::tensor({0, 0}, torch::kLong);
  const auto skin = torch::rand({4, 2});
  const int T = 5;
  const int J = 2;
  const auto quats = torch::zeros({T, J, 4});
  quats.select(2, 3).fill_(1.0);  // identity quaternions (x,y,z,w)=(0,0,0,1)
  const auto times = torch::arange(T, torch::kFloat) / 30.0F;
  const auto path = (std::filesystem::temp_directory_path() / "ncg_test_anim.glb").string();
  ncg::mesh::write_glb_animated(v, f, torch::rand({4, 3}), torch::rand({4, 3}), joints, parents,
                                skin, quats, times, path);

  const auto b = read_file(path);
  REQUIRE(std::string(b.data(), 4) == "glTF");
  REQUIRE(u32_at(b, 8) == b.size());
  const std::string json(b.data() + 20, u32_at(b, 12));
  REQUIRE(json.find("\"animations\"") != std::string::npos);
  REQUIRE(json.find("\"samplers\"") != std::string::npos);
  REQUIRE(json.find("\"path\":\"rotation\"") != std::string::npos);
  REQUIRE(json.find("\"skins\"") != std::string::npos);
}
