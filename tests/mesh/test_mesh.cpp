#include <catch2/catch_test_macros.hpp>

#include <ncg/mesh/extract.hpp>
#include <ncg/recon/gaussian_model.hpp>

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <vector>

// Marching cubes on an analytic field (a centered blob) must yield a non-empty closed-ish
// surface; extract_mesh on a Gaussian cloud must produce geometry; writers must emit files.

TEST_CASE("marching cubes extracts a surface from a blob field", "[mesh]") {
  const int g = 24;
  std::vector<float> field(static_cast<size_t>(g) * g * g);
  const float c = (g - 1) / 2.0F;
  for (int z = 0; z < g; ++z) {
    for (int y = 0; y < g; ++y) {
      for (int x = 0; x < g; ++x) {
        const float dx = x - c;
        const float dy = y - c;
        const float dz = z - c;
        field[(static_cast<size_t>(z) * g + y) * g + x] =
            std::exp(-0.02F * (dx * dx + dy * dy + dz * dz));
      }
    }
  }
  const auto mesh = ncg::mesh::marching_cubes(field, g, g, g, /*iso=*/0.5F, {0, 0, 0}, {1, 1, 1});
  REQUIRE(mesh.num_verts() > 0);
  REQUIRE(mesh.num_faces() > 0);
  REQUIRE(mesh.faces.max().item<int64_t>() < mesh.num_verts());

  const auto normals = ncg::mesh::compute_vertex_normals(mesh);
  REQUIRE(normals.sizes() == mesh.vertices.sizes());
  const auto norm_len = normals.norm(2, 1);  // unit length
  REQUIRE(torch::allclose(norm_len, torch::ones_like(norm_len), 1e-4, 1e-4));
}

TEST_CASE("extract_mesh + writers produce files", "[mesh]") {
  const auto opts = torch::TensorOptions().dtype(torch::kFloat);
  // A small spherical shell of Gaussians.
  const int n = 200;
  auto dirs = torch::randn({n, 3}, opts);
  dirs = dirs / dirs.norm(2, 1, true);
  ncg::recon::GaussianCloud cloud;
  cloud.positions = dirs * 0.5;
  cloud.scales = torch::full({n, 3}, 0.08F, opts);
  cloud.rotations = torch::zeros({n, 4}, opts);
  cloud.rotations.select(1, 0).fill_(1.0);
  cloud.opacities = torch::ones({n, 1}, opts);
  cloud.colors = torch::full({n, 3}, 0.6F, opts);
  cloud.validate();

  const auto mesh = ncg::mesh::extract_mesh(cloud, /*res=*/32);
  REQUIRE(mesh.num_verts() > 0);
  REQUIRE(mesh.num_faces() > 0);

  const auto obj = (std::filesystem::temp_directory_path() / "ncg_mesh.obj").string();
  const auto ply = (std::filesystem::temp_directory_path() / "ncg_mesh.ply").string();
  ncg::mesh::write_obj(mesh, obj);
  ncg::mesh::write_ply(mesh, ply);
  REQUIRE(std::filesystem::file_size(obj) > 0);
  REQUIRE(std::filesystem::file_size(ply) > 0);
  std::filesystem::remove(obj);
  std::filesystem::remove(ply);
}
