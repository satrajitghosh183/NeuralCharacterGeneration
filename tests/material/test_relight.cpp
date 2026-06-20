#include <catch2/catch_test_macros.hpp>

#include <ncg/material/relight.hpp>

#include <torch/torch.h>

TEST_CASE("Lambertian relighting responds to light direction", "[material]") {
  const auto albedo = torch::full({3, 4, 4}, 0.8F);
  auto normal = torch::zeros({3, 4, 4});
  normal.select(0, 2).fill_(1.0);  // normals point +z

  // Light from +z: full diffuse -> albedo * (ambient + 1).
  const auto lit = ncg::material::relight(albedo, normal, {0.0F, 0.0F, 1.0F}, {1.0F, 1.0F, 1.0F},
                                          0.1F);
  REQUIRE(torch::allclose(lit, torch::full_like(albedo, 0.8F * 1.1F), 1e-5, 1e-5));

  // Light from -z: back-facing -> ambient only.
  const auto dark = ncg::material::relight(albedo, normal, {0.0F, 0.0F, -1.0F}, {1.0F, 1.0F, 1.0F},
                                           0.1F);
  REQUIRE(torch::allclose(dark, torch::full_like(albedo, 0.8F * 0.1F), 1e-5, 1e-5));
}
