#include <catch2/catch_test_macros.hpp>

#include <ncg/core/device.hpp>
#include <ncg/core/kernels/elementwise.hpp>

#include <torch/torch.h>

// The canonical "custom kernel vs LibTorch reference" pattern: any custom kernel must match
// a trusted LibTorch op on random input before it is trusted. This both validates the kernel
// and proves the CUDA + LibTorch + CMake + Catch2 toolchain on the H100.

TEST_CASE("saxpy matches LibTorch reference", "[cuda][core]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available on this machine");
  }
  torch::manual_seed(0);

  const double a = 2.5;
  const auto x = torch::randn({1031}, ncg::f32_cuda());  // non-multiple of block size
  const auto y = torch::randn({1031}, ncg::f32_cuda());

  const auto out = ncg::saxpy(x, y, a);
  const auto ref = a * x + y;

  REQUIRE(out.sizes() == x.sizes());
  REQUIRE(out.device() == x.device());
  REQUIRE(torch::allclose(out, ref, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST_CASE("saxpy rejects shape mismatch", "[cuda][core]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available on this machine");
  }
  const auto x = torch::randn({16}, ncg::f32_cuda());
  const auto y = torch::randn({8}, ncg::f32_cuda());
  REQUIRE_THROWS(ncg::saxpy(x, y, 1.0));
}

TEST_CASE("saxpy handles empty input", "[cuda][core]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available on this machine");
  }
  const auto x = torch::empty({0}, ncg::f32_cuda());
  const auto y = torch::empty({0}, ncg::f32_cuda());
  const auto out = ncg::saxpy(x, y, 1.0);
  REQUIRE(out.numel() == 0);
}
