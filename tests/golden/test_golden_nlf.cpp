#include <catch2/catch_test_macros.hpp>

#include <golden_fixture.hpp>
#include <tensor_compare.hpp>

#include <ncg/body/nlf.hpp>
#include <ncg/core/device.hpp>
#include <ncg/io/npy.hpp>

#include <torch/torch.h>

// Per-layer parity for the NLF port. OBSOLETE for the shipped path: NLF is now loaded as a
// TorchScript module (torch::jit), so forward parity is exact by construction and there is no
// per-layer port to validate. This test therefore only runs against a legacy *per-layer* golden
// (manifest with weights + stages); the TorchScript I/O golden written by tools/dump_nlf.py is a
// different format and is intentionally skipped here.

TEST_CASE("NLF per-layer parity", "[golden]") {
  if (!ncg::test::golden_available("nlf")) {
    SKIP("no golden data for nlf (vendor weights + run tools/dump_golden.py to enable)");
  }
  ncg::set_deterministic_fp32(true);

  const auto manifest = ncg::test::load_manifest("nlf");
  if (manifest.weights.empty() || manifest.stages.empty()) {
    SKIP("nlf golden is not a per-layer dump (TorchScript NLF parity is exact by construction)");
  }
  const auto input = ncg::test::load_golden("nlf", manifest.input);
  const auto weights = (ncg::test::golden_dir("nlf") / manifest.weights).string();

  auto nlf = ncg::body::Nlf::load(weights, at::kCPU);
  const auto params = nlf.predict(input);

  const auto& last = manifest.stages.back();
  const auto expected = ncg::test::load_golden("nlf", last.ref_file);
  // The final stage convention is the predicted pose (adjust when the port lands).
  const auto r = ncg::test::compare_allclose(params.pose_aa, expected, last.rtol, last.atol);
  INFO(r.summary());
  REQUIRE(r.passed);
}
