#pragma once

#include <ncg/core/error.hpp>

#include <spdlog/fmt/fmt.h>
#include <torch/torch.h>

#include <string>

namespace ncg::test {

/// Rich result of an allclose-style comparison: localizes the worst element so a failing
/// parity test tells you *where* and *how badly* it diverged, not just pass/fail.
struct CompareResult {
  bool passed = false;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  int64_t n_violations = 0;
  int64_t numel = 0;
  int64_t argmax_flat = -1;
  double worst_actual = 0.0;
  double worst_expected = 0.0;

  std::string summary() const {
    return fmt::format(
        "{} | max_abs_err={:.3e} max_rel_err={:.3e} violations={}/{} "
        "@flat[{}] (actual={:.6g} expected={:.6g})",
        passed ? "PASS" : "FAIL", max_abs_err, max_rel_err, n_violations, numel, argmax_flat,
        worst_actual, worst_expected);
  }
};

/// Compares against the numpy/torch convention |a - e| <= atol + rtol*|e|, in fp64 on CPU.
inline CompareResult compare_allclose(const at::Tensor& actual, const at::Tensor& expected,
                                      double rtol = 1e-3, double atol = 1e-4) {
  NCG_CHECK(actual.sizes() == expected.sizes(),
            "compare_allclose: shape mismatch (actual {} vs expected {})",
            actual.sizes(), expected.sizes());

  const auto a = actual.detach().to(at::kCPU, at::kDouble).contiguous();
  const auto e = expected.detach().to(at::kCPU, at::kDouble).contiguous();

  CompareResult r;
  r.numel = a.numel();
  if (r.numel == 0) {
    r.passed = true;
    return r;
  }

  const auto abs_err = (a - e).abs();
  const auto tol = atol + rtol * e.abs();
  const auto violations = (abs_err > tol);

  r.n_violations = violations.sum().item<int64_t>();
  r.max_abs_err = abs_err.max().item<double>();
  r.max_rel_err = (abs_err / (e.abs() + 1e-12)).max().item<double>();
  r.argmax_flat = abs_err.argmax().item<int64_t>();
  r.worst_actual = a.flatten()[r.argmax_flat].item<double>();
  r.worst_expected = e.flatten()[r.argmax_flat].item<double>();
  r.passed = (r.n_violations == 0);
  return r;
}

}  // namespace ncg::test
