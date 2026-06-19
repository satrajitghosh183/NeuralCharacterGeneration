#pragma once

#include <ncg/body/smplx.hpp>
#include <ncg/core/tensor.hpp>

#include <string>

namespace ncg::body {

/// NLF (Neural Localizer Fields, Sarandi & Pons-Moll, NeurIPS 2024): regresses SMPL-X pose +
/// shape from a single in-the-wild image. This is the project's primary body anchor.
///
/// PORT STATUS: scaffold. The exact upstream architecture + pretrained weights must be
/// vendored before `load`/`predict` are implemented (the first real porting deliverable —
/// see docs/parity.md, tests/golden/test_golden_nlf.cpp). Until then both throw, and the
/// Phase-1 vertical slice uses SmplxModel::neutral_params() or params loaded from a .npy.
class Nlf {
public:
  /// Build the module and load ported weights from a safetensors file.
  static Nlf load(const std::string& weights_path, at::Device device);

  /// Predict SMPL-X params (batch 1) from a CHW float image in [0, 1].
  SmplxParams predict(const Tensor& image_chw) const;
};

}  // namespace ncg::body
