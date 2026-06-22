#pragma once

#include <ncg/body/smplx.hpp>
#include <ncg/core/tensor.hpp>

#include <memory>
#include <string>

namespace ncg::body {

/// NLF (Neural Localizer Fields, Sarandi & Pons-Moll, NeurIPS 2024): regresses SMPL/SMPL-X
/// pose + shape from a single in-the-wild image. This is the project's primary body anchor.
///
/// PORT STRATEGY (see docs/parity.md + the cpp-cuda-direction memory): NLF is released as a
/// **TorchScript** module (`nlf_l_multi.torchscript`), so we load it directly with
/// `torch::jit::load` in C++ and run its `detect_smpl_batched` method — no Python at runtime,
/// no blind layer-by-layer reimplementation, and parity is exact by construction (same graph).
/// We port (load weights), we do not retrain. Custom CUDA kernels can replace hot paths later
/// without changing this interface.
///
/// Confirm the output structure against the real checkpoint with `tools/dump_nlf.py` (dumps the
/// output dict layout + a golden), then lock it with `tests/golden/test_golden_nlf.cpp`.
struct NlfConfig {
  /// TorchScript entry point. `detect_smpl_batched(frames_u8, model_name=...)`.
  std::string method = "detect_smpl_batched";
  /// Body model to fit. NLF is model-agnostic; "smplx" yields a [55*3] pose + 10 betas that
  /// match SmplxModel. (Default "smpl" would give a 24-joint pose that SmplxModel can't use.)
  std::string model_name = "smplx";
  /// Which detection to keep when the image has several people (0 = first / highest score).
  int detection = 0;
};

/// Full NLF prediction for one detection: parametric body + the data needed for appearance
/// capture. `vertices2d` are the SMPL-X mesh vertices projected into the source image (pixel
/// coords), so they can be sampled for per-vertex color (see recon::sample_vertex_colors).
struct NlfPrediction {
  SmplxParams params;   // pose/betas/transl for SmplxModel::forward
  Tensor vertices2d;    // [V,2] image-space (x,y) of the posed mesh vertices
  Tensor vertices3d;    // [V,3] camera-space mesh vertices (z = depth, for visibility)
};

class Nlf {
public:
  /// Load the released TorchScript module onto `device`. Throws if the file is missing or not
  /// a loadable TorchScript graph.
  static Nlf load(const std::string& torchscript_path, at::Device device, NlfConfig cfg = {});

  /// Full prediction (params + projected vertices) for the chosen detection, from a CHW float
  /// image in [0,1]. The image is converted to the uint8 RGB [1,3,H,W] batch NLF expects.
  NlfPrediction detect(const Tensor& image_chw) const;

  /// Convenience: just the SMPL-X params (batch 1), ready for SmplxModel::forward.
  SmplxParams predict(const Tensor& image_chw) const { return detect(image_chw).params; }

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace ncg::body
