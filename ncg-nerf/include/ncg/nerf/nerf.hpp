#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace ncg::nerf {

struct NerfConfig {
  int num_freqs = 6;      // positional-encoding frequency bands
  int hidden = 128;       // MLP width
  int hidden_layers = 4;  // MLP depth
  int samples = 64;       // samples per ray
  float near = 0.1F;
  float far = 6.0F;
};

/// A compact NeRF: positional-encoded MLP mapping a world point -> (density, RGB).
/// Position-only (no view direction) for v1. Trained by per-scene optimization (no dataset),
/// exactly like 3DGS fitting — the implicit-field counterpart to the explicit Gaussian cloud.
class TinyNerf : public torch::nn::Module {
public:
  explicit TinyNerf(const NerfConfig& cfg);

  /// points: [...,3] world coords. Returns {sigma [...,1] >=0, rgb [...,3] in (0,1)}.
  std::pair<Tensor, Tensor> forward(const Tensor& points);

  const NerfConfig& config() const { return cfg_; }

private:
  Tensor encode(const Tensor& x) const;  // positional encoding

  NerfConfig cfg_;
  int enc_dim_;
  torch::nn::Sequential trunk_{nullptr};
  torch::nn::Linear sigma_head_{nullptr};
  torch::nn::Linear rgb_head_{nullptr};
};

/// World-space pixel rays for a camera. Returns {origins [H*W,3], dirs [H*W,3] (unit)}.
std::pair<Tensor, Tensor> camera_rays(const runtime::Camera& camera);

/// Differentiable volume rendering of a TinyNerf from a camera. Output matches the splat
/// renderer's RenderOutput so the two are interchangeable / compositable.
runtime::RenderOutput render_volume(TinyNerf& nerf, const runtime::Camera& camera,
                                    std::array<float, 3> background = {0.0F, 0.0F, 0.0F});

/// Hybrid composite: `front` (e.g. the opaque Gaussian surface) over `back` (e.g. the NeRF
/// volume) using the **premultiplied-alpha** "over" operator:
///   out.image = front.image + (1 - front.alpha) * back.image
/// This matches what render_volume and the splat renderer emit (image is the sum of weighted
/// color, already premultiplied by occupancy), so a transparent pixel contributes nothing and
/// a half-covered edge blends correctly without double-darkening. Inputs are assumed
/// premultiplied; do not pass straight-alpha (non-premultiplied) colors.
runtime::RenderOutput composite_over(const runtime::RenderOutput& front,
                                     const runtime::RenderOutput& back);

struct NerfFitConfig {
  int iterations = 200;
  double lr = 5e-3;
  int log_every = 25;
};

/// Per-scene fit (optimization, no dataset): overfit a TinyNerf to target views via Adam.
std::shared_ptr<TinyNerf> fit_nerf_to_views(const std::vector<Tensor>& targets,
                                            const std::vector<runtime::Camera>& cameras,
                                            const NerfConfig& nerf_cfg, const NerfFitConfig& fit_cfg,
                                            record::Recorder* recorder = nullptr);

}  // namespace ncg::nerf
