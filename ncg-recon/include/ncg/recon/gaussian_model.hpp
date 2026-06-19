#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::recon {

/// A set of 3D Gaussians (the splat representation). All tensors share leading dim N and
/// live on the same device. Colors are linear RGB in [0, 1] (SH degree 0 for now).
struct GaussianCloud {
  Tensor positions;  // [N,3] f32 world-space means
  Tensor scales;     // [N,3] f32 world-space std-devs (per-axis)
  Tensor rotations;  // [N,4] f32 unit quaternion (w,x,y,z)
  Tensor opacities;  // [N,1] f32 in [0,1]
  Tensor colors;     // [N,3] f32 linear RGB in [0,1]

  int64_t size() const { return positions.defined() ? positions.size(0) : 0; }
  at::Device device() const { return positions.device(); }

  void to_(at::Device device);
  /// Throws if shapes/devices are inconsistent.
  void validate() const;
};

}  // namespace ncg::recon
