#pragma once

#include <ncg/core/tensor.hpp>

#include <string>

namespace ncg::body {

/// SMPL-X parameters for a batch of bodies. Axis-angle pose, one 3-vector per joint
/// (including the root as joint 0).
struct SmplxParams {
  Tensor betas;    // [B, n_betas]
  Tensor pose_aa;  // [B, J, 3] axis-angle (joint 0 = global orientation)
  Tensor transl;   // [B, 3]
};

struct SmplxOutput {
  Tensor vertices;  // [B, V, 3]
  Tensor joints;    // [B, J, 3]
};

/// Minimal SMPL-X forward (linear blend skinning). The model buffers are loaded from a
/// safetensors file converted offline from the official SMPL-X .npz (see docs/build.md):
///   v_template [V,3], shapedirs [V,3,n_betas], posedirs [V,3,9*(J-1)],
///   J_regressor [J,V], lbs_weights [V,J], parents [J] (int64).
class SmplxModel {
public:
  static SmplxModel load(const std::string& safetensors_path, at::Device device);

  /// Skins the template given shape + pose. Pose may be [B,J,3] or [B,J*3].
  SmplxOutput forward(const SmplxParams& p) const;

  int64_t num_verts() const { return v_template_.size(0); }
  int64_t num_joints() const { return parents_.size(0); }
  int64_t num_betas() const { return shapedirs_.size(2); }
  at::Device device() const { return v_template_.device(); }

  /// Zero shape + T-pose params (batch B) on this model's device — the slice default.
  SmplxParams neutral_params(int64_t batch = 1) const;

private:
  Tensor v_template_;   // [V,3]
  Tensor shapedirs_;    // [V,3,n_betas]
  Tensor posedirs_;     // [V,3,9*(J-1)]
  Tensor J_regressor_;  // [J,V]
  Tensor lbs_weights_;  // [V,J]
  Tensor parents_;      // [J] int64
};

}  // namespace ncg::body
