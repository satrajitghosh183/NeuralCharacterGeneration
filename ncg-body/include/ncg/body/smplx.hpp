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
  Tensor vertices;          // [B, V, 3]
  Tensor joints;            // [B, J, 3]
  Tensor vertex_transforms; // [B, V, 4, 4] per-vertex LBS transform (rest -> posed)
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

  /// Per-vertex skinning weights [V,J] and the joint parent indices [J] — exposed so a
  /// reconstructed mesh can inherit the SMPL-X rig (see ncg::rig::transfer_skinning).
  Tensor lbs_weights() const { return lbs_weights_; }
  Tensor parents() const { return parents_; }

  /// Mesh triangles [F,3] int64 (present iff the converted model included faces). Needed for
  /// vertex normals (relighting) and mesh/glTF export. Undefined for the dummy UV-sphere model.
  Tensor faces() const { return faces_; }
  bool has_faces() const { return faces_.defined() && faces_.numel() > 0; }

  /// UV texture layout (present iff the converted model included `vt`/`ft`): `uv_coords` [n_uv,2]
  /// texture coordinates, `uv_faces` [F,3] indices into `uv_coords` (separate from geometry `faces`
  /// at UV seams). Needed for per-texel albedo recovery and textured glTF export.
  Tensor uv_coords() const { return uv_coords_; }
  Tensor uv_faces() const { return uv_faces_; }
  bool has_uv() const { return uv_coords_.defined() && uv_coords_.numel() > 0; }

  /// Zero shape + T-pose params (batch B) on this model's device — the slice default.
  SmplxParams neutral_params(int64_t batch = 1) const;

private:
  Tensor v_template_;   // [V,3]
  Tensor shapedirs_;    // [V,3,n_betas]
  Tensor posedirs_;     // [V,3,9*(J-1)]
  Tensor J_regressor_;  // [J,V]
  Tensor lbs_weights_;  // [V,J]
  Tensor parents_;      // [J] int64
  Tensor faces_;        // [F,3] int64 (optional)
  Tensor uv_coords_;    // [n_uv,2] f32 (optional)
  Tensor uv_faces_;     // [F,3] int64 (optional)
};

}  // namespace ncg::body
