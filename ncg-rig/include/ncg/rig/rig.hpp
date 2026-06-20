#pragma once

#include <ncg/core/tensor.hpp>

#include <string>

namespace ncg::rig {

/// A skinned mesh: geometry + skeleton + per-vertex skinning weights, ready to export.
struct RiggedMesh {
  Tensor vertices;       // [V,3]
  Tensor faces;          // [F,3] int64
  Tensor joints;         // [J,3]
  Tensor parents;        // [J] int64
  Tensor skin_weights;   // [V,J]
};

/// Package an explicit rig into a RiggedMesh (e.g. inherited from SMPL-X: joints, parents and
/// per-vertex skinning come for free from the body model). Validates shapes. REAL.
RiggedMesh make_rigged(const Tensor& vertices, const Tensor& faces, const Tensor& joints,
                       const Tensor& parents, const Tensor& skin_weights);

/// Auto-rig an arbitrary mesh (UniRig-class, learned). GATED on UniRig weights — throws.
RiggedMesh autorig(const Tensor& vertices, const Tensor& faces);

/// Export a rigged mesh to `<path>.obj` (geometry) + `<path>.rig.json` (skeleton + skinning).
/// A portable intermediate; native FBX/glTF engine export is a later phase. REAL.
void export_rigged(const RiggedMesh& mesh, const std::string& path);

}  // namespace ncg::rig
