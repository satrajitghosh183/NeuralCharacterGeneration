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

/// Auto-rig a mesh (UniRig-class) or inherit the SMPL-X rig. SKELETON — later phase.
RiggedMesh autorig(const Tensor& vertices, const Tensor& faces);

/// Export a rigged mesh to FBX/glTF for Unity/Unreal. SKELETON — later phase.
void export_rigged(const RiggedMesh& mesh, const std::string& path);

}  // namespace ncg::rig
