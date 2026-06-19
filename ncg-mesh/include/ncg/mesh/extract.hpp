#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>

namespace ncg::mesh {

struct TriMesh {
  Tensor vertices;  // [V,3]
  Tensor faces;     // [F,3] int64
};

/// Extract a surface mesh from a Gaussian cloud (e.g. via a density field + marching cubes)
/// and bake PBR textures. SKELETON — Phase 4 (docs/plan.md).
TriMesh extract_mesh(const recon::GaussianCloud& gaussians, int grid_resolution = 256);

}  // namespace ncg::mesh
