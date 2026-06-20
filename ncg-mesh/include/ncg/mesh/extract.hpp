#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>

#include <array>
#include <string>
#include <vector>

namespace ncg::mesh {

struct TriMesh {
  Tensor vertices;  // [V,3] f32
  Tensor faces;     // [F,3] int64
  int64_t num_verts() const { return vertices.defined() ? vertices.size(0) : 0; }
  int64_t num_faces() const { return faces.defined() ? faces.size(0) : 0; }
};

/// Marching cubes on a dense scalar field `field` of shape [gz*gy*gx] (x fastest), extracting
/// the `iso` level surface. `origin` is the world position of voxel (0,0,0); `spacing` is the
/// per-axis voxel size. Produces a (non-welded) triangle mesh. Self-contained, CPU.
TriMesh marching_cubes(const std::vector<float>& field, int gx, int gy, int gz, float iso,
                       std::array<float, 3> origin, std::array<float, 3> spacing);

/// Extracts a surface mesh from a Gaussian cloud: samples a density field on a
/// `grid_resolution`^3 grid over the cloud's bounding box, then marching cubes.
TriMesh extract_mesh(const recon::GaussianCloud& gaussians, int grid_resolution = 128);

/// Area-weighted per-vertex normals [V,3] (unit length). Useful for shading / relighting.
Tensor compute_vertex_normals(const TriMesh& mesh);

/// Write a TriMesh to Wavefront OBJ / binary-free ASCII PLY.
void write_obj(const TriMesh& mesh, const std::string& path);
void write_ply(const TriMesh& mesh, const std::string& path);

}  // namespace ncg::mesh
