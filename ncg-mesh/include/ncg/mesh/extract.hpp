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

/// Export a triangle mesh as binary glTF (.glb) — the format Unity and Unreal import directly.
/// `vertices` [V,3], `faces` [F,3]; `normals` [V,3] and `colors` [V,3] are optional (pass an
/// undefined Tensor to omit). Colors become the COLOR_0 vertex attribute. Produces a single-file,
/// self-contained, vertex-colored static mesh (embedded skinning is a later addition).
void write_glb(const Tensor& vertices, const Tensor& faces, const Tensor& normals,
               const Tensor& colors, const std::string& path);

/// Export a *rigged* binary glTF (.glb): the mesh plus a skeleton + per-vertex skinning, so the
/// character animates in Unity/Unreal. Skin weights are reduced to the glTF-standard top-4
/// influences per vertex and renormalized.
///   joints [J,3] rest-pose joint positions, parents [J] int64 hierarchy, skin_weights [V,J].
void write_glb_skinned(const Tensor& vertices, const Tensor& faces, const Tensor& normals,
                       const Tensor& colors, const Tensor& joints, const Tensor& parents,
                       const Tensor& skin_weights, const std::string& path);

/// Rigged glTF (.glb) WITH a baked skeletal animation — the avatar plays it on import in
/// Unity/Unreal. `rot_quats` [T,J,4] are per-frame local joint rotations as glTF quaternions
/// (x,y,z,w); `times` [T] are keyframe times in seconds. Requires normals + colors.
void write_glb_animated(const Tensor& vertices, const Tensor& faces, const Tensor& normals,
                        const Tensor& colors, const Tensor& joints, const Tensor& parents,
                        const Tensor& skin_weights, const Tensor& rot_quats, const Tensor& times,
                        const std::string& path);

/// Export a **UV-textured, rigged** binary glTF (.glb): the SMPL-X mesh with a baked albedo texture
/// (the per-texel C5 albedo) instead of per-vertex colors, plus the skeleton + skinning. Because
/// glTF needs one UV per vertex, vertices are unwelded at UV seams by (geometry-vertex, uv-vertex)
/// pairs. `uv_coords` [n_uv,2] + `uv_faces` [F,3] are the SMPL-X UV layout (separate indices from
/// geometry `faces`). The texture is embedded by reading `texture_png_path` (a PNG already written
/// by save_png) into the GLB image buffer. This is what makes the high-res recovered face visible on
/// the actual character in any engine.
void write_glb_textured(const Tensor& vertices, const Tensor& faces, const Tensor& normals,
                        const Tensor& uv_coords, const Tensor& uv_faces, const Tensor& joints,
                        const Tensor& parents, const Tensor& skin_weights,
                        const std::string& texture_png_path, const std::string& path,
                        const std::string& normal_png_path = "", const Tensor& rot_quats = {},
                        const Tensor& times = {});

/// Export a Gaussian cloud as a **standard 3DGS binary .ply** — the Inria/3DGS convention read by
/// Unity/Unreal Gaussian-splat plugins and standalone viewers (per-splat x,y,z; nx,ny,nz; f_dc_0..2
/// = SH-DC of color; opacity as inverse-sigmoid; scale_0..2 as log; rot_0..3 quaternion w,x,y,z).
/// This is the high-fidelity, engine-portable render asset. When `skin_joints` [N,4] (int) and
/// `skin_weights` [N,4] are given, a sidecar "<path>.skin" is also written (binary: int32 N, then
/// N×(4×int32 joint + 4×float32 weight)) so a GS-skinning shader deforms the splats with the
/// SMPL-X skeleton — the same rig the physics body uses, so the splats follow real physics.
void write_gaussian_ply(const recon::GaussianCloud& cloud, const std::string& path,
                        const Tensor& skin_joints = {}, const Tensor& skin_weights = {});

}  // namespace ncg::mesh
