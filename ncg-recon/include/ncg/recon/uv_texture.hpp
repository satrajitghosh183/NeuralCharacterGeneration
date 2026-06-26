#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::recon {

/// Per-texel correspondence from a UV rasterization of the SMPL-X UV layout. For a `res`×`res`
/// texture, each texel either falls inside a UV triangle (then it carries that triangle's geometry
/// face index and barycentric weights) or is empty. This is what lifts appearance recovery from
/// per-vertex (~10⁴) to per-texel (res², e.g. 6.5×10⁴ at 256) — the resolution a recognizable face
/// needs.
struct UVRaster {
  Tensor face;  // [res*res] int64, geometry-face index per texel, or -1 if empty
  Tensor bary;  // [res*res, 3] f32 barycentric weights (rows sum to 1 on valid texels, else 0)
  int res = 0;
};

/// Rasterize the UV triangles into a `res`×`res` grid. `uv_coords` [n_uv,2] are texture coords in
/// [0,1]; `uv_faces` [F,3] index into `uv_coords`; the returned `face` indexes the *geometry* face
/// list (same row `f`, so geometry vertices are `faces[face]`). Texel centers are sampled at
/// ((c+0.5)/res, (r+0.5)/res) with v measured top-down (row 0 = v=0); flip outside if your texture
/// convention differs. CPU, one-time precompute.
UVRaster uv_rasterize(const Tensor& uv_coords, const Tensor& uv_faces, int res);

/// Gather a per-vertex (or per-face-vertex) quantity into a UV texture by barycentric interpolation.
/// `vertex_values` [V,C]; `faces` [F,3] geometry vertex indices (so texel `t` with face `f` and
/// barycentric `β` gets `Σ_k β_k · vertex_values[faces[f,k]]`). Returns [res,res,C] plus a [res,res]
/// validity mask. Used to bake the recovered albedo into a texture and to seed per-texel solves.
Tensor bake_to_uv(const UVRaster& ras, const Tensor& vertex_values, const Tensor& faces,
                  Tensor& mask_out);

}  // namespace ncg::recon
