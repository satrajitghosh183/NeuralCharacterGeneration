#pragma once

#include <ncg/core/tensor.hpp>

#include <vector>

namespace ncg::recon {

/// Single-photo appearance capture: bilinearly sample an image at projected vertex locations to
/// get a per-vertex color. This is the first step from a gray body to one that looks like the
/// person — color is keyed by vertex identity, so it is independent of the body pose/orientation
/// we render in (sample from the photo's frame, paint the canonical avatar).
///
///   image_chw : [3,H,W] float RGB in [0,1]
///   verts2d   : [V,2] pixel coordinates (x=column, y=row), e.g. NLF's `vertices2d`
/// Returns [V,3] linear RGB in [0,1]; samples outside the image clamp to the border.
///
/// v1 has no visibility test, so vertices facing away from the camera sample whatever pixel they
/// project onto (to be resolved by the z-buffer/uncertainty weighting and cross-photo fusion).
Tensor sample_vertex_colors(const Tensor& image_chw, const Tensor& verts2d);

/// Per-vertex visibility for one view, via a point z-buffer over the projected vertices: a
/// vertex is visible (weight 1) only if it is (near) the frontmost vertex landing on its pixel,
/// so occluded / back-facing vertices that project onto the silhouette are rejected.
///
///   verts2d   : [V,2] pixel coords (x=column, y=row)
///   depth     : [V] camera-space depth, smaller = closer to the camera (e.g. vertices3d z)
///   height/width : image size the projection lives in
///   depth_tol : a vertex within this of the frontmost depth at its pixel still counts visible
/// Returns [V] float in {0,1}. (No surface rasterization — point visibility on the dense mesh
/// is enough to gate appearance; refined later if needed.)
Tensor vertex_visibility(const Tensor& verts2d, const Tensor& depth, int64_t height,
                         int64_t width, double depth_tol = 0.05);

/// Cross-photo appearance fusion (the project's core novelty): merge per-vertex colors from
/// several casual photos into one coherent texture, weighting each view by its per-vertex
/// confidence (visibility, and optionally foreshortening/uncertainty). Vertices seen in no view
/// keep `coverage` 0 and a neutral fill, so the caller can flag them for inpainting/symmetry.
struct FusedAppearance {
  Tensor colors;    // [V,3] fused linear RGB
  Tensor coverage;  // [V] summed weight across views (0 => never seen)
};

/// colors[i], weights[i] are the [V,3] color and [V] confidence from view i (same V, same
/// vertex ordering across views). Returns the confidence-weighted mean per vertex.
FusedAppearance fuse_vertex_colors(const std::vector<Tensor>& colors,
                                   const std::vector<Tensor>& weights);

}  // namespace ncg::recon
