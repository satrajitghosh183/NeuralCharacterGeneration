#pragma once

#include <ncg/core/tensor.hpp>

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

}  // namespace ncg::recon
