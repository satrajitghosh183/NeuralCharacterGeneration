#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>

namespace ncg::recon {

/// Deforms a Gaussian cloud by a per-Gaussian rigid transform (linear blend skinning result).
/// Positions are transformed by the homogeneous 4x4; scales/opacity/colors are carried.
/// For body avatars, `transforms` is SmplxOutput::vertex_transforms when the cloud was
/// initialized one Gaussian per SMPL-X vertex — i.e. this animates the avatar to a new pose.
///   transforms : [N,4,4] matching the cloud size.
/// NOTE: isotropic-Gaussian MVP — rotations are not yet re-oriented (fine for the equal-axis
/// init from gaussians_on_body); anisotropic rotation update is a later refinement.
GaussianCloud deform_gaussians(const GaussianCloud& cloud, const Tensor& transforms);

}  // namespace ncg::recon
