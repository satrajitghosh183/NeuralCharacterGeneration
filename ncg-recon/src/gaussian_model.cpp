#include <ncg/recon/gaussian_model.hpp>

#include <ncg/core/error.hpp>

namespace ncg::recon {

void GaussianCloud::to_(at::Device device) {
  positions = positions.to(device);
  scales = scales.to(device);
  rotations = rotations.to(device);
  opacities = opacities.to(device);
  colors = colors.to(device);
}

void GaussianCloud::validate() const {
  const int64_t n = size();
  NCG_CHECK(positions.dim() == 2 && positions.size(1) == 3, "GaussianCloud: positions must be [N,3]");
  NCG_CHECK(scales.sizes() == positions.sizes(), "GaussianCloud: scales must be [N,3]");
  NCG_CHECK(rotations.dim() == 2 && rotations.size(0) == n && rotations.size(1) == 4,
            "GaussianCloud: rotations must be [N,4]");
  NCG_CHECK(opacities.dim() == 2 && opacities.size(0) == n && opacities.size(1) == 1,
            "GaussianCloud: opacities must be [N,1]");
  NCG_CHECK(colors.dim() == 2 && colors.size(0) == n && colors.size(1) == 3,
            "GaussianCloud: colors must be [N,3]");
  const auto dev = positions.device();
  NCG_CHECK(scales.device() == dev && rotations.device() == dev && opacities.device() == dev &&
                colors.device() == dev,
            "GaussianCloud: tensors on mixed devices");
}

}  // namespace ncg::recon
