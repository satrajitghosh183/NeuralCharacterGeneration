#include <ncg/recon/animate.hpp>

#include <ncg/core/error.hpp>

namespace ncg::recon {

GaussianCloud deform_gaussians(const GaussianCloud& cloud, const Tensor& transforms) {
  cloud.validate();
  const int64_t n = cloud.size();
  NCG_CHECK(transforms.dim() == 3 && transforms.size(0) == n && transforms.size(1) == 4 &&
                transforms.size(2) == 4,
            "deform_gaussians: transforms must be [N,4,4] matching cloud size {}", n);

  const auto T = transforms.to(cloud.positions.device(), at::kFloat);
  const auto ones = torch::ones({n, 1}, cloud.positions.options());
  const auto ph = torch::cat({cloud.positions, ones}, /*dim=*/1).unsqueeze(-1);  // [N,4,1]
  const auto moved = torch::matmul(T, ph).squeeze(-1).index(
      {torch::indexing::Slice(), torch::indexing::Slice(0, 3)});  // [N,3]

  GaussianCloud out = cloud;  // shares scales/rotations/opacity/colors
  out.positions = moved.contiguous();
  out.validate();
  return out;
}

}  // namespace ncg::recon
