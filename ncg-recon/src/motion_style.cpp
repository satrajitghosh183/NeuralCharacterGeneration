#include <ncg/recon/motion_style.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <vector>

namespace ncg::recon {

MotionStyleResult solve_motion_style(const std::vector<Tensor>& clips_in,
                                     const MotionStyleConfig& cfg) {
  NCG_CHECK(!clips_in.empty(), "solve_motion_style: no clips");
  const auto opts = clips_in.front().options();
  const int64_t D = clips_in.front().size(1);
  std::vector<Tensor> clips;
  int64_t total = 0;
  for (const auto& c : clips_in) {
    NCG_CHECK(c.dim() == 2 && c.size(1) == D, "solve_motion_style: clips must be [T,D] with equal D");
    clips.push_back(c.to(at::kFloat).contiguous());
    total += c.size(0);
  }
  const int64_t R = std::min<int64_t>({cfg.rank, D, total});
  const auto Ir = torch::eye(R, opts);

  // Init the shared style basis from the top-R right singular vectors of all poses (PCA).
  auto Yall = torch::cat(clips, 0);  // [total, D]
  auto svd = torch::linalg_svd(Yall, /*full_matrices=*/false);
  auto W = std::get<2>(svd).narrow(0, 0, R).contiguous();  // [R,D]

  std::vector<Tensor> contents(clips.size());
  std::vector<Tensor> cons(clips.size());
  for (size_t i = 0; i < clips.size(); ++i)
    cons[i] = torch::ones({clips[i].size(0)}, opts);

  for (int it = 0; it < cfg.iterations; ++it) {
    const auto WWt_inv = torch::linalg_inv(torch::matmul(W, W.t()) + cfg.ridge * Ir);  // [R,R]

    // E-step content + robust weights, accumulate style normal equations.
    auto A = torch::zeros({R, R}, opts);
    auto B = torch::zeros({R, D}, opts);
    for (size_t i = 0; i < clips.size(); ++i) {
      const auto& Y = clips[i];                                  // [T,D]
      const auto Phi = torch::matmul(torch::matmul(Y, W.t()), WWt_inv);  // [T,R]
      contents[i] = Phi;
      if (cfg.robust) {
        const auto resid = (Y - torch::matmul(Phi, W)).norm(2, 1);  // [T]
        float scale = cfg.robust_scale;
        if (scale <= 0.0F) {
          const auto med = resid.median().item<float>();
          scale = std::max(1e-4F, 1.4826F * med);
        }
        cons[i] = torch::exp(-(resid * resid) / (2.0F * scale * scale));  // [T]
      }
      const auto wPhi = cons[i].unsqueeze(1) * Phi;  // [T,R]
      A = A + torch::matmul(Phi.t(), wPhi);
      B = B + torch::matmul(wPhi.t(), Y);
    }
    W = torch::matmul(torch::linalg_inv(A + cfg.ridge * Ir), B);  // [R,D]
  }

  return {W, contents, cons};
}

Tensor apply_motion_style(const Tensor& poses, const Tensor& style) {
  NCG_CHECK(poses.dim() == 2 && style.dim() == 2 && poses.size(1) == style.size(1),
            "apply_motion_style: poses [T,D] and style [R,D] need matching D");
  const auto W = style.to(poses.options());
  const auto Ir = torch::eye(W.size(0), W.options());
  // Project the poses onto the style row-space: P = W^T (W W^T)^-1 W.
  const auto WWt_inv = torch::linalg_inv(torch::matmul(W, W.t()) + 1e-6 * Ir);
  return torch::matmul(torch::matmul(torch::matmul(poses, W.t()), WWt_inv), W);
}

}  // namespace ncg::recon
