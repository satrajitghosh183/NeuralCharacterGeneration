#include <ncg/select/selector.hpp>

#include <ncg/core/error.hpp>
#include <ncg/io/image.hpp>

#include <algorithm>

namespace ncg::select {

double sharpness_score(const Tensor& image_chw) {
  NCG_CHECK(image_chw.dim() == 3, "sharpness_score: expected CHW image");
  // Luminance.
  const auto img = image_chw.to(at::kFloat);
  const auto c = img.size(0);
  Tensor gray;
  if (c >= 3) {
    const auto w = torch::tensor({0.299F, 0.587F, 0.114F}, img.options()).view({3, 1, 1});
    gray = (img.slice(0, 0, 3) * w).sum(0, /*keepdim=*/true);  // [1,H,W]
  } else {
    gray = img.slice(0, 0, 1);
  }

  // 3x3 Laplacian convolution.
  const auto kernel = torch::tensor({0.0F, 1.0F, 0.0F, 1.0F, -4.0F, 1.0F, 0.0F, 1.0F, 0.0F},
                                    img.options())
                          .view({1, 1, 3, 3});
  const auto lap = torch::conv2d(gray.unsqueeze(0), kernel);  // [1,1,H',W']
  return lap.var().item<double>();
}

std::vector<ScoredImage> score_images(const std::vector<std::string>& paths) {
  std::vector<ScoredImage> out;
  out.reserve(paths.size());
  for (const auto& p : paths) {
    out.push_back({p, sharpness_score(io::load_image(p, 3))});
  }
  std::sort(out.begin(), out.end(),
            [](const ScoredImage& a, const ScoredImage& b) { return a.sharpness > b.sharpness; });
  return out;
}

std::string select_best(const std::vector<std::string>& paths) {
  NCG_CHECK(!paths.empty(), "select_best: no images provided");
  if (paths.size() == 1) return paths.front();
  return score_images(paths).front().path;
}

}  // namespace ncg::select
