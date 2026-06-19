#include <ncg/record/metrics.hpp>

#include <ncg/core/error.hpp>

#include <cmath>

namespace ncg::record {
namespace {

Tensor as_bchw(const Tensor& t) {
  NCG_CHECK(t.dim() == 3 || t.dim() == 4, "metrics: expected CHW or BCHW, got dim {}", t.dim());
  auto x = t.to(at::kFloat);
  return t.dim() == 3 ? x.unsqueeze(0) : x;
}

// [C,1,ws,ws] separable Gaussian window for grouped conv.
Tensor gaussian_window(int ws, double sigma, int64_t channels, const at::TensorOptions& opts) {
  auto coords = torch::arange(ws, opts) - (ws - 1) / 2.0;
  auto g = torch::exp(-(coords * coords) / (2.0 * sigma * sigma));
  g = g / g.sum();
  auto w2 = torch::matmul(g.unsqueeze(1), g.unsqueeze(0));  // [ws,ws]
  return w2.view({1, 1, ws, ws}).expand({channels, 1, ws, ws}).contiguous();
}

}  // namespace

double mae(const Tensor& a, const Tensor& b) {
  NCG_CHECK(a.sizes() == b.sizes(), "mae: shape mismatch");
  return (a.to(at::kFloat) - b.to(at::kFloat)).abs().mean().item<double>();
}

double psnr(const Tensor& a, const Tensor& b, double max_val) {
  NCG_CHECK(a.sizes() == b.sizes(), "psnr: shape mismatch");
  const double mse = (a.to(at::kFloat) - b.to(at::kFloat)).pow(2).mean().item<double>();
  if (mse <= 1e-12) return 99.0;
  return 10.0 * std::log10((max_val * max_val) / mse);
}

double ssim(const Tensor& a, const Tensor& b, double max_val) {
  NCG_CHECK(a.sizes() == b.sizes(), "ssim: shape mismatch");
  const auto x = as_bchw(a);
  const auto y = as_bchw(b);
  const int64_t c = x.size(1);
  const int ws = 11;
  const auto window = gaussian_window(ws, 1.5, c, x.options());
  const int pad = ws / 2;

  auto conv = [&](const Tensor& t) {
    return torch::conv2d(t, window, /*bias=*/{}, /*stride=*/1, /*padding=*/pad, /*dilation=*/1,
                         /*groups=*/c);
  };

  const auto mu_x = conv(x);
  const auto mu_y = conv(y);
  const auto mu_x2 = mu_x * mu_x;
  const auto mu_y2 = mu_y * mu_y;
  const auto mu_xy = mu_x * mu_y;
  const auto sig_x2 = conv(x * x) - mu_x2;
  const auto sig_y2 = conv(y * y) - mu_y2;
  const auto sig_xy = conv(x * y) - mu_xy;

  const double c1 = (0.01 * max_val) * (0.01 * max_val);
  const double c2 = (0.03 * max_val) * (0.03 * max_val);
  const auto ssim_map =
      ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2));
  return ssim_map.mean().item<double>();
}

}  // namespace ncg::record
