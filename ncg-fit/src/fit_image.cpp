#include <ncg/fit/fit_image.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

namespace ncg::fit {

recon::GaussianCloud fit_gaussians_to_image(const Tensor& target_chw,
                                            const runtime::Camera& camera, const FitConfig& cfg,
                                            record::Recorder* rec) {
  NCG_CHECK(target_chw.dim() == 3 && target_chw.size(0) == 3, "fit: target must be [3,H,W]");
  const auto device = camera.R.device();
  const auto target = target_chw.to(device, at::kFloat);
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(device);
  const int64_t n = cfg.num_gaussians;

  // Constrained parameters via unconstrained leaves:
  //   scales = exp(log_scales), opacity = sigmoid(opacity_logits), color = sigmoid(color_logits).
  auto positions = (torch::randn({n, 3}, opts) * cfg.init_spread).detach().requires_grad_(true);
  auto log_scales =
      torch::full({n, 3}, std::log(cfg.init_scale), opts).detach().requires_grad_(true);
  auto color_logits = torch::zeros({n, 3}, opts).detach().requires_grad_(true);
  auto opacity_logits = torch::zeros({n, 1}, opts).detach().requires_grad_(true);

  // Constant identity rotations (no gradient).
  auto rotations = torch::zeros({n, 4}, opts);
  rotations.select(1, 0).fill_(1.0);

  torch::optim::Adam optimizer({positions, log_scales, color_logits, opacity_logits},
                               torch::optim::AdamOptions(cfg.lr));

  auto build_cloud = [&]() {
    recon::GaussianCloud g;
    g.positions = positions;
    g.scales = torch::exp(log_scales);
    g.rotations = rotations;
    g.opacities = torch::sigmoid(opacity_logits);
    g.colors = torch::sigmoid(color_logits);
    return g;
  };

  for (int it = 0; it < cfg.iterations; ++it) {
    optimizer.zero_grad();
    const auto out = runtime::render_soft(build_cloud(), camera, {0.0F, 0.0F, 0.0F});
    const auto loss = torch::mse_loss(out.image, target);
    loss.backward();
    optimizer.step();

    if (rec != nullptr && (it % cfg.log_every == 0 || it == cfg.iterations - 1)) {
      const double l = loss.item<double>();
      const double ps = record::psnr(out.image.detach(), target);
      rec->log_scalar("fit", "loss", l);
      rec->log_scalar("fit", "psnr", ps);
      if (it == cfg.iterations - 1) rec->log_image("fit", "final", out.image.detach());
      NCG_LOG_INFO("fit it={} loss={:.5f} psnr={:.2f}", it, l, ps);
    }
  }

  // Detached result.
  torch::NoGradGuard ng;
  auto g = build_cloud();
  g.positions = g.positions.detach();
  g.scales = g.scales.detach();
  g.opacities = g.opacities.detach();
  g.colors = g.colors.detach();
  g.validate();
  return g;
}

}  // namespace ncg::fit
