#include <ncg/fit/fit_image.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

namespace ncg::fit {

recon::GaussianCloud fit_gaussians_to_views(const std::vector<Tensor>& targets_in,
                                            const std::vector<runtime::Camera>& cameras,
                                            const FitConfig& cfg, record::Recorder* rec) {
  NCG_CHECK(!targets_in.empty(), "fit: no targets");
  NCG_CHECK(targets_in.size() == cameras.size(), "fit: targets/cameras count mismatch");

  const auto device = cameras.front().R.device();
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(device);

  std::vector<Tensor> targets;
  targets.reserve(targets_in.size());
  for (const auto& t : targets_in) {
    NCG_CHECK(t.dim() == 3 && t.size(0) == 3, "fit: each target must be [3,H,W]");
    targets.push_back(t.to(device, at::kFloat));
  }

  const int64_t n = cfg.num_gaussians;
  // Unconstrained leaves -> constrained params via exp/sigmoid.
  auto positions = (torch::randn({n, 3}, opts) * cfg.init_spread).detach().requires_grad_(true);
  auto log_scales =
      torch::full({n, 3}, std::log(cfg.init_scale), opts).detach().requires_grad_(true);
  auto color_logits = torch::zeros({n, 3}, opts).detach().requires_grad_(true);
  auto opacity_logits = torch::zeros({n, 1}, opts).detach().requires_grad_(true);

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
    const auto g = build_cloud();
    Tensor loss = torch::zeros({}, opts);
    double psnr_sum = 0.0;
    for (size_t v = 0; v < cameras.size(); ++v) {
      const auto out = runtime::render_soft(g, cameras[v], {0.0F, 0.0F, 0.0F});
      loss = loss + torch::mse_loss(out.image, targets[v]);
      if (rec != nullptr) psnr_sum += record::psnr(out.image.detach(), targets[v]);
    }
    loss.backward();
    optimizer.step();

    if (rec != nullptr && (it % cfg.log_every == 0 || it == cfg.iterations - 1)) {
      const double l = loss.item<double>();
      const double ps = psnr_sum / static_cast<double>(cameras.size());
      rec->log_scalar("fit", "loss", l);
      rec->log_scalar("fit", "psnr", ps);
      NCG_LOG_INFO("fit it={} views={} loss={:.5f} psnr={:.2f}", it, cameras.size(), l, ps);
    }
  }

  torch::NoGradGuard ng;
  auto g = build_cloud();
  g.positions = g.positions.detach();
  g.scales = g.scales.detach();
  g.opacities = g.opacities.detach();
  g.colors = g.colors.detach();
  g.validate();
  return g;
}

recon::GaussianCloud fit_gaussians_to_image(const Tensor& target_chw,
                                            const runtime::Camera& camera, const FitConfig& cfg,
                                            record::Recorder* rec) {
  auto cloud = fit_gaussians_to_views({target_chw}, {camera}, cfg, rec);
  if (rec != nullptr) {
    const auto out = runtime::render_soft(cloud, camera, {0.0F, 0.0F, 0.0F});
    rec->log_image("fit", "final", out.image.detach());
  }
  return cloud;
}

}  // namespace ncg::fit
