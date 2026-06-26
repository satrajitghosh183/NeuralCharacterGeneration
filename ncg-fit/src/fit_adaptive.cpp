#include <ncg/fit/fit_adaptive.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <memory>

namespace ncg::fit {
namespace {

Tensor logit(const Tensor& p) {
  const auto c = p.clamp(1e-4, 1.0 - 1e-4);
  return torch::log(c / (1.0 - c));
}

Tensor gaussian_window(int ws, double sigma, const at::TensorOptions& opts) {
  auto coords = torch::arange(ws, opts) - (ws - 1) / 2.0;
  auto g = torch::exp(-(coords * coords) / (2.0 * sigma * sigma));
  g = g / g.sum();
  return torch::outer(g, g);  // [ws,ws]
}

}  // namespace

Tensor ssim(const Tensor& a_in, const Tensor& b_in) {
  NCG_CHECK(a_in.dim() == 3 && a_in.size(0) == 3, "ssim: inputs must be [3,H,W]");
  const auto a = a_in.unsqueeze(0);  // [1,3,H,W]
  const auto b = b_in.unsqueeze(0);
  const int ws = 11;
  const int ch = static_cast<int>(a.size(1));
  const auto w = gaussian_window(ws, 1.5, a.options())
                     .view({1, 1, ws, ws})
                     .expand({ch, 1, ws, ws})
                     .contiguous();
  const int pad = ws / 2;
  auto conv = [&](const Tensor& x) {
    return torch::conv2d(x, w, /*bias=*/{}, /*stride=*/1, /*padding=*/pad, /*dilation=*/1,
                         /*groups=*/ch);
  };
  const auto mu_a = conv(a);
  const auto mu_b = conv(b);
  const auto mu_a2 = mu_a * mu_a;
  const auto mu_b2 = mu_b * mu_b;
  const auto mu_ab = mu_a * mu_b;
  const auto sig_a2 = conv(a * a) - mu_a2;
  const auto sig_b2 = conv(b * b) - mu_b2;
  const auto sig_ab = conv(a * b) - mu_ab;
  const double c1 = 0.01 * 0.01;
  const double c2 = 0.03 * 0.03;
  const auto ssim_map =
      ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2));
  return ssim_map.mean();
}

recon::GaussianCloud fit_adaptive(const std::vector<Tensor>& targets_in,
                                  const std::vector<runtime::Camera>& cameras,
                                  const recon::GaussianCloud& init, const AdaptiveFitConfig& cfg,
                                  record::Recorder* rec) {
  NCG_CHECK(!targets_in.empty(), "fit_adaptive: no targets");
  NCG_CHECK(targets_in.size() == cameras.size(), "fit_adaptive: targets/cameras mismatch");
  NCG_CHECK(init.size() > 0, "fit_adaptive: init cloud is empty (geometry seed required)");

  const auto device = init.positions.device();
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(device);
  const int64_t V = static_cast<int64_t>(targets_in.size());

  std::vector<Tensor> targets;
  targets.reserve(V);
  for (const auto& t : targets_in) {
    NCG_CHECK(t.dim() == 3 && t.size(0) == 3, "fit_adaptive: each target must be [3,H,W]");
    targets.push_back(t.to(device, at::kFloat));
  }

  // Scene extent (bbox diagonal) sets the split size threshold.
  const double extent = (init.positions.max(0).values - init.positions.min(0).values)
                            .norm()
                            .item<double>();

  // Per-view body masks from the init render's alpha (keeps the fit off the background).
  std::vector<Tensor> masks(V);
  if (cfg.use_mask) {
    torch::NoGradGuard ng;
    for (int64_t v = 0; v < V; ++v) {
      const auto a0 = runtime::render_soft_aniso(init, cameras[v]).alpha;  // [1,H,W]
      masks[v] = (a0 > 0.05).to(at::kFloat);
    }
  }

  // Optimizable leaves (unconstrained; mapped to constrained params per render).
  auto positions = init.positions.detach().clone().set_requires_grad(true);
  auto log_scales = init.scales.detach().clamp_min(1e-6).log().set_requires_grad(true);
  auto quats = init.rotations.detach().clone().set_requires_grad(true);
  auto color_logits = logit(init.colors.detach()).set_requires_grad(true);
  auto opacity_logits = logit(init.opacities.detach()).set_requires_grad(true);
  auto gain = torch::ones({V, 3}, opts).set_requires_grad(cfg.per_view_exposure);
  auto bias = torch::zeros({V, 3}, opts).set_requires_grad(cfg.per_view_exposure);

  using torch::optim::Adam;
  using torch::optim::AdamOptions;
  using torch::optim::OptimizerParamGroup;
  auto make_opt = [&]() {
    auto grp = [](std::vector<Tensor> p, double lr) {
      OptimizerParamGroup g(std::move(p));
      g.set_options(std::make_unique<AdamOptions>(lr));
      return g;
    };
    std::vector<OptimizerParamGroup> groups;
    groups.push_back(grp({positions}, cfg.lr_position));
    groups.push_back(grp({log_scales}, cfg.lr_scale));
    groups.push_back(grp({quats}, cfg.lr_rotation));
    groups.push_back(grp({color_logits}, cfg.lr_color));
    groups.push_back(grp({opacity_logits}, cfg.lr_opacity));
    if (cfg.per_view_exposure) {
      groups.push_back(grp({gain}, 1e-3));
      groups.push_back(grp({bias}, 1e-3));
    }
    return std::make_unique<Adam>(groups, AdamOptions(cfg.lr_position));
  };
  auto optimizer = make_opt();

  auto build_cloud = [&]() {
    recon::GaussianCloud g;
    g.positions = positions;
    g.scales = torch::exp(log_scales);
    g.rotations = quats;
    g.opacities = torch::sigmoid(opacity_logits);
    g.colors = torch::sigmoid(color_logits);
    return g;
  };

  // Positional-gradient accumulator for adaptive density control.
  auto grad_accum = torch::zeros({positions.size(0)}, opts);
  int accum_count = 0;

  for (int it = 0; it < cfg.iterations; ++it) {
    const int64_t v = torch::randint(0, V, {1}, at::kLong).item<int64_t>();
    optimizer->zero_grad();
    const auto g = build_cloud();
    auto out = runtime::render_soft_aniso(g, cameras[v]);
    auto pred = out.image;
    if (cfg.per_view_exposure) {
      pred = (pred * gain[v].view({3, 1, 1}) + bias[v].view({3, 1, 1})).clamp(0.0, 1.0);
    }
    auto tgt = targets[v];
    if (cfg.use_mask) {
      pred = pred * masks[v];
      tgt = tgt * masks[v];
    }
    const auto l1 = torch::l1_loss(pred, tgt);
    const auto dssim = 1.0 - ssim(pred, tgt);
    const auto loss = (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * dssim;
    loss.backward();
    {
      torch::NoGradGuard ng;
      if (positions.grad().defined()) {
        grad_accum = grad_accum + positions.grad().norm(2, /*dim=*/1);
        accum_count += 1;
      }
    }
    optimizer->step();

    // ---- adaptive density control ----
    const bool in_densify = it >= cfg.densify_from && it < cfg.densify_until;
    if (in_densify && it % cfg.densify_every == 0 && accum_count > 0) {
      torch::NoGradGuard ng;
      const auto avg = grad_accum / static_cast<double>(accum_count);
      const auto scales = torch::exp(log_scales);
      const auto max_scale = std::get<0>(scales.max(1));            // [N]
      const auto sel = avg > cfg.grad_threshold;                   // [N] bool
      const auto big = max_scale > cfg.split_scale_frac * extent;  // [N] bool
      const auto clone_m = sel & big.logical_not();
      const auto split_m = sel & big;
      const int64_t n_now = positions.size(0);
      const bool room = n_now < cfg.max_gaussians;

      auto idx_of = [](const Tensor& m) { return m.nonzero().squeeze(1); };
      const auto clone_idx = room ? idx_of(clone_m) : torch::empty({0}, at::kLong);
      const auto split_idx = room ? idx_of(split_m) : torch::empty({0}, at::kLong);

      // Children of split Gaussians: two samples jittered within the parent, smaller scale.
      auto gather = [](const Tensor& src, const Tensor& idx) { return src.index_select(0, idx); };
      std::vector<Tensor> pos_parts{positions};
      std::vector<Tensor> ls_parts{log_scales};
      std::vector<Tensor> q_parts{quats};
      std::vector<Tensor> col_parts{color_logits};
      std::vector<Tensor> op_parts{opacity_logits};
      Tensor keep_m = torch::ones({n_now}, at::kBool).to(device);

      if (clone_idx.numel() > 0) {
        pos_parts.push_back(gather(positions, clone_idx));
        ls_parts.push_back(gather(log_scales, clone_idx));
        q_parts.push_back(gather(quats, clone_idx));
        col_parts.push_back(gather(color_logits, clone_idx));
        op_parts.push_back(gather(opacity_logits, clone_idx));
      }
      if (split_idx.numel() > 0) {
        keep_m.index_put_({split_idx}, false);  // parents removed, replaced by children
        const double inv = 1.0 / cfg.split_jitter;
        for (int child = 0; child < 2; ++child) {
          const auto ps = gather(scales, split_idx);             // [K,3]
          const auto jitter = torch::randn_like(ps) * ps;        // per-axis, isotropic approx
          pos_parts.push_back(gather(positions, split_idx) + jitter);
          ls_parts.push_back((gather(scales, split_idx) * inv).clamp_min(1e-6).log());
          q_parts.push_back(gather(quats, split_idx));
          col_parts.push_back(gather(color_logits, split_idx));
          op_parts.push_back(gather(opacity_logits, split_idx));
        }
      }

      auto positions_n = torch::cat(pos_parts, 0);
      auto log_scales_n = torch::cat(ls_parts, 0);
      auto quats_n = torch::cat(q_parts, 0);
      auto color_logits_n = torch::cat(col_parts, 0);
      auto opacity_logits_n = torch::cat(op_parts, 0);

      // Prune transparent Gaussians (and the just-removed split parents).
      auto op_keep = (torch::sigmoid(opacity_logits_n).squeeze(1) > cfg.prune_opacity);
      // keep_m applies to the first n_now rows; appended rows are all kept.
      auto appended = torch::ones({positions_n.size(0) - n_now}, at::kBool).to(device);
      auto struct_keep = torch::cat({keep_m, appended}, 0);
      auto final_keep = (op_keep & struct_keep).nonzero().squeeze(1);

      positions = positions_n.index_select(0, final_keep).detach().set_requires_grad(true);
      log_scales = log_scales_n.index_select(0, final_keep).detach().set_requires_grad(true);
      quats = quats_n.index_select(0, final_keep).detach().set_requires_grad(true);
      color_logits = color_logits_n.index_select(0, final_keep).detach().set_requires_grad(true);
      opacity_logits = opacity_logits_n.index_select(0, final_keep).detach().set_requires_grad(true);

      grad_accum = torch::zeros({positions.size(0)}, opts);
      accum_count = 0;
      optimizer = make_opt();
    }

    // Periodic opacity reset (cull persistent floaters; they must re-earn opacity).
    if (cfg.opacity_reset_every > 0 && it > 0 && it % cfg.opacity_reset_every == 0 && in_densify) {
      torch::NoGradGuard ng;
      const auto reset = torch::min(torch::sigmoid(opacity_logits),
                                    torch::full_like(opacity_logits, 0.01));
      opacity_logits = logit(reset).detach().set_requires_grad(true);
      optimizer = make_opt();
    }

    if (rec != nullptr && (it % cfg.log_every == 0 || it == cfg.iterations - 1)) {
      const double l = loss.item<double>();
      rec->log_scalar("adaptive", "loss", l);
      rec->log_scalar("adaptive", "gaussians", static_cast<double>(positions.size(0)));
      NCG_LOG_INFO("adaptive it={} N={} loss={:.5f}", it, positions.size(0), l);
      if (cfg.dump_every > 0 && it % cfg.dump_every == 0) {
        const auto im = runtime::render_soft_aniso(build_cloud(), cameras[0]).image.detach();
        rec->log_image("adaptive", "view0", im);
      }
    }
  }

  torch::NoGradGuard ng;
  auto g = build_cloud();
  g.positions = g.positions.detach();
  g.scales = g.scales.detach();
  g.rotations = g.rotations.detach();
  g.opacities = g.opacities.detach();
  g.colors = g.colors.detach();
  g.validate();
  return g;
}

}  // namespace ncg::fit
