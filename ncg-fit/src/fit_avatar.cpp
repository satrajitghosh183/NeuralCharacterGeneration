#include <ncg/fit/fit_avatar.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/fit/fit_adaptive.hpp>  // ssim()
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

#include <memory>

namespace ncg::fit {
namespace {

Tensor inv_sigmoid(const Tensor& p) {
  const auto c = p.clamp(1e-4, 1.0 - 1e-4);
  return torch::log(c / (1.0 - c));
}

// Proper-rotation matrices [N,3,3] -> unit quaternions (w,x,y,z) [N,4], branchless (Shepperd).
Tensor rotmat_to_quat(const Tensor& R) {
  const auto r00 = R.index({torch::indexing::Slice(), 0, 0});
  const auto r01 = R.index({torch::indexing::Slice(), 0, 1});
  const auto r02 = R.index({torch::indexing::Slice(), 0, 2});
  const auto r10 = R.index({torch::indexing::Slice(), 1, 0});
  const auto r11 = R.index({torch::indexing::Slice(), 1, 1});
  const auto r12 = R.index({torch::indexing::Slice(), 1, 2});
  const auto r20 = R.index({torch::indexing::Slice(), 2, 0});
  const auto r21 = R.index({torch::indexing::Slice(), 2, 1});
  const auto r22 = R.index({torch::indexing::Slice(), 2, 2});
  auto nz_sign = [](const Tensor& x) {  // sign with sign(0) -> +1
    return torch::where(x >= 0, torch::ones_like(x), -torch::ones_like(x));
  };
  const auto qw = 0.5 * torch::sqrt(torch::clamp_min(1 + r00 + r11 + r22, 0.0));
  const auto qx = 0.5 * torch::sqrt(torch::clamp_min(1 + r00 - r11 - r22, 0.0)) * nz_sign(r21 - r12);
  const auto qy = 0.5 * torch::sqrt(torch::clamp_min(1 - r00 + r11 - r22, 0.0)) * nz_sign(r02 - r20);
  const auto qz = 0.5 * torch::sqrt(torch::clamp_min(1 - r00 - r11 + r22, 0.0)) * nz_sign(r10 - r01);
  auto q = torch::stack({qw, qx, qy, qz}, 1);  // [N,4]
  return q / q.norm(2, 1, true).clamp_min(1e-8);
}

// Hamilton product of unit quaternions (w,x,y,z), batched.
Tensor quat_mul(const Tensor& a, const Tensor& b) {
  const auto aw = a.select(1, 0);
  const auto ax = a.select(1, 1);
  const auto ay = a.select(1, 2);
  const auto az = a.select(1, 3);
  const auto bw = b.select(1, 0);
  const auto bx = b.select(1, 1);
  const auto by = b.select(1, 2);
  const auto bz = b.select(1, 3);
  const auto w = aw * bw - ax * bx - ay * by - az * bz;
  const auto x = aw * bx + ax * bw + ay * bz - az * by;
  const auto y = aw * by - ax * bz + ay * bw + az * bx;
  const auto z = aw * bz + ax * by - ay * bx + az * bw;
  return torch::stack({w, x, y, z}, 1);
}

}  // namespace

recon::GaussianCloud deform_avatar(const recon::GaussianCloud& canonical, const Tensor& vt_in,
                                   const Tensor& binding) {
  NCG_CHECK(vt_in.dim() == 3 && vt_in.size(1) == 4 && vt_in.size(2) == 4,
            "deform_avatar: vertex_transforms must be [V,4,4]");
  // Select each Gaussian's skinning transform: 1:1 when no binding, else gather by vertex index.
  const auto vt = binding.defined() && binding.numel() > 0 ? vt_in.index_select(0, binding) : vt_in;
  NCG_CHECK(canonical.size() == vt.size(0),
            "deform_avatar: cloud size must match transforms (provide --binding after densify)");
  using torch::indexing::Slice;
  const auto Rm = vt.index({Slice(), Slice(0, 3), Slice(0, 3)});  // [N,3,3]
  const auto tm = vt.index({Slice(), Slice(0, 3), 3});            // [N,3]

  recon::GaussianCloud g;
  g.positions = torch::matmul(Rm, canonical.positions.unsqueeze(2)).squeeze(2) + tm;  // [V,3]
  g.rotations = quat_mul(rotmat_to_quat(Rm), canonical.rotations);                    // [V,4]
  g.scales = canonical.scales;
  g.opacities = canonical.opacities;
  g.colors = canonical.colors;
  return g;
}

AvatarFitResult fit_avatar(const body::SmplxModel& model, const Tensor& betas_in,
                           const std::vector<AvatarFrame>& frames, const Tensor& init_colors,
                           const AvatarFitConfig& cfg, record::Recorder* rec,
                           const Tensor& coverage) {
  NCG_CHECK(!frames.empty(), "fit_avatar: no frames");
  const auto device = model.device();
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(device);
  const int64_t V = model.num_verts();
  const int64_t J = model.num_joints();
  const int64_t F = static_cast<int64_t>(frames.size());
  const auto betas = betas_in.to(opts).reshape({1, -1});
  NCG_CHECK(init_colors.size(0) == V && init_colors.size(1) == 3,
            "fit_avatar: init_colors must be [V,3]");

  // Rest-pose canonical vertices (this avatar's bind pose).
  body::SmplxParams rest;
  rest.betas = betas;
  rest.pose_aa = torch::zeros({1, J, 3}, opts);
  rest.transl = torch::zeros({1, 3}, opts);
  const auto rest_verts = model.forward(rest).vertices.squeeze(0).detach();  // [V,3]

  // Precompute each frame's per-vertex rest->posed transforms (pose is fixed per frame).
  std::vector<Tensor> transforms(F);
  for (int64_t f = 0; f < F; ++f) {
    body::SmplxParams p;
    p.betas = betas;
    p.pose_aa = frames[f].pose_aa.to(opts).reshape({1, J, 3});
    p.transl = frames[f].transl.defined() && frames[f].transl.numel() == 3
                   ? frames[f].transl.to(opts).reshape({1, 3})
                   : torch::zeros({1, 3}, opts);
    transforms[f] = model.forward(p).vertex_transforms.squeeze(0).detach();  // [V,4,4]
  }

  // Densification needs a position-gradient signal, so positions must be trainable then (even at a
  // tiny lr they barely move; the gradient drives clone/split decisions).
  const double lr_pos = (cfg.densify && cfg.lr_position <= 0) ? 1e-4 : cfg.lr_position;

  // Optimizable canonical leaves (reassigned wholesale when densification changes the count).
  auto positions = rest_verts.clone().set_requires_grad(lr_pos > 0);
  auto log_scales = torch::full({V, 3}, std::log(cfg.init_scale), opts).set_requires_grad(true);
  auto quats = torch::zeros({V, 4}, opts);
  quats.select(1, 0).fill_(1.0);
  quats = quats.set_requires_grad(true);
  auto color_logits = inv_sigmoid(init_colors.to(opts)).set_requires_grad(true);
  auto opacity_logits = inv_sigmoid(torch::full({V, 1}, 0.9F, opts)).set_requires_grad(true);
  auto binding = torch::arange(V, at::TensorOptions().dtype(at::kLong).device(device));  // [N]→vert
  auto gain = torch::ones({F, 3}, opts).set_requires_grad(cfg.per_view_exposure);
  auto bias = torch::zeros({F, 3}, opts).set_requires_grad(cfg.per_view_exposure);

  // Bound scales to a human-scale range: collapse (→0) makes the projected covariance singular and
  // explodes the conic-inverse gradient; runaway growth lets one Gaussian dominate the normalized
  // splat. Both drive the fit to NaN, so clamp the rendered scale (gradient still flows in-range).
  const double smin = cfg.min_scale;  // raise (e.g. 3mm) so densified splats overlap into a surface
  const double smax = 0.05;
  auto canonical = [&]() {
    recon::GaussianCloud g;
    g.positions = positions;
    g.scales = torch::exp(log_scales).clamp(smin, smax);
    g.rotations = quats;
    g.opacities = torch::sigmoid(opacity_logits);
    g.colors = torch::sigmoid(color_logits);
    return g;
  };

  using torch::optim::Adam;
  using torch::optim::AdamOptions;
  using torch::optim::OptimizerParamGroup;
  std::vector<Tensor> clip_leaves;
  auto make_opt = [&]() {
    auto grp = [](std::vector<Tensor> p, double lr) {
      OptimizerParamGroup g(std::move(p));
      g.set_options(std::make_unique<AdamOptions>(lr));
      return g;
    };
    std::vector<OptimizerParamGroup> groups;
    groups.push_back(grp({log_scales}, cfg.lr_scale));
    groups.push_back(grp({quats}, cfg.lr_rotation));
    groups.push_back(grp({color_logits}, cfg.lr_color));
    groups.push_back(grp({opacity_logits}, cfg.lr_opacity));
    if (lr_pos > 0) groups.push_back(grp({positions}, lr_pos));
    if (cfg.per_view_exposure) {
      groups.push_back(grp({gain}, 1e-3));
      groups.push_back(grp({bias}, 1e-3));
    }
    clip_leaves = {log_scales, quats, color_logits, opacity_logits};
    if (lr_pos > 0) clip_leaves.push_back(positions);
    if (cfg.per_view_exposure) {
      clip_leaves.push_back(gain);
      clip_leaves.push_back(bias);
    }
    return std::make_unique<Adam>(groups, AdamOptions(cfg.lr_color));
  };
  auto optimizer = make_opt();

  // Per-frame body masks from the posed init render.
  std::vector<Tensor> targets(F);
  std::vector<Tensor> masks(F);
  {
    torch::NoGradGuard ng;
    const auto c0 = canonical();
    for (int64_t f = 0; f < F; ++f) {
      targets[f] = frames[f].target.to(opts);
      if (cfg.use_mask) {
        const auto a0 =
            runtime::render_soft_aniso(deform_avatar(c0, transforms[f], binding), frames[f].camera)
                .alpha;
        masks[f] = (a0 > 0.05).to(at::kFloat);
      }
    }
  }

  auto grad_accum = torch::zeros({positions.size(0)}, opts);
  int accum_count = 0;

  // E1/E5 extraction weighting: sample frames PROPORTIONAL to their confidence (sharpness ×
  // pose-consistency × coverage), so low-confidence frames (blur, jumpy pose, redundant view)
  // contribute less WITHOUT being discarded. Uniform if all weights are equal/unset.
  auto fw = torch::ones({F}, opts);
  for (int64_t i = 0; i < F; ++i) fw[i] = std::max(1e-3F, frames[static_cast<size_t>(i)].weight);
  const bool weighted = (fw.max() - fw.min()).item<float>() > 1e-4F;

  for (int it = 0; it < cfg.iterations; ++it) {
    const int64_t f = weighted ? torch::multinomial(fw, 1).item<int64_t>()
                               : torch::randint(0, F, {1}, at::kLong).item<int64_t>();
    optimizer->zero_grad();
    const auto posed = deform_avatar(canonical(), transforms[f], binding);
    auto pred = runtime::render_soft_aniso(posed, frames[f].camera).image;
    if (cfg.per_view_exposure) {
      pred = (pred * gain[f].view({3, 1, 1}) + bias[f].view({3, 1, 1})).clamp(0.0, 1.0);
    }
    auto tgt = targets[f];
    if (cfg.use_mask) {
      pred = pred * masks[f];
      tgt = tgt * masks[f];
    }
    Tensor l1;
    if (cfg.robust) {
      // Auto-scaled Welsch (IRLS): per-pixel weight exp(-½(r/c)²), c = k·median residual over the
      // body region. Down-weights pixels the canonical can't reconcile (conflicting outfits, bad
      // cameras, junk) so the consensus appearance wins instead of a blurred average.
      const auto diff = pred - tgt;                          // [3,H,W]
      const auto a = diff.abs().mean(0, /*keepdim=*/true);    // [1,H,W] per-pixel residual
      Tensor med;
      if (cfg.use_mask) {
        const auto sel = a.masked_select(masks[f] > 0.5);
        med = sel.numel() > 0 ? sel.median() : a.median();
      } else {
        med = a.median();
      }
      const auto c = (cfg.robust_k * med).clamp_min(1e-3).detach();
      auto w = torch::exp(-0.5 * (a / c).pow(2)).detach();   // [1,H,W]
      if (cfg.use_mask) w = w * masks[f];
      l1 = (w * diff.abs()).sum() / (w.sum() * 3.0 + 1e-6);  // weighted mean over body pixels
    } else {
      l1 = torch::l1_loss(pred, tgt);
    }
    auto loss = (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * (1.0 - ssim(pred, tgt));
    // Deviation regularizer (anti-floater): pull each splat toward its bound vertex's rest position.
    if (cfg.position_reg > 0.0 && lr_pos > 0) {
      const auto anchor = rest_verts.index_select(0, binding);  // [N,3] bound-vertex rest position
      loss = loss + cfg.position_reg * (positions - anchor).pow(2).sum(1).mean();
    }
    // Skip a non-finite step rather than poison Adam's moments with NaN.
    if (!std::isfinite(loss.item<double>())) {
      optimizer->zero_grad();
      continue;
    }
    loss.backward();
    // Zero any non-finite gradient element FIRST: clip_grad_norm_ uses a global norm, so a single
    // NaN/inf grad from one degenerate Gaussian would otherwise poison every parameter.
    for (auto& p : clip_leaves) {
      if (p.grad().defined()) p.mutable_grad() = torch::nan_to_num(p.grad());
    }
    torch::nn::utils::clip_grad_norm_(clip_leaves, 1.0);  // tame remaining gradient spikes
    {
      torch::NoGradGuard ng;
      if (lr_pos > 0 && positions.grad().defined()) {
        grad_accum = grad_accum + positions.grad().norm(2, /*dim=*/1);
        accum_count += 1;
      }
    }
    optimizer->step();

    // HARD anti-floater clamp: a free splat may not leave a thin band around its bound vertex's rest
    // position. With few casual views, soft regularization alone lets splats proliferate as floaters
    // (which wreck the render + identity); this GUARANTEES they stay on the surface so densification
    // only adds detail. cfg.max_dev <= 0 disables (legacy 1:1 path).
    if (cfg.max_dev > 0.0 && lr_pos > 0) {
      torch::NoGradGuard ng;
      const auto anchor = rest_verts.index_select(0, binding);            // [N,3]
      const auto off = positions.detach() - anchor;
      const auto d = off.norm(2, 1, true).clamp_min(1e-9);
      const auto clamped = anchor + off * (d.clamp_max(cfg.max_dev) / d);
      positions.detach().copy_(clamped);
    }
    // FIX2 — opacity floor in COVERED regions: a splat bound to a well-photographed vertex may not
    // go transparent (that is what punched the dark holes where photos disagreed). Unobserved splats
    // are free to fade. Clamp opacity_logits up to the floor where coverage is high.
    if (cfg.opacity_floor > 0.0 && coverage.defined() && coverage.numel() > 0) {
      torch::NoGradGuard ng;
      const auto cov = coverage.to(opts).index_select(0, binding).view({-1, 1});  // [N,1]
      const auto floor_logit = inv_sigmoid(torch::full_like(opacity_logits, cfg.opacity_floor));
      const auto need = (cov > 0.5F) & (opacity_logits.detach() < floor_logit);
      opacity_logits.detach().copy_(torch::where(need, floor_logit, opacity_logits.detach()));
    }

    // ---- adaptive density control (clone/split high-gradient Gaussians; children inherit binding) ----
    const bool in_densify = cfg.densify && it >= cfg.densify_from && it < cfg.densify_until;
    if (in_densify && it % cfg.densify_every == 0 && accum_count > 0 &&
        positions.size(0) < cfg.max_gaussians) {
      torch::NoGradGuard ng;
      const auto avg = grad_accum / static_cast<double>(accum_count);
      const auto scales = torch::exp(log_scales).clamp(smin, smax);
      const auto max_scale = std::get<0>(scales.max(1));                      // [N]
      const auto sel = avg > cfg.densify_grad;                                // [N]
      const auto big = max_scale > cfg.densify_scale_frac * cfg.init_scale;   // [N]
      const auto clone_idx = (sel & big.logical_not()).nonzero().squeeze(1);
      const auto split_idx = (sel & big).nonzero().squeeze(1);
      const int64_t n_now = positions.size(0);
      auto gather = [](const Tensor& s, const Tensor& i) { return s.index_select(0, i); };

      std::vector<Tensor> pos{positions}, ls{log_scales}, qs{quats}, cl{color_logits},
          op{opacity_logits}, bd{binding};
      auto keep = torch::ones({n_now}, at::kBool).to(device);
      if (clone_idx.numel() > 0) {
        pos.push_back(gather(positions, clone_idx));
        ls.push_back(gather(log_scales, clone_idx));
        qs.push_back(gather(quats, clone_idx));
        cl.push_back(gather(color_logits, clone_idx));
        op.push_back(gather(opacity_logits, clone_idx));
        bd.push_back(gather(binding, clone_idx));
      }
      if (split_idx.numel() > 0) {
        keep.index_put_({split_idx}, false);  // parents replaced by 2 children each
        for (int child = 0; child < 2; ++child) {
          const auto ps = gather(scales, split_idx);
          pos.push_back(gather(positions, split_idx) + torch::randn_like(ps) * ps);
          ls.push_back((gather(scales, split_idx) / 1.6).clamp_min(smin).log());
          qs.push_back(gather(quats, split_idx));
          cl.push_back(gather(color_logits, split_idx));
          op.push_back(gather(opacity_logits, split_idx));
          bd.push_back(gather(binding, split_idx));
        }
      }
      auto pos_n = torch::cat(pos, 0);
      auto ls_n = torch::cat(ls, 0);
      auto qs_n = torch::cat(qs, 0);
      auto cl_n = torch::cat(cl, 0);
      auto op_n = torch::cat(op, 0);
      auto bd_n = torch::cat(bd, 0);
      const auto appended = torch::ones({pos_n.size(0) - n_now}, at::kBool).to(device);
      const auto op_keep = torch::sigmoid(op_n).squeeze(1) > cfg.prune_opacity;
      const auto final_keep = (torch::cat({keep, appended}, 0) & op_keep).nonzero().squeeze(1);

      positions = pos_n.index_select(0, final_keep).set_requires_grad(lr_pos > 0);
      log_scales = ls_n.index_select(0, final_keep).set_requires_grad(true);
      quats = qs_n.index_select(0, final_keep).set_requires_grad(true);
      color_logits = cl_n.index_select(0, final_keep).set_requires_grad(true);
      opacity_logits = op_n.index_select(0, final_keep).set_requires_grad(true);
      binding = bd_n.index_select(0, final_keep);
      grad_accum = torch::zeros({positions.size(0)}, opts);
      accum_count = 0;
      optimizer = make_opt();
    }

    if (rec != nullptr && (it % cfg.log_every == 0 || it == cfg.iterations - 1)) {
      rec->log_scalar("avatar", "loss", loss.item<double>());
      rec->log_scalar("avatar", "gaussians", static_cast<double>(positions.size(0)));
      NCG_LOG_INFO("avatar it={} frame={} N={} loss={:.5f}", it, f, positions.size(0),
                   loss.item<double>());
      if (cfg.dump_every > 0 && it % cfg.dump_every == 0) {
        const auto im = runtime::render_soft_aniso(deform_avatar(canonical(), transforms[0], binding),
                                                   frames[0].camera)
                            .image.detach();
        rec->log_image("avatar", "frame0", im);
      }
    }
  }

  torch::NoGradGuard ng;
  auto g = canonical();
  g.positions = torch::nan_to_num(g.positions.detach());
  g.scales = torch::nan_to_num(g.scales.detach());
  g.rotations = torch::nan_to_num(g.rotations.detach());
  g.rotations = g.rotations / g.rotations.norm(2, 1, true).clamp_min(1e-8);  // re-unit quaternions
  g.opacities = torch::nan_to_num(g.opacities.detach());
  g.colors = torch::nan_to_num(g.colors.detach());
  g.validate();
  return {g, binding.detach()};
}

}  // namespace ncg::fit
