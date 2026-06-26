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

recon::GaussianCloud deform_avatar(const recon::GaussianCloud& canonical, const Tensor& vt) {
  NCG_CHECK(vt.dim() == 3 && vt.size(1) == 4 && vt.size(2) == 4,
            "deform_avatar: vertex_transforms must be [V,4,4]");
  NCG_CHECK(canonical.size() == vt.size(0),
            "deform_avatar: cloud size must equal vertex count (1:1 binding)");
  using torch::indexing::Slice;
  const auto Rm = vt.index({Slice(), Slice(0, 3), Slice(0, 3)});  // [V,3,3]
  const auto tm = vt.index({Slice(), Slice(0, 3), 3});            // [V,3]

  recon::GaussianCloud g;
  g.positions = torch::matmul(Rm, canonical.positions.unsqueeze(2)).squeeze(2) + tm;  // [V,3]
  g.rotations = quat_mul(rotmat_to_quat(Rm), canonical.rotations);                    // [V,4]
  g.scales = canonical.scales;
  g.opacities = canonical.opacities;
  g.colors = canonical.colors;
  return g;
}

recon::GaussianCloud fit_avatar(const body::SmplxModel& model, const Tensor& betas_in,
                                const std::vector<AvatarFrame>& frames, const Tensor& init_colors,
                                const AvatarFitConfig& cfg, record::Recorder* rec) {
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

  // Optimizable canonical leaves.
  auto positions = rest_verts.clone().set_requires_grad(cfg.lr_position > 0);
  auto log_scales =
      torch::full({V, 3}, std::log(cfg.init_scale), opts).set_requires_grad(true);
  auto quats = torch::zeros({V, 4}, opts);
  quats.select(1, 0).fill_(1.0);
  quats = quats.set_requires_grad(true);
  auto color_logits = inv_sigmoid(init_colors.to(opts)).set_requires_grad(true);
  auto opacity_logits =
      inv_sigmoid(torch::full({V, 1}, 0.9F, opts)).set_requires_grad(true);
  auto gain = torch::ones({F, 3}, opts).set_requires_grad(cfg.per_view_exposure);
  auto bias = torch::zeros({F, 3}, opts).set_requires_grad(cfg.per_view_exposure);

  // Bound scales to a human-scale range: collapse (→0) makes the projected covariance singular and
  // explodes the conic-inverse gradient; runaway growth lets one Gaussian dominate the normalized
  // splat. Both drive the fit to NaN, so clamp the rendered scale (gradient still flows in-range).
  const double smin = 1e-3;
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
  std::vector<Tensor> clip_leaves{log_scales, quats, color_logits, opacity_logits};
  if (cfg.lr_position > 0) clip_leaves.push_back(positions);
  if (cfg.per_view_exposure) {
    clip_leaves.push_back(gain);
    clip_leaves.push_back(bias);
  }

  // Per-frame body masks from the posed init render.
  std::vector<Tensor> targets(F);
  std::vector<Tensor> masks(F);
  {
    torch::NoGradGuard ng;
    const auto c0 = canonical();
    for (int64_t f = 0; f < F; ++f) {
      targets[f] = frames[f].target.to(opts);
      if (cfg.use_mask) {
        const auto a0 = runtime::render_soft_aniso(deform_avatar(c0, transforms[f]),
                                                   frames[f].camera)
                            .alpha;
        masks[f] = (a0 > 0.05).to(at::kFloat);
      }
    }
  }

  using torch::optim::Adam;
  using torch::optim::AdamOptions;
  using torch::optim::OptimizerParamGroup;
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
  if (cfg.lr_position > 0) groups.push_back(grp({positions}, cfg.lr_position));
  if (cfg.per_view_exposure) {
    groups.push_back(grp({gain}, 1e-3));
    groups.push_back(grp({bias}, 1e-3));
  }
  Adam optimizer(groups, AdamOptions(cfg.lr_color));

  for (int it = 0; it < cfg.iterations; ++it) {
    const int64_t f = torch::randint(0, F, {1}, at::kLong).item<int64_t>();
    optimizer.zero_grad();
    const auto posed = deform_avatar(canonical(), transforms[f]);
    auto pred = runtime::render_soft_aniso(posed, frames[f].camera).image;
    if (cfg.per_view_exposure) {
      pred = (pred * gain[f].view({3, 1, 1}) + bias[f].view({3, 1, 1})).clamp(0.0, 1.0);
    }
    auto tgt = targets[f];
    if (cfg.use_mask) {
      pred = pred * masks[f];
      tgt = tgt * masks[f];
    }
    const auto l1 = torch::l1_loss(pred, tgt);
    const auto loss = (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * (1.0 - ssim(pred, tgt));
    // Skip a non-finite step rather than poison Adam's moments with NaN.
    if (!std::isfinite(loss.item<double>())) {
      optimizer.zero_grad();
      continue;
    }
    loss.backward();
    torch::nn::utils::clip_grad_norm_(clip_leaves, 1.0);  // tame conic-inverse gradient spikes
    optimizer.step();

    if (rec != nullptr && (it % cfg.log_every == 0 || it == cfg.iterations - 1)) {
      rec->log_scalar("avatar", "loss", loss.item<double>());
      NCG_LOG_INFO("avatar it={} frame={} loss={:.5f}", it, f, loss.item<double>());
      if (cfg.dump_every > 0 && it % cfg.dump_every == 0) {
        const auto im = runtime::render_soft_aniso(deform_avatar(canonical(), transforms[0]),
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
  return g;
}

}  // namespace ncg::fit
