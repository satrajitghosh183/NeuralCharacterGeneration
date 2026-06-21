#include <ncg/nerf/nerf.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/record/metrics.hpp>

#include <cmath>

namespace ncg::nerf {

TinyNerf::TinyNerf(const NerfConfig& cfg) : cfg_(cfg) {
  enc_dim_ = 3 + 3 * 2 * cfg.num_freqs;

  torch::nn::Sequential trunk;
  int in = enc_dim_;
  for (int i = 0; i < cfg.hidden_layers; ++i) {
    trunk->push_back(torch::nn::Linear(in, cfg.hidden));
    trunk->push_back(torch::nn::ReLU());
    in = cfg.hidden;
  }
  trunk_ = register_module("trunk", trunk);
  sigma_head_ = register_module("sigma_head", torch::nn::Linear(cfg.hidden, 1));
  rgb_head_ = register_module("rgb_head", torch::nn::Linear(cfg.hidden, 3));
}

Tensor TinyNerf::encode(const Tensor& x) const {
  std::vector<Tensor> outs;
  outs.push_back(x);
  for (int i = 0; i < cfg_.num_freqs; ++i) {
    const double f = std::pow(2.0, i) * M_PI;
    outs.push_back(torch::sin(x * f));
    outs.push_back(torch::cos(x * f));
  }
  return torch::cat(outs, -1);
}

std::pair<Tensor, Tensor> TinyNerf::forward(const Tensor& points) {
  const auto h = trunk_->forward(encode(points));
  const auto sigma = torch::softplus(sigma_head_->forward(h));  // [...,1] >= 0
  const auto rgb = torch::sigmoid(rgb_head_->forward(h));       // [...,3] in (0,1)
  return {sigma, rgb};
}

std::pair<Tensor, Tensor> camera_rays(const runtime::Camera& cam) {
  const auto opts = cam.R.options();
  const auto Rt = cam.R.t();                       // camera->world rotation
  const auto eye = -torch::matmul(Rt, cam.t);      // camera centre in world [3]
  const int H = cam.height;
  const int W = cam.width;

  const auto u = (torch::arange(W, opts) + 0.5).view({1, W}).expand({H, W});
  const auto v = (torch::arange(H, opts) + 0.5).view({H, 1}).expand({H, W});
  const auto dx = (u - cam.cx) / cam.fx;
  const auto dy = (v - cam.cy) / cam.fy;
  const auto dz = torch::ones({H, W}, opts);
  const auto d_cam = torch::stack({dx, dy, dz}, -1);                 // [H,W,3]
  auto d_world = torch::einsum("ij,hwj->hwi", {Rt, d_cam});          // [H,W,3]
  d_world = d_world / d_world.norm(2, -1, /*keepdim=*/true).clamp_min(1e-8);

  auto dirs = d_world.reshape({-1, 3});
  auto origins = eye.view({1, 3}).expand({H * W, 3});
  return {origins.contiguous(), dirs.contiguous()};
}

runtime::RenderOutput render_volume(TinyNerf& nerf, const runtime::Camera& cam,
                                    std::array<float, 3> background) {
  const auto& cfg = nerf.config();
  const auto opts = cam.R.options();
  const int H = cam.height;
  const int W = cam.width;
  const int64_t P = static_cast<int64_t>(H) * W;
  const int S = cfg.samples;

  auto [origins, dirs] = camera_rays(cam);  // [P,3]

  const auto tvals = torch::linspace(cfg.near, cfg.far, S, opts);              // [S]
  const auto pts = origins.unsqueeze(1) + dirs.unsqueeze(1) * tvals.view({1, S, 1});  // [P,S,3]

  auto [sigma, rgb] = nerf.forward(pts.reshape({-1, 3}));
  sigma = sigma.reshape({P, S});       // [P,S]
  rgb = rgb.reshape({P, S, 3});        // [P,S,3]

  // Distances between samples (last one large to capture the far tail).
  auto deltas = tvals.narrow(0, 1, S - 1) - tvals.narrow(0, 0, S - 1);  // [S-1]
  deltas = torch::cat({deltas, torch::full({1}, 1e10, opts)}, 0).view({1, S});  // [1,S]

  const auto alpha = 1.0 - torch::exp(-sigma * deltas);                  // [P,S]
  const auto one_minus = 1.0 - alpha + 1e-10;
  const auto cp = torch::cumprod(one_minus, /*dim=*/1);                  // inclusive
  const auto trans = torch::cat({torch::ones({P, 1}, opts), cp.narrow(1, 0, S - 1)}, 1);  // [P,S]
  const auto weights = alpha * trans;                                    // [P,S]

  const auto color = (weights.unsqueeze(-1) * rgb).sum(1);   // [P,3]
  const auto acc = weights.sum(1, /*keepdim=*/true);         // [P,1]
  const auto bg = torch::tensor({background[0], background[1], background[2]}, opts).view({1, 3});
  const auto comp = color + (1.0 - acc) * bg;                // [P,3]

  runtime::RenderOutput out;
  out.image = comp.reshape({H, W, 3}).permute({2, 0, 1}).contiguous();  // [3,H,W]
  out.alpha = acc.reshape({H, W}).unsqueeze(0).contiguous();            // [1,H,W]
  return out;
}

runtime::RenderOutput composite_over(const runtime::RenderOutput& front,
                                     const runtime::RenderOutput& back) {
  NCG_CHECK(front.image.sizes() == back.image.sizes(), "composite_over: image size mismatch");
  const auto fa = front.alpha;  // [1,H,W]
  runtime::RenderOutput out;
  out.image = front.image + (1.0 - fa) * back.image;
  out.alpha = front.alpha + (1.0 - fa) * back.alpha;
  return out;
}

std::shared_ptr<TinyNerf> fit_nerf_to_views(const std::vector<Tensor>& targets_in,
                                            const std::vector<runtime::Camera>& cameras,
                                            const NerfConfig& nc, const NerfFitConfig& fc,
                                            record::Recorder* rec) {
  NCG_CHECK(!targets_in.empty(), "fit_nerf: no targets");
  NCG_CHECK(targets_in.size() == cameras.size(), "fit_nerf: targets/cameras mismatch");
  const auto device = cameras.front().R.device();

  std::vector<Tensor> targets;
  for (const auto& t : targets_in) targets.push_back(t.to(device, at::kFloat));

  auto nerf = std::make_shared<TinyNerf>(nc);
  nerf->to(device);
  torch::optim::Adam opt(nerf->parameters(), torch::optim::AdamOptions(fc.lr));

  for (int it = 0; it < fc.iterations; ++it) {
    opt.zero_grad();
    Tensor loss = torch::zeros({}, targets.front().options());
    double psnr_sum = 0.0;
    for (size_t v = 0; v < cameras.size(); ++v) {
      const auto out = render_volume(*nerf, cameras[v]);
      loss = loss + torch::mse_loss(out.image, targets[v]);
      if (rec != nullptr) psnr_sum += record::psnr(out.image.detach(), targets[v]);
    }
    loss.backward();
    opt.step();

    if (rec != nullptr && (it % fc.log_every == 0 || it == fc.iterations - 1)) {
      rec->log_scalar("nerf", "loss", loss.item<double>());
      rec->log_scalar("nerf", "psnr", psnr_sum / static_cast<double>(cameras.size()));
      NCG_LOG_INFO("nerf it={} loss={:.5f}", it, loss.item<double>());
    }
  }
  return nerf;
}

}  // namespace ncg::nerf
