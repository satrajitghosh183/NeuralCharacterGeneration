#include <ncg/diffuse/sd_guidance.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/io/safetensors.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace ncg::diffuse {

struct SdGuidance::Impl {
  torch::jit::script::Module unet;
  torch::jit::script::Module vae;
  Tensor cond;    // [1,77,C]
  Tensor uncond;  // [1,77,C]
  // SDXL + IP-Adapter (FaceID) mode: present when the cond safetensors carries pooled/ip embeds.
  bool sdxl_ip = false;
  Tensor cond_pooled, uncond_pooled;  // [1,1280]
  Tensor ip_cond, ip_uncond;          // [1,K,512] subject identity embeds / zeros
  at::Device device = at::kCPU;
  SdGuidanceConfig cfg;
};

SdGuidance::SdGuidance() = default;
SdGuidance::SdGuidance(SdGuidance&&) noexcept = default;
SdGuidance& SdGuidance::operator=(SdGuidance&&) noexcept = default;
SdGuidance::~SdGuidance() = default;

namespace {
void require_file(const std::string& p, const char* what) {
  if (!std::filesystem::exists(p)) NCG_THROW("SdGuidance::load: missing {} '{}'", what, p);
}
}  // namespace

SdGuidance SdGuidance::load(const std::string& unet_ts, const std::string& vae_ts,
                            const std::string& cond_safetensors, at::Device device,
                            const SdGuidanceConfig& cfg) {
  require_file(unet_ts, "UNet TorchScript");
  require_file(vae_ts, "VAE TorchScript");
  require_file(cond_safetensors, "conditioning safetensors");

  SdGuidance g;
  g.impl_ = std::make_unique<Impl>();
  g.impl_->device = device;
  g.impl_->cfg = cfg;
  try {
    if (device.is_mps()) {
      // MPS has no float64. The diffusers trace carries a few f64 buffers, so deserializing
      // straight onto MPS throws. Load on CPU, cast everything to f32, THEN move to Metal.
      g.impl_->unet = torch::jit::load(unet_ts, at::Device(at::kCPU));
      g.impl_->vae = torch::jit::load(vae_ts, at::Device(at::kCPU));
      g.impl_->unet.to(at::kFloat);
      g.impl_->vae.to(at::kFloat);
      g.impl_->unet.to(device);
      g.impl_->vae.to(device);
    } else {
      g.impl_->unet = torch::jit::load(unet_ts, device);
      g.impl_->vae = torch::jit::load(vae_ts, device);
    }
  } catch (const std::exception& e) {
    NCG_THROW("SdGuidance::load: torch::jit::load failed: {}", e.what());
  }
  g.impl_->unet.eval();
  g.impl_->vae.eval();

  // Precomputed CLIP text embeddings (a safetensors with "cond" and "uncond").
  auto st = ncg::io::SafeTensors::open(cond_safetensors);
  g.impl_->cond = st.view("cond").clone().to(device, at::kFloat);
  g.impl_->uncond = st.view("uncond").clone().to(device, at::kFloat);
  if (st.has("cond_pooled") && st.has("ip_cond")) {  // SDXL + FaceID export
    g.impl_->sdxl_ip = true;
    g.impl_->cond_pooled = st.view("cond_pooled").clone().to(device, at::kFloat);
    g.impl_->uncond_pooled = st.view("uncond_pooled").clone().to(device, at::kFloat);
    g.impl_->ip_cond = st.view("ip_cond").clone().to(device, at::kFloat);
    g.impl_->ip_uncond = st.view("ip_uncond").clone().to(device, at::kFloat);
    NCG_LOG_INFO("SdGuidance: SDXL+FaceID mode ({} identity embeds)", g.impl_->ip_cond.size(1));
  }
  NCG_CHECK(g.impl_->cond.dim() == 3 && g.impl_->uncond.dim() == 3,
            "SdGuidance: cond/uncond must be [1,77,C]");
  NCG_LOG_INFO("SdGuidance: loaded UNet+VAE; ctx dim {}", g.impl_->cond.size(2));
  return g;
}

at::Device SdGuidance::device() const { return impl_->device; }

Tensor SdGuidance::encode_image(const Tensor& rgb) const {
  NCG_CHECK(impl_, "SdGuidance: not loaded");
  const auto x = rgb.to(impl_->device, at::kFloat);
  // VAE encode returns the posterior mean already; scale to the UNet's latent convention.
  auto module = impl_->vae;  // jit modules: method calls are non-const
  const auto lat = module.run_method("encode", x).toTensor();
  return impl_->cfg.vae_scale * lat;
}

Tensor SdGuidance::decode_latent(const Tensor& latent) const {
  NCG_CHECK(impl_, "SdGuidance: not loaded");
  auto module = impl_->vae;
  const auto img = module.run_method("decode", latent / impl_->cfg.vae_scale).toTensor();
  return img.clamp(0.0, 1.0);
}

Tensor SdGuidance::img2img(const Tensor& init_rgb, float strength, int steps,
                           const DdpmSchedule& schedule) const {
  NCG_CHECK(impl_, "SdGuidance: not loaded");
  NCG_CHECK(steps >= 1, "img2img: steps must be >= 1");
  torch::NoGradGuard ng;
  const int T = schedule.num_timesteps();
  const int t_start = std::min(T - 1, std::max(1, static_cast<int>(strength * (T - 1))));
  const auto abar = schedule.alphas_cumprod().to(impl_->device);  // [T]
  const auto pred = predictor();

  auto z0 = encode_image(init_rgb);                               // [B,4,h,w]
  const auto opts_l = at::TensorOptions().dtype(at::kLong).device(impl_->device);

  // Descending DDIM timestep schedule t_start -> 0.
  std::vector<int64_t> ts;
  ts.reserve(static_cast<size_t>(steps) + 1);
  for (int i = 0; i < steps; ++i)
    ts.push_back(static_cast<int64_t>(std::llround(t_start * (1.0 - static_cast<double>(i) / steps))));
  ts.push_back(0);

  // Start from z0 noised to t_start (SDEdit).
  auto z = schedule.add_noise(z0, torch::randn_like(z0),
                              torch::full({z0.size(0)}, ts.front(), opts_l));
  for (size_t i = 0; i + 1 < ts.size(); ++i) {
    const auto t = torch::full({z0.size(0)}, ts[i], opts_l);
    const auto eps = pred(z, t);                                  // CFG noise estimate
    const double at = abar[ts[i]].item<double>();
    const double an = abar[ts[i + 1]].item<double>();
    // predicted x0, then deterministic DDIM step to the next timestep (eta = 0).
    auto z0p = (z - std::sqrt(1.0 - at) * eps) / std::sqrt(at);
    z0p = z0p.clamp(-4.0, 4.0);
    z = std::sqrt(an) * z0p + std::sqrt(1.0 - an) * eps;
  }
  return decode_latent(z);
}

Tensor SdGuidance::img2img_control(const Tensor& init_rgb, const Tensor& control_rgb, float strength,
                                   int steps, const DdpmSchedule& schedule) const {
  NCG_CHECK(impl_, "SdGuidance: not loaded");
  NCG_CHECK(steps >= 1, "img2img_control: steps must be >= 1");
  torch::NoGradGuard ng;
  Impl* p = impl_.get();
  const float guidance = impl_->cfg.guidance;
  const auto ctrl = control_rgb.to(p->device, at::kFloat);
  // ControlNet+UNet predictor: unet(z, t, ctx, control) with classifier-free guidance.
  const NoisePredictor pred = [p, guidance, &ctrl](const Tensor& z, const Tensor& t) -> Tensor {
    const int64_t B = z.size(0);
    auto unet = p->unet;
    const auto tl = t.to(p->device, at::kLong);
    const auto cu = p->uncond.expand({B, p->uncond.size(1), p->uncond.size(2)});
    const auto cc = p->cond.expand({B, p->cond.size(1), p->cond.size(2)});
    const auto eps_u = unet.forward({z, tl, cu, ctrl}).toTensor();
    const auto eps_c = unet.forward({z, tl, cc, ctrl}).toTensor();
    return cfg_eps(eps_u, eps_c, guidance);
  };

  const int T = schedule.num_timesteps();
  const int t_start = std::min(T - 1, std::max(1, static_cast<int>(strength * (T - 1))));
  const auto abar = schedule.alphas_cumprod().to(p->device);
  const auto opts_l = at::TensorOptions().dtype(at::kLong).device(p->device);
  auto z0 = encode_image(init_rgb);
  std::vector<int64_t> ts;
  for (int i = 0; i < steps; ++i)
    ts.push_back(static_cast<int64_t>(std::llround(t_start * (1.0 - static_cast<double>(i) / steps))));
  ts.push_back(0);
  auto z = schedule.add_noise(z0, torch::randn_like(z0),
                              torch::full({z0.size(0)}, ts.front(), opts_l));
  for (size_t i = 0; i + 1 < ts.size(); ++i) {
    const auto t = torch::full({z0.size(0)}, ts[i], opts_l);
    const auto eps = pred(z, t);
    const double at = abar[ts[i]].item<double>();
    const double an = abar[ts[i + 1]].item<double>();
    auto z0p = ((z - std::sqrt(1.0 - at) * eps) / std::sqrt(at)).clamp(-4.0, 4.0);
    z = std::sqrt(an) * z0p + std::sqrt(1.0 - an) * eps;
  }
  return decode_latent(z);
}

NoisePredictor SdGuidance::predictor() const {
  NCG_CHECK(impl_, "SdGuidance: not loaded");
  Impl* p = impl_.get();
  const float guidance = impl_->cfg.guidance;
  return [p, guidance](const Tensor& x_t, const Tensor& t) -> Tensor {
    const int64_t B = x_t.size(0);
    auto unet = p->unet;
    const auto tl = t.to(p->device, at::kLong);
    const auto ctx_u = p->uncond.expand({B, p->uncond.size(1), p->uncond.size(2)});
    const auto ctx_c = p->cond.expand({B, p->cond.size(1), p->cond.size(2)});
    Tensor eps_u, eps_c;
    if (p->sdxl_ip) {
      eps_u = unet.forward({x_t, tl, ctx_u, p->uncond_pooled, p->ip_uncond}).toTensor();
      eps_c = unet.forward({x_t, tl, ctx_c, p->cond_pooled, p->ip_cond}).toTensor();
    } else {
      eps_u = unet.forward({x_t, tl, ctx_u}).toTensor();
      eps_c = unet.forward({x_t, tl, ctx_c}).toTensor();
    }
    return cfg_eps(eps_u, eps_c, guidance);
  };
}

}  // namespace ncg::diffuse
