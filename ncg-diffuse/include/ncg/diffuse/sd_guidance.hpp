#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/diffuse/sds.hpp>

#include <memory>
#include <string>

namespace ncg::diffuse {

// ============================================================================================
// The ported diffusion denoiser behind the SDS prior (docs/method.md §M10). Follows the PROVEN NLF
// precedent: load the released Stable-Diffusion UNet + VAE as TorchScript (`torch::jit::load`) —
// native C++, NO Python at runtime — rather than dlopen a Python extension. The graph is the real SD
// graph, so forward parity holds by construction (no per-layer reimplementation to drift).
//
// Latent-space SDS (DreamFusion/threestudio convention): the avatar's differentiable RGB render is
// VAE-encoded to a 4-channel latent, SDS runs there, and the completion gradient flows back through
// the (frozen) VAE encoder into the render → renderer params. Text conditioning is PRECOMPUTED on the
// Mac side (CLIP tokenizer + text encoder, tools/export_sd.py) and shipped as a tiny safetensors of
// {cond, uncond} embeddings — so the C++ runtime needs no tokenizer.
//
// Assets (produced by tools/export_sd.py, staged on the H100):
//   sd_unet.ts   forward(latent[B,4,h,w], t[B] long, ctx[B,77,C]) -> eps[B,4,h,w]
//   sd_vae.ts    encode(rgb[B,3,H,W] in [0,1]) -> latent[B,4,h,w];  decode(latent) -> rgb[B,3,H,W]
//   sd_cond.safetensors  { "cond":[1,77,C], "uncond":[1,77,C] }  (one prompt + the empty prompt)
// ============================================================================================

struct SdGuidanceConfig {
  float guidance = 100.0F;     // classifier-free guidance scale (SDS uses large g)
  float vae_scale = 0.18215F;  // SD latent scaling: latent = vae_scale * encoder_mean
};

class SdGuidance {
 public:
  // Load the three assets onto `device`. Throws (NCG_THROW) if any is missing/incompatible.
  static SdGuidance load(const std::string& unet_ts, const std::string& vae_ts,
                         const std::string& cond_safetensors, at::Device device,
                         const SdGuidanceConfig& cfg = {});

  // RGB [B,3,H,W] in [0,1]  ->  latent [B,4,H/8,W/8] (already vae_scale-multiplied). Differentiable
  // (encoder kept in the graph) so SDS gradients reach the render.
  Tensor encode_image(const Tensor& rgb) const;
  Tensor decode_latent(const Tensor& latent) const;  // latent -> RGB [0,1]

  // A NoisePredictor closing over the stored {cond, uncond} embeddings + guidance: runs the UNet
  // twice (uncond, cond) and CFG-combines. Drop straight into sds_loss(...).
  NoisePredictor predictor() const;

  at::Device device() const;

  SdGuidance();
  SdGuidance(SdGuidance&&) noexcept;
  SdGuidance& operator=(SdGuidance&&) noexcept;
  ~SdGuidance();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ncg::diffuse
