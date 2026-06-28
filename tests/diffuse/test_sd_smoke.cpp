#include <catch2/catch_test_macros.hpp>

#include <ncg/core/tensor.hpp>
#include <ncg/diffuse/scheduler.hpp>
#include <ncg/diffuse/sd_guidance.hpp>
#include <ncg/diffuse/sds.hpp>

#include <torch/torch.h>

#include <cstdlib>
#include <string>

// Validates the C++ <-> TorchScript contract for the ported SD UNet+VAE (the SDS prior). Gated on
// NCG_SD_DIR pointing at a dir with sd_unet.ts / sd_vae.ts / sd_cond.safetensors (produced by
// tools/export_sd.py); skips otherwise so the suite stays green on boxes without the weights.

TEST_CASE("sd_guidance: encode -> predict -> decode -> sds_loss runs and is finite", "[cuda;diffuse]") {
  const char* dir = std::getenv("NCG_SD_DIR");
  if (dir == nullptr) {
    SKIP("NCG_SD_DIR not set (SD TorchScript assets absent)");
  }
  if (!ncg::cuda_available()) {
    SKIP("SD guidance smoke test needs a CUDA device");
  }
  const auto device = at::Device(at::kCUDA, 0);
  const std::string d = dir;
  auto sd = ncg::diffuse::SdGuidance::load(d + "/sd_unet.ts", d + "/sd_vae.ts",
                                           d + "/sd_cond.safetensors", device);

  // A render-sized RGB image in [0,1].
  auto rgb = torch::rand({1, 3, 512, 512}, at::TensorOptions().device(device)).requires_grad_(true);
  const auto latent = sd.encode_image(rgb);
  REQUIRE(latent.dim() == 4);
  REQUIRE(latent.size(1) == 4);
  REQUIRE(latent.size(2) == 64);  // 512/8
  REQUIRE(torch::isfinite(latent).all().item<bool>());

  // Round-trip through the VAE decoder.
  const auto recon = sd.decode_latent(latent.detach());
  REQUIRE(recon.sizes() == rgb.sizes());
  REQUIRE(recon.min().item<float>() >= 0.0F);
  REQUIRE(recon.max().item<float>() <= 1.0F);

  // The denoiser predicts noise of the latent's shape.
  const auto pred = sd.predictor();
  const auto t = torch::full({1}, 500, at::TensorOptions().dtype(at::kLong).device(device));
  const auto eps = pred(latent.detach(), t);
  REQUIRE(eps.sizes() == latent.sizes());
  REQUIRE(torch::isfinite(eps).all().item<bool>());

  // SDS-as-loss through the (differentiable) VAE encoder back to the image — the completion signal.
  ncg::diffuse::DdpmSchedule sch;
  ncg::diffuse::SdsConfig scfg;
  const auto r = ncg::diffuse::sds_loss(latent, sch, pred, scfg);
  r.loss.backward();
  REQUIRE(rgb.grad().defined());
  REQUIRE(torch::isfinite(rgb.grad()).all().item<bool>());
  INFO("SDS grad_norm = " << r.grad_norm);
  REQUIRE(r.grad_norm > 0.0);
}
