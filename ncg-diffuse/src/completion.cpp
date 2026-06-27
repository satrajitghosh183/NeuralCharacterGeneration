#include <ncg/diffuse/completion.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

namespace ncg::diffuse {

Tensor completion_gate(const Tensor& obs, const CompletionConfig& cfg) {
  NCG_CHECK(cfg.obs_lo < cfg.obs_hi, "completion_gate: require obs_lo < obs_hi");
  // t = clamp((o - lo)/(hi - lo), 0, 1); smoothstep s = t^2(3 - 2t); gate = 1 - s.
  // s(0)=0, s(1)=1, s'(0)=s'(1)=0 → gate is C¹ and EXACTLY 1 at t<=0, EXACTLY 0 at t>=1.
  const auto t = ((obs - cfg.obs_lo) / (cfg.obs_hi - cfg.obs_lo)).clamp(0.0, 1.0);
  const auto s = t * t * (3.0 - 2.0 * t);
  return 1.0 - s;
}

namespace {
// Broadcast a per-pixel map to [B,1,H,W] so it multiplies a [B,C,H,W] gradient over channels.
Tensor as_bchw(const Tensor& m) {
  if (m.dim() == 4) return m;            // [B,1,H,W] (or [B,C,H,W])
  if (m.dim() == 3) return m.unsqueeze(1);  // [B,H,W] -> [B,1,H,W]
  NCG_THROW("completion: observability map must be [B,H,W] or [B,1,H,W]");
}
}  // namespace

Tensor apply_completion_gate(const Tensor& sds_grad, const Tensor& obs_pixel,
                             const CompletionConfig& cfg) {
  const auto g = completion_gate(as_bchw(obs_pixel), cfg).to(sds_grad.dtype());
  return sds_grad * g;  // broadcasts [B,1,H,W] over the C channels of the gradient
}

Tensor provenance_mask(const Tensor& obs_pixel, const CompletionConfig& cfg) {
  // Synthesized wherever the gate passes ANY signal (g > 0), i.e. o < obs_hi; observed otherwise.
  return (completion_gate(obs_pixel, cfg) > 0.0).to(at::kFloat);
}

}  // namespace ncg::diffuse
