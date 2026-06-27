#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::diffuse {

// ============================================================================================
// RENDER-CONSISTENT, OBSERVABILITY-GATED COMPLETION (docs/method.md §M10/§M11) — the novel half of
// Phase E. DreamFusion-style SDS hallucinates EVERYWHERE; that is wrong for a personalized avatar —
// it would overwrite the parts we actually photographed (the identity) with diffusion's prior of a
// generic person. The contribution: SDS is a COMPLETION operator, applied ONLY where the surface is
// unobserved, and IDENTICALLY ZERO where it is observed.
//
// Drive it with the SAME observability field o(v) that gates the Phase-B Δv solve (one firewall for
// geometry AND appearance). Rasterize o(v) to a per-pixel map o(p) ∈ [0,1]; the completion gate is
//     g(o) = 1                      for o <= obs_lo   (fully unobserved → full SDS)
//     g(o) = 0                      for o >= obs_hi   (observed → SDS EXACTLY zero)
//     g(o) = 1 - smoothstep(...)    in between        (C¹: g'(obs_lo)=g'(obs_hi)=0)
// smoothstep makes g C¹ across BOTH seams while still hitting EXACT 0 on Ω_obs = {o >= obs_hi}.
// That reconciles the two locked requirements: "g(o) C¹" (Phase F handoff) AND "Hard/exact-zero on
// Ω_obs" (identity protection — the photographed person is never diffusion-rewritten).
//
// THEOREM (identity protection, M11): for every pixel p with o(p) >= obs_hi, the completion gradient
// g(o(p))·grad_SDS(p) == 0 exactly, so the optimizer receives NO diffusion signal on observed
// surface; the avatar's identity there is determined solely by the photos. Proven by test (the gate
// is bit-exact zero, not merely small) — see tests/diffuse/test_completion.cpp.
// ============================================================================================

struct CompletionConfig {
  float obs_lo = 0.15F;  // o below this: unobserved → full SDS weight
  float obs_hi = 0.35F;  // o at/above this: observed → SDS exactly zero (identity-protected)
};

// g(o) ∈ [0,1], C¹, exact 1 below obs_lo and exact 0 at/above obs_hi. `obs` any shape in [0,1].
Tensor completion_gate(const Tensor& obs, const CompletionConfig& cfg = {});

// Multiply a per-pixel SDS gradient [B,C,H,W] by the gate from a per-pixel observability map
// [B,1,H,W] or [B,H,W] (broadcast over channels). Returns the render-consistent completion gradient.
Tensor apply_completion_gate(const Tensor& sds_grad, const Tensor& obs_pixel,
                             const CompletionConfig& cfg = {});

// Provenance mask (for the Phase-F handoff + export labeling): 1 where the pixel is SYNTHESIZED by
// the prior (g>0), 0 where it is PHOTO-OBSERVED (g==0). Same spatial shape as `obs_pixel`.
Tensor provenance_mask(const Tensor& obs_pixel, const CompletionConfig& cfg = {});

}  // namespace ncg::diffuse
