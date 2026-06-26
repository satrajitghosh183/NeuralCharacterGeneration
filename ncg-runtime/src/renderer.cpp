#include <ncg/runtime/renderer.hpp>

#include <ncg/core/error.hpp>
#include <ncg/runtime/splat_raster.hpp>

#include <algorithm>

namespace ncg::runtime {

RenderOutput render_gaussians(const recon::GaussianCloud& g, const Camera& cam,
                              std::array<float, 3> background) {
  g.validate();
  NCG_CHECK(g.device().is_cuda(), "render_gaussians: cloud must be on CUDA");
  NCG_CHECK(cam.width > 0 && cam.height > 0, "render_gaussians: invalid image size");

  Tensor uv;
  Tensor depth;
  cam.project(g.positions, uv, depth);  // uv [N,2], depth [N]

  const auto u = uv.select(1, 0).contiguous();
  const auto v = uv.select(1, 1).contiguous();

  // Screen-space std-dev from world scale and depth; floor at half a pixel.
  const auto mean_scale = g.scales.mean(/*dim=*/1);                                  // [N]
  const auto sigma_px = (cam.fx * mean_scale / depth.clamp_min(1e-3)).clamp_min(0.5);  // [N]
  const auto inv_s2 = (1.0 / (sigma_px * sigma_px)).to(at::kFloat).contiguous();

  // Cull Gaussians at/behind the image plane by zeroing their opacity.
  const auto in_front = (depth > 0.01).to(at::kFloat);
  const auto op = (g.opacities.squeeze(1) * in_front).contiguous();

  // Composite near -> far.
  const auto order = depth.argsort(/*dim=*/0, /*descending=*/false);
  auto sel = [&](const Tensor& x) { return x.index_select(0, order).contiguous(); };

  auto [image, alpha] = splat_render_cuda(sel(u), sel(v), sel(inv_s2), sel(op),
                                          sel(g.colors), cam.height, cam.width, background);
  return {image, alpha};
}

RenderOutput render_soft(const recon::GaussianCloud& g, const Camera& cam,
                         std::array<float, 3> background, int64_t chunk) {
  g.validate();
  NCG_CHECK(cam.width > 0 && cam.height > 0, "render_soft: invalid image size");
  const int H = cam.height;
  const int W = cam.width;
  const auto opts = g.positions.options();

  Tensor uv;
  Tensor depth;
  cam.project(g.positions, uv, depth);  // differentiable

  const auto mean_scale = g.scales.mean(/*dim=*/1);                                  // [N]
  const auto sigma_px = (cam.fx * mean_scale / depth.clamp_min(1e-3)).clamp_min(0.5);  // [N]
  const auto inv_s2 = 1.0 / (sigma_px * sigma_px);                                     // [N]
  const auto in_front = (depth > 0.01).to(at::kFloat).detach();                        // mask
  const auto op = g.opacities.squeeze(1) * in_front;                                   // [N]

  const auto xs = (torch::arange(W, opts) + 0.5).view({1, 1, W});  // [1,1,W]
  const auto ys = (torch::arange(H, opts) + 0.5).view({1, H, 1});  // [1,H,1]

  auto wsum = torch::zeros({H, W}, opts);
  auto csum = torch::zeros({3, H, W}, opts);

  const int64_t n = g.positions.size(0);
  for (int64_t s = 0; s < n; s += chunk) {
    const int64_t e = std::min(s + chunk, n);
    using torch::indexing::Slice;
    const auto cu = uv.index({Slice(s, e), 0}).view({-1, 1, 1});       // [C,1,1]
    const auto cv = uv.index({Slice(s, e), 1}).view({-1, 1, 1});       // [C,1,1]
    const auto dx = xs - cu;                                          // [C,1,W]
    const auto dy = ys - cv;                                          // [C,H,1]
    const auto d2 = dx * dx + dy * dy;                                // [C,H,W]
    const auto w = op.index({Slice(s, e)}).view({-1, 1, 1}) *
                   torch::exp(-0.5 * d2 * inv_s2.index({Slice(s, e)}).view({-1, 1, 1}));  // [C,H,W]
    wsum = wsum + w.sum(0);
    csum = csum + torch::einsum("chw,ck->khw", {w, g.colors.index({Slice(s, e)})});  // [3,H,W]
  }

  const auto coverage = (1.0 - torch::exp(-wsum)).unsqueeze(0);  // [1,H,W] soft alpha
  const auto fg = csum / (wsum.unsqueeze(0) + 1e-8);
  const auto bg = torch::tensor({background[0], background[1], background[2]}, opts).view({3, 1, 1});
  RenderOutput out;
  out.image = fg * coverage + bg * (1.0 - coverage);
  out.alpha = coverage;
  return out;
}

namespace {
// Unit quaternion (w,x,y,z) [N,4] -> rotation matrices [N,3,3]. Differentiable; normalizes q so
// gradients to the raw (unconstrained) quaternion leaves stay well-defined.
Tensor quat_to_rotmat(const Tensor& q_in) {
  const auto q = q_in / q_in.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-8);
  const auto w = q.select(1, 0);
  const auto x = q.select(1, 1);
  const auto y = q.select(1, 2);
  const auto z = q.select(1, 3);
  const auto r00 = 1 - 2 * (y * y + z * z);
  const auto r01 = 2 * (x * y - w * z);
  const auto r02 = 2 * (x * z + w * y);
  const auto r10 = 2 * (x * y + w * z);
  const auto r11 = 1 - 2 * (x * x + z * z);
  const auto r12 = 2 * (y * z - w * x);
  const auto r20 = 2 * (x * z - w * y);
  const auto r21 = 2 * (y * z + w * x);
  const auto r22 = 1 - 2 * (x * x + y * y);
  return torch::stack({torch::stack({r00, r01, r02}, 1), torch::stack({r10, r11, r12}, 1),
                       torch::stack({r20, r21, r22}, 1)},
                      1);  // [N,3,3]
}
}  // namespace

RenderOutput render_soft_aniso(const recon::GaussianCloud& g, const Camera& cam,
                               std::array<float, 3> background, int64_t chunk, float dilation) {
  g.validate();
  NCG_CHECK(cam.width > 0 && cam.height > 0, "render_soft_aniso: invalid image size");
  const int H = cam.height;
  const int W = cam.width;
  const auto opts = g.positions.options();
  const int64_t n = g.positions.size(0);

  // Camera-space mean + depth (differentiable; mirrors Camera::project).
  const auto Rv = cam.R.to(opts);                                            // [3,3] world->cam
  const auto tv = cam.t.to(opts).view({1, 3});                              // [1,3]
  const auto xc = torch::matmul(g.positions, Rv.t()) + tv;                  // [N,3]
  const auto xcx = xc.select(1, 0);
  const auto xcy = xc.select(1, 1);
  const auto z = xc.select(1, 2);
  const auto zc = z.clamp_min(1e-3);
  const auto u = cam.fx * xcx / zc + cam.cx;  // [N]
  const auto v = cam.fy * xcy / zc + cam.cy;  // [N]

  // 3D world covariance Σ = R diag(s²) Rᵀ, built as M Mᵀ with M = R · diag(s).
  const auto Rq = quat_to_rotmat(g.rotations);                              // [N,3,3]
  const auto M = Rq * g.scales.unsqueeze(1);                                // [N,3,3] cols scaled
  const auto Sigma = torch::matmul(M, M.transpose(1, 2));                   // [N,3,3]
  // Camera-space covariance Σ_c = Rv Σ Rvᵀ (Rv broadcasts over N).
  const auto Sigma_c = torch::matmul(Rv, torch::matmul(Sigma, Rv.t()));     // [N,3,3]

  // Perspective Jacobian J (2x3) of (u,v) wrt camera-space point, per Gaussian.
  const auto z2 = zc * zc;
  const auto zeros = torch::zeros({n}, opts);
  const auto j00 = cam.fx / zc;
  const auto j02 = -cam.fx * xcx / z2;
  const auto j11 = cam.fy / zc;
  const auto j12 = -cam.fy * xcy / z2;
  const auto J = torch::stack({torch::stack({j00, zeros, j02}, 1),
                               torch::stack({zeros, j11, j12}, 1)},
                              1);  // [N,2,3]
  // 2D screen covariance Σ' = J Σ_c Jᵀ + dilation·I  (low-pass so sub-pixel splats stay finite).
  auto cov2d = torch::matmul(J, torch::matmul(Sigma_c, J.transpose(1, 2)));  // [N,2,2]
  const auto dil = torch::eye(2, opts) * dilation;
  cov2d = cov2d + dil.unsqueeze(0);
  // Closed-form 2x2 inverse -> conic (a, b, c) with [[a,b],[b,c]].
  const auto a = cov2d.select(1, 0).select(1, 0);
  const auto b = cov2d.select(1, 0).select(1, 1);
  const auto c = cov2d.select(1, 1).select(1, 1);
  const auto det = (a * c - b * b).clamp_min(1e-9);
  const auto con_a = c / det;        // [N]
  const auto con_b = -b / det;       // [N]
  const auto con_c = a / det;        // [N]

  const auto in_front = (z > 0.01).to(at::kFloat).detach();  // depth cull mask
  const auto op = g.opacities.squeeze(1) * in_front;         // [N]

  const auto xs = (torch::arange(W, opts) + 0.5).view({1, 1, W});  // [1,1,W]
  const auto ys = (torch::arange(H, opts) + 0.5).view({1, H, 1});  // [1,H,1]

  auto wsum = torch::zeros({H, W}, opts);
  auto csum = torch::zeros({3, H, W}, opts);
  for (int64_t s = 0; s < n; s += chunk) {
    const int64_t e = std::min(s + chunk, n);
    using torch::indexing::Slice;
    const auto cu = u.index({Slice(s, e)}).view({-1, 1, 1});  // [C,1,1]
    const auto cv = v.index({Slice(s, e)}).view({-1, 1, 1});
    const auto dx = xs - cu;  // [C,1,W]
    const auto dy = ys - cv;  // [C,H,1]
    const auto ca = con_a.index({Slice(s, e)}).view({-1, 1, 1});
    const auto cb = con_b.index({Slice(s, e)}).view({-1, 1, 1});
    const auto cc = con_c.index({Slice(s, e)}).view({-1, 1, 1});
    // Mahalanobis power. For a positive-definite conic the quadratic form is ≥0 so power ≤0; clamp
    // to [-30,0] so float error can't make exp(power) overflow to inf (which yields inf gradients).
    const auto power =
        (-0.5 * (ca * dx * dx + 2 * cb * dx * dy + cc * dy * dy)).clamp(-30.0, 0.0);
    const auto w = op.index({Slice(s, e)}).view({-1, 1, 1}) * torch::exp(power);  // [C,H,W]
    wsum = wsum + w.sum(0);
    csum = csum + torch::einsum("chw,ck->khw", {w, g.colors.index({Slice(s, e)})});
  }

  const auto coverage = (1.0 - torch::exp(-wsum)).unsqueeze(0);  // [1,H,W]
  const auto fg = csum / (wsum.unsqueeze(0) + 1e-8);
  const auto bg = torch::tensor({background[0], background[1], background[2]}, opts).view({3, 1, 1});
  RenderOutput out;
  // Sanitize: a few near-degenerate Gaussians can leave isolated non-finite pixels that would
  // otherwise poison metrics/exports. nan_to_num is differentiable (passes finite grads through).
  out.image = torch::nan_to_num(fg * coverage + bg * (1.0 - coverage));
  out.alpha = torch::nan_to_num(coverage);
  return out;
}

}  // namespace ncg::runtime
