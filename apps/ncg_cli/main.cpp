// ncg_cli — pipeline driver with subcommands. Phase 1 wires the parts that exist.
//
//   ncg_cli render --smplx model.safetensors --out out.png [--width .. --height ..]
//   ncg_cli select --images a.jpg,b.jpg,c.jpg
//   ncg_cli fit    --image photo.jpg --weights nlf.safetensors   (NLF port: not implemented)
#include <args.hpp>

#include <ncg/body/nlf.hpp>
#include <ncg/body/smplx.hpp>
#include <ncg/core/device.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/fit/fit_avatar.hpp>
#include <ncg/fit/fit_image.hpp>
#include <ncg/io/image.hpp>
#include <ncg/io/npy.hpp>
#include <ncg/mesh/extract.hpp>
#include <ncg/nerf/nerf.hpp>
#include <ncg/recon/appearance.hpp>
#include <ncg/recon/init_from_body.hpp>
#include <ncg/recon/inverse_render.hpp>
#include <ncg/recon/motion_style.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/rig/rig.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>
#include <ncg/select/selector.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Axis-angle [...,3] -> glTF quaternion [...,4] (x,y,z,w).
torch::Tensor aa_to_quat(const torch::Tensor& aa) {
  const auto angle = aa.norm(2, -1, true);
  const auto axis = aa / angle.clamp_min(1e-8);
  return torch::cat({axis * torch::sin(angle * 0.5), torch::cos(angle * 0.5)}, -1);
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

int cmd_render(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "render requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  const auto out = model.forward(model.neutral_params(1));
  const auto verts = out.vertices.squeeze(0);
  auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.012F));
  cloud.to_(device);
  const auto cam = ncg::runtime::Camera::orbit(verts.mean(0), args.get_float("radius", 2.5F),
                                               args.get_float("azimuth", 20.0F),
                                               args.get_float("elevation", 10.0F), 50.0F,
                                               args.get_int("width", 512),
                                               args.get_int("height", 512), device);
  const auto render = ncg::runtime::render_gaussians(cloud, cam);
  ncg::io::save_png(args.get("out", "out.png"), render.image);
  NCG_LOG_INFO("render coverage={:.4f}", render.alpha.mean().item<double>());
  return 0;
}

// Render a turntable of the SMPL-X body and record every frame (Phase-5 eval helper).
int cmd_turntable(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "turntable requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  const int width = args.get_int("width", 512);
  const int height = args.get_int("height", 512);
  const int frames = args.get_int("frames", 36);

  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  const auto verts = model.forward(model.neutral_params(1)).vertices.squeeze(0);
  auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.012F));
  cloud.to_(device);

  auto rec = ncg::record::Recorder::create("runs", args.get("run", "turntable"));
  const auto cams = ncg::runtime::orbit_trajectory(verts.mean(0), args.get_float("radius", 2.5F),
                                                   args.get_float("elevation", 10.0F), frames,
                                                   50.0F, width, height, device);
  for (size_t i = 0; i < cams.size(); ++i) {
    const auto out = ncg::runtime::render_gaussians(cloud, cams[i]);
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%03zu", i);
    rec.log_image("turntable", name, out.image);
    rec.log_scalar("turntable", "coverage", out.alpha.mean().item<double>());
  }
  NCG_LOG_INFO("turntable {} frames -> {}", frames, rec.dir().string());
  return 0;
}

// Full asset pipeline: SMPL-X body -> Gaussians -> turntable -> marching-cubes mesh ->
// inherit SMPL-X rig (NN skinning transfer) -> export rigged OBJ + rig JSON. All recorded.
int cmd_pipeline(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "pipeline requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  const int width = args.get_int("width", 512);
  const int height = args.get_int("height", 512);
  const int frames = args.get_int("frames", 12);
  const int grid = args.get_int("res", 96);

  auto rec = ncg::record::Recorder::create("runs", args.get("run", "pipeline"));

  // 1. Body.
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  const auto body = model.forward(model.neutral_params(1));
  const auto verts = body.vertices.squeeze(0);  // [V,3]
  rec.log_scalar("body", "num_verts", static_cast<double>(model.num_verts()));

  // 2. Gaussians + turntable render.
  auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.012F));
  cloud.to_(device);
  const auto cams = ncg::runtime::orbit_trajectory(verts.mean(0), args.get_float("radius", 2.5F),
                                                   10.0F, frames, 50.0F, width, height, device);
  for (size_t i = 0; i < cams.size(); ++i) {
    const auto out = ncg::runtime::render_gaussians(cloud, cams[i]);
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%03zu", i);
    rec.log_image("turntable", name, out.image);
  }

  // 3. Mesh (marching cubes over the Gaussian density field).
  const auto mesh = ncg::mesh::extract_mesh(cloud, grid);
  rec.log_scalar("mesh", "num_verts", static_cast<double>(mesh.num_verts()));
  rec.log_scalar("mesh", "num_faces", static_cast<double>(mesh.num_faces()));
  ncg::mesh::write_obj(mesh, (rec.dir() / "mesh.obj").string());

  // 4. Inherit the SMPL-X rig: transfer skinning from body verts to the mesh, keep skeleton.
  const auto skin = ncg::rig::transfer_skinning(mesh.vertices, verts.to(at::kCPU),
                                                model.lbs_weights().to(at::kCPU));
  const auto rigged = ncg::rig::make_rigged(mesh.vertices, mesh.faces,
                                            body.joints.squeeze(0).to(at::kCPU),
                                            model.parents().to(at::kCPU), skin);
  ncg::rig::export_rigged(rigged, (rec.dir() / "avatar").string());

  NCG_LOG_INFO("pipeline done: {} mesh verts, {} faces -> {}/avatar.obj (+.rig.json)",
               mesh.num_verts(), mesh.num_faces(), rec.dir().string());
  return 0;
}

int cmd_select(const ncg::app::Args& args) {
  const auto paths = split_csv(args.require("images"));
  NCG_CHECK(!paths.empty(), "select: --images is empty");
  for (const auto& s : ncg::select::score_images(paths)) {
    std::printf("%10.2f  %s\n", s.sharpness, s.path.c_str());
  }
  NCG_LOG_INFO("best: {}", ncg::select::select_best(paths));
  return 0;
}

// image -> NLF -> SMPL-X params -> (with --smplx) posed body Gaussians -> render.
//   ncg_cli fit --image me.jpg --weights nlf_l_multi.torchscript [--smplx smplx.safetensors --out posed.png]
int cmd_fit(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "fit requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);

  ncg::body::NlfConfig nc;
  nc.detection = args.get_int("detection", 0);  // which person, if the photo has several
  auto nlf = ncg::body::Nlf::load(args.require("weights"), device, nc);
  const auto image = ncg::io::load_image(args.require("image"), 3);
  const auto pred = nlf.detect(image);
  const auto& params = pred.params;
  NCG_LOG_INFO("NLF predicted: {} joints, {} betas", params.pose_aa.size(1),
               params.betas.size(1));

  if (!args.has("smplx")) {
    NCG_LOG_INFO("fit: pass --smplx <model.safetensors> to pose + render the predicted body");
    return 0;
  }

  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  ncg::body::SmplxParams p;  // move predicted params onto the model device
  p.betas = params.betas.to(device);
  p.pose_aa = params.pose_aa.to(device);
  p.transl = params.transl.to(device);

  // NLF returns the global orientation (joint 0) in its camera frame (Y-down), which renders
  // upside-down in our Y-up world. For an avatar we want the body canonical-upright (then
  // animate), so by default we zero the root orientation, keeping NLF's estimated body pose.
  // Pass --canonical 0 to keep NLF's camera-relative orientation.
  if (args.get_int("canonical", 1) != 0) {
    p.pose_aa.select(1, 0).zero_();
  }
  // From a partial (e.g. upper-body) photo NLF must hallucinate unseen limbs, which contorts
  // the mesh and sprays the splats. --restpose renders a clean neutral A-pose using NLF's shape
  // (betas) only; appearance still maps correctly since color is keyed by vertex identity.
  if (args.get_int("restpose", 0) != 0) {
    p.pose_aa.zero_();
  }
  const auto verts = model.forward(p).vertices.squeeze(0);

  // Appearance capture: sample the photo's color at each vertex's 2D projection (NLF's
  // vertices2d). Color is keyed by vertex identity, so it is independent of the canonical
  // render pose. Requires vertices2d to index the same SMPL-X mesh we splat on.
  torch::Tensor colors;  // empty => gray default
  if (pred.vertices2d.size(0) == verts.size(0)) {
    const auto v2d = pred.vertices2d.to(device);
    colors = ncg::recon::sample_vertex_colors(image.to(device), v2d).clamp(0.0, 1.0);
    // Cull occluded / back-facing vertices (they sample background or the wrong surface) to a
    // neutral gray, so only genuinely visible vertices carry photo color.
    const auto depth = pred.vertices3d.select(1, 2).to(device);
    const auto vis = ncg::recon::vertex_visibility(v2d, depth,
                                                   static_cast<int64_t>(image.size(1)),
                                                   static_cast<int64_t>(image.size(2)));
    colors = torch::where(vis.unsqueeze(1) > 0, colors, torch::full_like(colors, 0.6F));
    NCG_LOG_INFO("appearance: {} verts, {:.0f}% visible & colored from photo", verts.size(0),
                 100.0 * vis.mean().item<double>());
  } else {
    NCG_LOG_WARN("NLF vertices2d count {} != mesh verts {} — rendering gray (no appearance)",
                 pred.vertices2d.size(0), verts.size(0));
  }

  // Per-subject 3DGS refinement: build the body in its source-photo frame (NLF vertices3d),
  // solve the camera from the 2D<->3D correspondence, then optimize the Gaussians against the
  // photo with the differentiable renderer. Sharpens the single-sample appearance.
  if (args.has("refine")) {
    const auto v3d = pred.vertices3d.to(device);
    // Downscale the (often huge) photo so the differentiable soft renderer is tractable, and
    // scale the 2D projection by the same factor so the solved camera matches.
    const auto img_full = image.to(device);
    const int maxdim = args.get_int("refine_res", 320);
    const double s = std::min(1.0, static_cast<double>(maxdim) /
                                       static_cast<double>(std::max(img_full.size(1), img_full.size(2))));
    namespace F = torch::nn::functional;
    const auto img_small =
        F::interpolate(img_full.unsqueeze(0),
                       F::InterpolateFuncOptions()
                           .scale_factor(std::vector<double>{s, s})
                           .mode(torch::kBilinear)
                           .align_corners(false))
            .squeeze(0);
    const int w = static_cast<int>(img_small.size(2));
    const int h = static_cast<int>(img_small.size(1));
    const auto v2d = pred.vertices2d.to(device) * s;
    const auto cam_s = ncg::runtime::solve_pinhole_camera(v3d, v2d, w, h);
    const auto pvs3 = ncg::recon::per_vertex_scale(v3d, args.get_float("scale_mult", 0.75F));
    auto cloud0 = ncg::recon::gaussians_on_body(v3d, args.get_float("scale", 0.012F), colors, pvs3);
    cloud0.to_(device);
    ncg::fit::RefineConfig rc;
    rc.iterations = args.get_int("refine_iters", 200);
    rc.lr = args.get_float("refine_lr", 0.01F);
    const auto refined = ncg::fit::refine_gaussians_to_image(cloud0, img_small, cam_s, rc, nullptr);
    const auto out = ncg::runtime::render_gaussians(refined, cam_s).image;
    ncg::io::save_png(args.get("out", "refined.png"), out);
    NCG_LOG_INFO("refine done -> {} ({} iters at {}x{}, photo-frame view)",
                 args.get("out", "refined.png"), rc.iterations, w, h);
    return 0;
  }

  // Adaptive per-vertex splat size (default on) so dense regions don't over-spray.
  torch::Tensor pvs;
  if (args.get_int("adaptive", 1) != 0) {
    pvs = ncg::recon::per_vertex_scale(verts, args.get_float("scale_mult", 0.75F));
  }
  auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.012F), colors, pvs);
  cloud.to_(device);
  const auto cam = ncg::runtime::Camera::orbit(verts.mean(0), args.get_float("radius", 2.5F),
                                               args.get_float("azimuth", 20.0F),
                                               args.get_float("elevation", 10.0F), 50.0F,
                                               args.get_int("width", 512),
                                               args.get_int("height", 512), device);
  const auto render = ncg::runtime::render_gaussians(cloud, cam);
  ncg::io::save_png(args.get("out", "posed.png"), render.image);
  NCG_LOG_INFO("fit done -> {} (coverage {:.4f})", args.get("out", "posed.png"),
               render.alpha.mean().item<double>());
  return 0;
}

// Cross-photo fusion (the project's core): several casual photos -> one coherent textured
// avatar. Body geometry from the first photo (canonical); per-vertex color fused across all
// photos, each weighted by per-view visibility.
//   ncg_cli fuse --images a.jpg,b.jpg,c.jpg --weights nlf.torchscript --smplx smplx.safetensors --out avatar.png
int cmd_fuse(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "fuse requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  const auto paths = split_csv(args.require("images"));
  NCG_CHECK(!paths.empty(), "fuse: --images is empty");

  auto nlf = ncg::body::Nlf::load(args.require("weights"), device);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);

  std::vector<torch::Tensor> view_colors;
  std::vector<torch::Tensor> view_weights;
  torch::Tensor ref_verts;  // canonical body geometry from the first photo
  int64_t V = 0;

  for (size_t i = 0; i < paths.size(); ++i) {
    const auto image = ncg::io::load_image(paths[i], 3);
    ncg::body::NlfPrediction pred;
    try {
      pred = nlf.detect(image);  // throws if NLF detects no person (e.g. a tight face crop)
    } catch (const std::exception& e) {
      NCG_LOG_WARN("fuse: skipping '{}' — no person detected ({})", paths[i], e.what());
      continue;
    }

    if (!ref_verts.defined()) {  // first view with a detection sets the body geometry
      ncg::body::SmplxParams p;
      p.betas = pred.params.betas.to(device);
      p.pose_aa = pred.params.pose_aa.to(device);
      p.transl = pred.params.transl.to(device);
      if (args.get_int("canonical", 1) != 0) p.pose_aa.select(1, 0).zero_();
      if (args.get_int("restpose", 0) != 0) p.pose_aa.zero_();
      ref_verts = model.forward(p).vertices.squeeze(0);  // [V,3]
      V = ref_verts.size(0);
    }
    if (pred.vertices2d.size(0) != V) {
      NCG_LOG_WARN("fuse: '{}' vertices2d count {} != mesh {} — skipping view", paths[i],
                   pred.vertices2d.size(0), V);
      continue;
    }
    const auto v2d = pred.vertices2d.to(device);
    const auto depth = pred.vertices3d.select(1, 2).to(device);  // camera-space z
    view_colors.push_back(
        ncg::recon::sample_vertex_colors(image.to(device), v2d).clamp(0.0, 1.0));
    view_weights.push_back(ncg::recon::vertex_visibility(
        v2d, depth, static_cast<int64_t>(image.size(1)), static_cast<int64_t>(image.size(2))));
    NCG_LOG_INFO("fuse: view {}/{} '{}' done", i + 1, paths.size(), paths[i]);
  }
  NCG_CHECK(!view_colors.empty(), "fuse: no usable views (vertices2d count never matched the mesh)");

  const auto fused = ncg::recon::fuse_vertex_colors(view_colors, view_weights);
  const double seen = (fused.coverage > 0).to(at::kFloat).mean().item<double>();
  NCG_LOG_INFO("fuse: {} views -> avatar; {:.1f}% of vertices seen in >=1 view", view_colors.size(),
               100.0 * seen);

  torch::Tensor pvs;
  if (args.get_int("adaptive", 1) != 0) {
    pvs = ncg::recon::per_vertex_scale(ref_verts, args.get_float("scale_mult", 0.75F));
  }
  auto cloud = ncg::recon::gaussians_on_body(ref_verts, args.get_float("scale", 0.012F),
                                             fused.colors, pvs);
  cloud.to_(device);
  const auto cam = ncg::runtime::Camera::orbit(ref_verts.mean(0), args.get_float("radius", 2.5F),
                                               args.get_float("azimuth", 20.0F),
                                               args.get_float("elevation", 10.0F), 50.0F,
                                               args.get_int("width", 512),
                                               args.get_int("height", 512), device);
  const auto render = ncg::runtime::render_gaussians(cloud, cam);
  ncg::io::save_png(args.get("out", "avatar.png"), render.image);
  NCG_LOG_INFO("fuse done -> {}", args.get("out", "avatar.png"));
  return 0;
}

// Optimize a Gaussian cloud to reproduce a target image (Phase-2 3DGS fitting), recording
// per-iteration metrics.
int cmd_fitimg(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "fitimg requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  const auto target = ncg::io::load_image(args.require("image"), 3).to(device);
  const int height = static_cast<int>(target.size(1));
  const int width = static_cast<int>(target.size(2));

  auto rec = ncg::record::Recorder::create("runs", args.get("run", "fit"));
  rec.set_config("{\"app\":\"ncg_cli fitimg\",\"image\":\"" + args.require("image") +
                 "\",\"width\":" + std::to_string(width) + ",\"height\":" +
                 std::to_string(height) + "}");
  rec.log_image("fit", "target", target);

  const auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}, target.options()),
                                               args.get_float("radius", 2.5F), 0.0F, 0.0F, 50.0F,
                                               width, height, device);
  ncg::fit::FitConfig cfg;
  cfg.iterations = args.get_int("iters", 300);
  cfg.num_gaussians = args.get_int("gaussians", 4000);
  cfg.lr = args.get_float("lr", 0.02F);

  const auto cloud = ncg::fit::fit_gaussians_to_image(target, cam, cfg, &rec);
  const auto out = ncg::runtime::render_soft(cloud, cam).image;
  ncg::io::save_png(args.get("out", "fit_out.png"), out);
  NCG_LOG_INFO("fit done -> {} | run={}", args.get("out", "fit_out.png"), rec.dir().string());
  return 0;
}

// Fit a TinyNerf (implicit volume) to a target image, then render it back. With --smplx, also
// composites the Gaussian body over the NeRF volume (hybrid path) and dumps that too.
int cmd_nerf(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "nerf requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  const auto target = ncg::io::load_image(args.require("image"), 3).to(device);
  const int height = static_cast<int>(target.size(1));
  const int width = static_cast<int>(target.size(2));

  auto rec = ncg::record::Recorder::create("runs", args.get("run", "nerf"));
  rec.log_image("nerf", "target", target);

  const auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}, target.options()),
                                               args.get_float("radius", 2.5F), 0.0F, 0.0F, 50.0F,
                                               width, height, device);
  ncg::nerf::NerfConfig nc;
  nc.samples = args.get_int("samples", 64);
  ncg::nerf::NerfFitConfig fc;
  fc.iterations = args.get_int("iters", 300);
  fc.lr = args.get_float("lr", 1e-3F);

  auto nerf = ncg::nerf::fit_nerf_to_views({target}, {cam}, nc, fc, &rec);
  const auto volume = ncg::nerf::render_volume(*nerf, cam);
  ncg::io::save_png(args.get("out", "nerf_out.png"), volume.image);
  rec.log_image("nerf", "render", volume.image);

  // Hybrid: opaque Gaussian body (front) over the learned NeRF volume (back).
  if (args.has("smplx")) {
    auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
    const auto verts = model.forward(model.neutral_params(1)).vertices.squeeze(0);
    auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.012F));
    cloud.to_(device);
    const auto front = ncg::runtime::render_gaussians(cloud, cam);
    const auto hybrid = ncg::nerf::composite_over(front, volume);
    ncg::io::save_png(args.get("hybrid_out", "nerf_hybrid.png"), hybrid.image);
    rec.log_image("nerf", "hybrid", hybrid.image);
  }

  NCG_LOG_INFO("nerf done -> {} | run={}", args.get("out", "nerf_out.png"), rec.dir().string());
  return 0;
}

// Relighting demo: render the body under an orbiting directional light to show it relight
// (uses the recovered/assigned albedo + SH shading from the inverse-rendering module).
//   ncg_cli relight --smplx smplx.safetensors [--albedo 0.78 --frames 24 --run relight]
int cmd_relight(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "relight requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  NCG_CHECK(model.has_faces(),
            "relight needs a SMPL-X model with faces — reconvert with tools/convert_smplx.py");
  const auto verts = model.forward(model.neutral_params(1)).vertices.squeeze(0);  // [V,3]

  ncg::mesh::TriMesh mesh;
  mesh.vertices = verts.to(at::kCPU);
  mesh.faces = model.faces().to(at::kCPU);
  const auto normals = ncg::mesh::compute_vertex_normals(mesh).to(device);  // [V,3]
  const auto albedo = torch::full({verts.size(0), 3}, args.get_float("albedo", 0.78F),
                                  verts.options());

  auto rec = ncg::record::Recorder::create("runs", args.get("run", "relight"));
  const int frames = args.get_int("frames", 24);
  const int width = args.get_int("width", 512);
  const int height = args.get_int("height", 512);
  const auto pvs = ncg::recon::per_vertex_scale(verts, args.get_float("scale_mult", 0.75F));
  const auto cam = ncg::runtime::Camera::orbit(verts.mean(0), args.get_float("radius", 2.5F),
                                               args.get_float("azimuth", 20.0F),
                                               args.get_float("elevation", 10.0F), 50.0F, width,
                                               height, device);
  const float el = args.get_float("light_elevation", 25.0F) * static_cast<float>(M_PI) / 180.0F;
  const auto white = torch::tensor({1.0F, 1.0F, 1.0F}, verts.options());
  for (int i = 0; i < frames; ++i) {
    const float az = 2.0F * static_cast<float>(M_PI) * static_cast<float>(i) / frames;
    const auto dir = torch::tensor(
        {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)}, verts.options());
    const auto light = ncg::recon::sh_directional_light(dir, white, args.get_float("ambient", 0.25F));
    const auto colors = ncg::recon::shade_sh(albedo, light, normals).clamp(0.0, 1.0);
    auto cloud = ncg::recon::gaussians_on_body(verts, 0.012F, colors, pvs);
    cloud.to_(device);
    const auto out = ncg::runtime::render_gaussians(cloud, cam);
    char name[32];
    std::snprintf(name, sizeof(name), "light_%03d", i);
    rec.log_image("relight", name, out.image);
  }
  NCG_LOG_INFO("relight: {} frames (orbiting light) -> {}", frames, rec.dir().string());
  return 0;
}

// Export the avatar as a binary glTF (.glb) for Unity/Unreal: SMPL-X body (neutral, or NLF
// rest-posed from a photo) + vertex normals + per-vertex color (sampled from the photo).
//   ncg_cli export --smplx smplx.safetensors [--image me.jpg --weights nlf.torchscript] --out avatar.glb
int cmd_export(const ncg::app::Args& args) {
  const bool multi = args.has("images") && args.has("weights");  // several photos -> fused albedo
  const bool want_color = !multi && args.has("image") && args.has("weights");
  const auto device = ncg::cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  NCG_CHECK(model.has_faces(), "export needs a SMPL-X model with faces — reconvert it");

  torch::Tensor verts;
  torch::Tensor colors;
  torch::Tensor joints;
  if (multi) {
    NCG_CHECK(ncg::cuda_available(), "export --images needs a CUDA device (NLF)");
    auto nlf = ncg::body::Nlf::load(args.require("weights"), device);
    const auto faces_cpu = model.faces().to(at::kCPU);
    const int64_t V = model.num_verts();
    std::vector<torch::Tensor> obs_l;
    std::vector<torch::Tensor> nrm_l;
    std::vector<torch::Tensor> w_l;
    torch::Tensor betas0;
    for (const auto& path : split_csv(args.require("images"))) {
      const auto image = ncg::io::load_image(path, 3);
      ncg::body::NlfPrediction pred;
      try {
        pred = nlf.detect(image);
      } catch (const std::exception& e) {
        NCG_LOG_WARN("export: skipping '{}' ({})", path, e.what());
        continue;
      }
      if (pred.vertices2d.size(0) != V) continue;
      if (!betas0.defined()) betas0 = pred.params.betas.to(device);
      const auto v2d = pred.vertices2d.to(device);
      ncg::mesh::TriMesh m;
      m.vertices = pred.vertices3d.to(at::kCPU);
      m.faces = faces_cpu;
      nrm_l.push_back(ncg::mesh::compute_vertex_normals(m).to(device));
      obs_l.push_back(ncg::recon::sample_vertex_colors(image.to(device), v2d).clamp(0.0, 1.0));
      w_l.push_back(ncg::recon::vertex_visibility(v2d, pred.vertices3d.select(1, 2).to(device),
                                                  static_cast<int64_t>(image.size(1)),
                                                  static_cast<int64_t>(image.size(2))));
    }
    NCG_CHECK(!obs_l.empty(), "export --images: no usable views");
    ncg::recon::InverseRenderConfig cfg;
    cfg.iterations = args.get_int("iters", 80);
    cfg.robust = (obs_l.size() >= 2);
    colors = ncg::recon::solve_inverse_render(torch::stack(obs_l, 0), torch::stack(nrm_l, 0),
                                              torch::stack(w_l, 0), cfg)
                 .albedo.clamp(0.0, 1.0);
    ncg::body::SmplxParams p;
    p.betas = betas0;
    p.pose_aa = torch::zeros({1, model.num_joints(), 3}, betas0.options());
    p.transl = torch::zeros({1, 3}, betas0.options());
    const auto body = model.forward(p);
    verts = body.vertices.squeeze(0);
    joints = body.joints.squeeze(0);
    NCG_LOG_INFO("export: fused albedo from {} photo(s)", obs_l.size());
  } else if (want_color) {
    NCG_CHECK(ncg::cuda_available(), "export with --image needs a CUDA device (NLF)");
    auto nlf = ncg::body::Nlf::load(args.require("weights"), device);
    const auto image = ncg::io::load_image(args.require("image"), 3);
    const auto pred = nlf.detect(image);
    ncg::body::SmplxParams p;
    p.betas = pred.params.betas.to(device);
    p.pose_aa = pred.params.pose_aa.to(device);
    p.transl = pred.params.transl.to(device);
    p.pose_aa.zero_();  // clean canonical rest pose
    const auto body = model.forward(p);
    verts = body.vertices.squeeze(0);
    joints = body.joints.squeeze(0);
    if (pred.vertices2d.size(0) == verts.size(0)) {
      const auto v2d = pred.vertices2d.to(device);
      colors = ncg::recon::sample_vertex_colors(image.to(device), v2d).clamp(0.0, 1.0);
      const auto depth = pred.vertices3d.select(1, 2).to(device);
      const auto vis = ncg::recon::vertex_visibility(v2d, depth, static_cast<int64_t>(image.size(1)),
                                                     static_cast<int64_t>(image.size(2)));
      colors = torch::where(vis.unsqueeze(1) > 0, colors, torch::full_like(colors, 0.6F));
    }
  } else {
    const auto body = model.forward(model.neutral_params(1));
    verts = body.vertices.squeeze(0);
    joints = body.joints.squeeze(0);
    colors = torch::full({verts.size(0), 3}, 0.75F, verts.options());
  }

  ncg::mesh::TriMesh mesh;
  mesh.vertices = verts.to(at::kCPU);
  mesh.faces = model.faces().to(at::kCPU);
  const auto normals = ncg::mesh::compute_vertex_normals(mesh);
  const auto out_path = args.get("out", "avatar.glb");
  const int64_t J = model.num_joints();

  if ((args.has("animate") || args.has("motion")) && J > 17) {
    const auto opts = verts.options();
    const float fps = args.get_float("fps", 30.0F);
    torch::Tensor motion;  // [T,J,3] axis-angle local pose per frame
    const char* kind = "idle";
    if (args.has("motion")) {
      // Drive the avatar with an extracted SMPL-X motion sequence (tools/extract_motion.py).
      motion = ncg::io::load_npy(args.require("motion")).to(opts);
      NCG_CHECK(motion.dim() == 3 && motion.size(1) == J && motion.size(2) == 3,
                "export: --motion .npy must be [T,{},3]", J);
      if (args.get_int("inplace", 1) != 0) motion.select(1, 0).zero_();  // drop global orient
      kind = "mocap";
    } else {
      // Gentle looping idle (breathing sway + head turn) so the avatar moves on import.
      const int T = 30;
      const auto s = torch::sin(torch::arange(T, opts) * (2.0 * M_PI / T));
      motion = torch::zeros({T, J, 3}, opts);
      motion.select(1, 3).select(1, 2).copy_(0.04 * s);
      motion.select(1, 6).select(1, 2).copy_(0.03 * s);
      motion.select(1, 15).select(1, 1).copy_(0.06 * s);
      motion.select(1, 16).select(1, 2).copy_(0.05 * s);
      motion.select(1, 17).select(1, 2).copy_(-0.05 * s);
    }
    const int64_t T = motion.size(0);
    const auto quats = aa_to_quat(motion);            // [T,J,4]
    const auto times = torch::arange(T, opts) / fps;  // [T] seconds
    ncg::mesh::write_glb_animated(mesh.vertices, mesh.faces, normals, colors.to(at::kCPU),
                                  joints.to(at::kCPU), model.parents().to(at::kCPU),
                                  model.lbs_weights().to(at::kCPU), quats.to(at::kCPU),
                                  times.to(at::kCPU), out_path);
    NCG_LOG_INFO("export (rigged + {} animation) -> {} ({} verts, {} joints, {} frames)", kind,
                 out_path, verts.size(0), J, T);
  } else if (args.get_int("rigged", 1) != 0) {  // rigged (animatable) character
    ncg::mesh::write_glb_skinned(mesh.vertices, mesh.faces, normals, colors.to(at::kCPU),
                                 joints.to(at::kCPU), model.parents().to(at::kCPU),
                                 model.lbs_weights().to(at::kCPU), out_path);
    NCG_LOG_INFO("export (rigged) -> {} ({} verts, {} faces, {} joints)", out_path, verts.size(0),
                 mesh.faces.size(0), model.num_joints());
  } else {
    ncg::mesh::write_glb(mesh.vertices, mesh.faces, normals, colors.to(at::kCPU), out_path);
    NCG_LOG_INFO("export -> {} ({} verts, {} faces)", out_path, verts.size(0), mesh.faces.size(0));
  }
  return 0;
}

// Real multi-photo delighting + relighting (the method on real photos): several casual photos
// of one person -> recover a single canonical albedo (per-photo SH lighting solved away, robust
// to inconsistency) -> render delit albedo + relit under novel lights.
//   ncg_cli delight --images a.jpg,b.jpg,c.jpg --smplx smplx.safetensors --weights nlf.torchscript --out delit.png
int cmd_delight(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "delight requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  const auto paths = split_csv(args.require("images"));
  NCG_CHECK(!paths.empty(), "delight: --images is empty");
  auto nlf = ncg::body::Nlf::load(args.require("weights"), device);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  NCG_CHECK(model.has_faces(), "delight needs a SMPL-X model with faces");
  const auto faces_cpu = model.faces().to(at::kCPU);
  const int64_t V = model.num_verts();

  std::vector<torch::Tensor> obs_l;
  std::vector<torch::Tensor> nrm_l;
  std::vector<torch::Tensor> w_l;
  torch::Tensor betas0;
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto image = ncg::io::load_image(paths[i], 3);
    ncg::body::NlfPrediction pred;
    try {
      pred = nlf.detect(image);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("delight: skipping '{}' ({})", paths[i], e.what());
      continue;
    }
    if (pred.vertices2d.size(0) != V) {
      NCG_LOG_WARN("delight: skipping '{}' (verts {} != {})", paths[i], pred.vertices2d.size(0), V);
      continue;
    }
    if (!betas0.defined()) betas0 = pred.params.betas.to(device);
    const auto v2d = pred.vertices2d.to(device);
    ncg::mesh::TriMesh m;  // per-photo posed normals from NLF's camera-space mesh
    m.vertices = pred.vertices3d.to(at::kCPU);
    m.faces = faces_cpu;
    nrm_l.push_back(ncg::mesh::compute_vertex_normals(m).to(device));
    obs_l.push_back(ncg::recon::sample_vertex_colors(image.to(device), v2d).clamp(0.0, 1.0));
    w_l.push_back(ncg::recon::vertex_visibility(v2d, pred.vertices3d.select(1, 2).to(device),
                                                static_cast<int64_t>(image.size(1)),
                                                static_cast<int64_t>(image.size(2))));
    NCG_LOG_INFO("delight: view {}/{} '{}'", i + 1, paths.size(), paths[i]);
  }
  NCG_CHECK(!obs_l.empty(), "delight: no usable views");
  const int N = static_cast<int>(obs_l.size());

  ncg::recon::InverseRenderConfig cfg;
  cfg.iterations = args.get_int("iters", 80);
  cfg.robust = (N >= 2);
  const auto res = ncg::recon::solve_inverse_render(torch::stack(obs_l, 0), torch::stack(nrm_l, 0),
                                                    torch::stack(w_l, 0), cfg);
  NCG_LOG_INFO("delight: recovered canonical albedo from {} view(s)", N);

  // Canonical rest-pose body for rendering the recovered albedo.
  ncg::body::SmplxParams p;
  p.betas = betas0;
  p.pose_aa = torch::zeros({1, model.num_joints(), 3}, betas0.options());
  p.transl = torch::zeros({1, 3}, betas0.options());
  const auto verts = model.forward(p).vertices.squeeze(0);
  ncg::mesh::TriMesh cm;
  cm.vertices = verts.to(at::kCPU);
  cm.faces = faces_cpu;
  const auto cnrm = ncg::mesh::compute_vertex_normals(cm).to(device);
  const auto pvs = ncg::recon::per_vertex_scale(verts, 0.75F);
  const auto cam = ncg::runtime::Camera::orbit(verts.mean(0), 2.5F, 20.0F, 10.0F, 50.0F, 512, 512,
                                               device);
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "delight"));

  // (a) recovered flat albedo (delit).
  auto albedo_cloud = ncg::recon::gaussians_on_body(verts, 0.012F, res.albedo.clamp(0.0, 1.0), pvs);
  albedo_cloud.to_(device);
  const auto albedo_img = ncg::runtime::render_gaussians(albedo_cloud, cam).image;
  rec.log_image("delight", "albedo", albedo_img);
  ncg::io::save_png(args.get("out", "delit_albedo.png"), albedo_img);

  // (b) relit under an orbiting novel light.
  const auto white = torch::ones({3}, verts.options());
  const float el = 25.0F * static_cast<float>(M_PI) / 180.0F;
  for (int k = 0; k < 8; ++k) {
    const float az = 2.0F * static_cast<float>(M_PI) * static_cast<float>(k) / 8.0F;
    const auto dir = torch::tensor(
        {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)}, verts.options());
    const auto L = ncg::recon::sh_directional_light(dir, white, 0.25F);
    const auto colors = ncg::recon::shade_sh(res.albedo, L, cnrm).clamp(0.0, 1.0);
    auto cloud = ncg::recon::gaussians_on_body(verts, 0.012F, colors, pvs);
    cloud.to_(device);
    char nm[32];
    std::snprintf(nm, sizeof(nm), "relit_%03d", k);
    rec.log_image("delight", nm, ncg::runtime::render_gaussians(cloud, cam).image);
  }
  NCG_LOG_INFO("delight done -> {} + relit frames in {}", args.get("out", "delit_albedo.png"),
               rec.dir().string());
  return 0;
}

// Real-time animate+relight runtime (the deployable forward path, the systems leg): per frame,
// pose the body (LBS), transport the shading normals with the bones (C3), relight under a moving
// light (SH), and splat — all on the GPU. Reports FPS. The differentiable counterpart for
// training is render_soft; a tiled fwd+bwd production rasterizer is further work.
//   ncg_cli runtime --smplx smplx.safetensors [--frames 60 --albedo 0.78]
int cmd_runtime(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "runtime requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  NCG_CHECK(model.has_faces(), "runtime needs a SMPL-X model with faces");
  const int64_t J = model.num_joints();
  const auto neutral = model.neutral_params(1);
  const auto vcanon = model.forward(neutral).vertices.squeeze(0);  // [V,3]
  ncg::mesh::TriMesh cm;
  cm.vertices = vcanon.to(at::kCPU);
  cm.faces = model.faces().to(at::kCPU);
  const auto ncanon = ncg::mesh::compute_vertex_normals(cm).to(device);     // [V,3]
  const auto pvs = ncg::recon::per_vertex_scale(vcanon, 0.75F);             // precompute once
  const auto albedo = torch::full({vcanon.size(0), 3}, args.get_float("albedo", 0.78F),
                                  vcanon.options());
  const auto pose0 = neutral.pose_aa.reshape({1, J, 3});
  const auto opts = vcanon.options();
  const auto white = torch::ones({3}, opts);
  const float el = 25.0F * static_cast<float>(M_PI) / 180.0F;

  torch::Tensor motion;  // [T,J,3] optional extracted-motion playback
  if (args.has("motion")) {
    motion = ncg::io::load_npy(args.require("motion")).to(opts);
    if (args.get_int("inplace", 1) != 0) motion.select(1, 0).zero_();
  }
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "runtime"));
  const int frames = motion.defined() ? static_cast<int>(motion.size(0)) : args.get_int("frames", 60);
  using clk = std::chrono::high_resolution_clock;
  std::vector<double> ft;
  for (int i = 0; i < frames; ++i) {
    const float ph = 2.0F * static_cast<float>(M_PI) * static_cast<float>(i) / frames;
    auto pose = pose0.clone();
    if (motion.defined()) {
      pose = motion[i].unsqueeze(0);  // play the extracted pose for this frame
    } else {
      const float ang = 0.6F * std::sin(ph);
      pose[0][16][2] = ang;    // swing the shoulders (LBS articulation)
      pose[0][17][2] = -ang;
    }
    ncg::body::SmplxParams p{neutral.betas, pose, neutral.transl};

    const auto t0 = clk::now();
    const auto body = model.forward(p);                     // LBS animate
    const auto verts = body.vertices.squeeze(0);
    const auto VT = body.vertex_transforms.squeeze(0);      // [V,4,4]
    const auto Rv = VT.narrow(1, 0, 3).narrow(2, 0, 3);     // [V,3,3] per-vertex rotation
    auto nt = torch::einsum("vab,vb->va", {Rv, ncanon});    // transport normals (C3)
    nt = nt / nt.norm(2, -1, true).clamp_min(1e-8);
    const auto az = ph;                                     // light orbits with the animation
    const auto dir = torch::tensor(
        {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)}, opts);
    const auto colors =
        ncg::recon::shade_sh(albedo, ncg::recon::sh_directional_light(dir, white, 0.25F), nt)
            .clamp(0.0, 1.0);
    auto cloud = ncg::recon::gaussians_on_body(verts, 0.012F, colors, pvs);
    cloud.to_(device);
    const auto cam = ncg::runtime::Camera::orbit(verts.mean(0), 2.5F, 20.0F, 10.0F, 50.0F, 512, 512,
                                                 device);
    const auto img = ncg::runtime::render_gaussians(cloud, cam).image;
    const double sync = img.sum().item<double>();
    ft.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    (void)sync;
    char nm[32];
    std::snprintf(nm, sizeof(nm), "frame_%03d", i);
    rec.log_image("runtime", nm, img);
  }
  double m = 0.0;
  for (double x : ft) m += x;
  m /= static_cast<double>(ft.size());
  NCG_LOG_INFO("runtime: animate+relight {} frames @ {:.2f} ms/frame ({:.0f} FPS) -> {}", frames, m,
               1000.0 / m, rec.dir().string());
  return 0;
}

// Benchmark the inverse-rendering method on real SMPL-X geometry (the paper's figures):
// albedo error vs #photos (C1) and vs corruption rate, robust vs non-robust (C2), + relighting
// error under a novel light. Synthetic ground-truth albedo so error is measurable.
//   ncg_cli benchmark --smplx smplx.safetensors [--seed 0 --run benchmark]
int cmd_benchmark(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "benchmark requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  NCG_CHECK(model.has_faces(), "benchmark needs a SMPL-X model with faces");
  const auto verts = model.forward(model.neutral_params(1)).vertices.squeeze(0);
  ncg::mesh::TriMesh mesh;
  mesh.vertices = verts.to(at::kCPU);
  mesh.faces = model.faces().to(at::kCPU);
  const auto normals = ncg::mesh::compute_vertex_normals(mesh).to(device);  // [V,3]
  const int64_t V = verts.size(0);
  const auto opts = verts.options();
  const int seeds = args.get_int("seeds", 5);
  const auto basis = ncg::recon::sh_basis(normals);  // [V,9] (geometry only)
  const auto Lnovel = ncg::recon::sh_directional_light(torch::tensor({0.4F, -0.7F, 0.6F}, opts),
                                                       torch::ones({3}, opts), 0.25F);
  using clk = std::chrono::high_resolution_clock;
  auto ms_since = [](clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
  };
  auto stat = [](const std::vector<double>& xs) {
    double m = 0.0;
    for (double x : xs) m += x;
    m /= static_cast<double>(xs.size());
    double v = 0.0;
    for (double x : xs) v += (x - m) * (x - m);
    v /= static_cast<double>(xs.size() > 1 ? xs.size() - 1 : 1);
    return std::make_pair(m, std::sqrt(v));
  };
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "benchmark"));
  NCG_LOG_INFO("=== Benchmark: V={} real geometry, {} seeds (mean ± std) ===", V, seeds);

  // C1 — albedo & relight error + solve time vs #photos.
  NCG_LOG_INFO("--- C1: albedo / relight error vs #photos ---");
  for (int N : {1, 2, 3, 5, 8, 12}) {
    std::vector<double> ae;
    std::vector<double> re;
    std::vector<double> ms;
    for (int s = 0; s < seeds; ++s) {
      torch::manual_seed(1000 + s);
      const auto a_true = torch::rand({V, 3}, opts) * 0.7F + 0.2F;
      auto L = torch::randn({N, 3, 9}, opts) * 0.25F;
      L.select(2, 0) += 1.2F;
      const auto obs = (a_true.unsqueeze(0) * torch::einsum("nck,vk->nvc", {L, basis})).clamp_min(0.0);
      const auto nv = normals.unsqueeze(0).expand({N, V, 3}).contiguous();
      ncg::recon::InverseRenderConfig cfg;
      cfg.iterations = 60;
      const auto t0 = clk::now();
      const auto r = ncg::recon::solve_inverse_render(obs, nv, torch::ones({N, V}, opts), cfg);
      const double sync = r.albedo.sum().item<double>();  // force GPU completion
      ms.push_back(ms_since(t0));
      (void)sync;
      const auto sc = (r.albedo * a_true).sum(0) / (r.albedo * r.albedo).sum(0).clamp_min(1e-8);
      ae.push_back((a_true - r.albedo * sc).abs().mean().item<double>());
      const auto gt = ncg::recon::shade_sh(a_true, Lnovel, normals);
      re.push_back((gt - ncg::recon::shade_sh(r.albedo * sc, Lnovel, normals)).norm().item<double>() /
                   gt.norm().clamp_min(1e-8).item<double>());
    }
    const auto [am, as] = stat(ae);
    const auto [rm, rs] = stat(re);
    const auto [tm, ts] = stat(ms);
    NCG_LOG_INFO("  N={:2d}  albedo={:.4f}±{:.4f}  relight={:.4f}±{:.4f}  solve={:6.1f}±{:.1f}ms", N,
                 am, as, rm, rs, tm, ts);
  }

  // C2 — robust vs plain albedo error vs corruption rate (N=8).
  NCG_LOG_INFO("--- C2: albedo error vs corruption (robust vs plain), N=8 ---");
  const int Nc = 8;
  for (double rho : {0.0, 0.1, 0.2, 0.35, 0.5}) {
    std::vector<double> er;
    std::vector<double> ep;
    for (int s = 0; s < seeds; ++s) {
      torch::manual_seed(2000 + s);
      const auto a_true = torch::rand({V, 3}, opts) * 0.7F + 0.2F;
      auto L = torch::randn({Nc, 3, 9}, opts) * 0.25F;
      L.select(2, 0) += 1.2F;
      const auto clean = (a_true.unsqueeze(0) * torch::einsum("nck,vk->nvc", {L, basis})).clamp_min(0.0);
      const auto corrupt = torch::rand({Nc, V}, opts) < rho;
      const auto obs = torch::where(corrupt.unsqueeze(-1), torch::rand({Nc, V, 3}, opts), clean);
      const auto nv = normals.unsqueeze(0).expand({Nc, V, 3}).contiguous();
      const auto w = torch::ones({Nc, V}, opts);
      ncg::recon::InverseRenderConfig rc;
      rc.iterations = 80;
      rc.robust = true;
      ncg::recon::InverseRenderConfig pc = rc;
      pc.robust = false;
      auto serr = [&](const torch::Tensor& a) {
        const auto sc = (a * a_true).sum(0) / (a * a).sum(0).clamp_min(1e-8);
        return (a_true - a * sc).abs().mean().item<double>();
      };
      er.push_back(serr(ncg::recon::solve_inverse_render(obs, nv, w, rc).albedo));
      ep.push_back(serr(ncg::recon::solve_inverse_render(obs, nv, w, pc).albedo));
    }
    const auto [rm, rs] = stat(er);
    const auto [pm, ps] = stat(ep);
    NCG_LOG_INFO("  corrupt={:3.0f}%  robust={:.4f}±{:.4f}  plain={:.4f}±{:.4f}", rho * 100, rm, rs,
                 pm, ps);
  }

  // Comparison — relightability vs a radiance baseline (what NeRF / vanilla 3DGS recover: one
  // baked appearance per vertex, no lighting model). Both evaluated under a NOVEL light. The
  // baseline is given its best global scale to GT, so this is its best case.
  NCG_LOG_INFO("--- Comparison: relight error (novel light), ours vs radiance baseline (NeRF/3DGS) ---");
  {
    const int Ncmp = 8;
    std::vector<double> eo;
    std::vector<double> eb;
    for (int s = 0; s < seeds; ++s) {
      torch::manual_seed(3000 + s);
      const auto a_true = torch::rand({V, 3}, opts) * 0.7F + 0.2F;
      auto L = torch::randn({Ncmp, 3, 9}, opts) * 0.25F;
      L.select(2, 0) += 1.2F;
      const auto obs = (a_true.unsqueeze(0) * torch::einsum("nck,vk->nvc", {L, basis})).clamp_min(0.0);
      const auto nv = normals.unsqueeze(0).expand({Ncmp, V, 3}).contiguous();
      const auto gt = ncg::recon::shade_sh(a_true, Lnovel, normals);  // GT under the novel light
      const double gtn = gt.norm().clamp_min(1e-8).item<double>();
      // Ours: recover albedo, relight under the novel light.
      ncg::recon::InverseRenderConfig cfg;
      cfg.iterations = 60;
      const auto ra = ncg::recon::solve_inverse_render(obs, nv, torch::ones({Ncmp, V}, opts), cfg).albedo;
      const auto sc = (ra * a_true).sum(0) / (ra * ra).sum(0).clamp_min(1e-8);
      eo.push_back((ncg::recon::shade_sh(ra * sc, Lnovel, normals) - gt).norm().item<double>() / gtn);
      // Radiance baseline (NeRF/3DGS): one baked color per vertex = mean radiance; it cannot
      // relight, so its output under the novel light is that baked image (best global scale).
      const auto baked = obs.mean(0);
      const auto scb = (baked * gt).sum() / (baked * baked).sum().clamp_min(1e-8);
      eb.push_back((baked * scb - gt).norm().item<double>() / gtn);
    }
    const auto [om, os] = stat(eo);
    const auto [bm, bs] = stat(eb);
    NCG_LOG_INFO("  ours (relightable) = {:.4f}±{:.4f}   radiance baseline (baked) = {:.4f}±{:.4f}",
                 om, os, bm, bs);
  }

  // C3 — animate/relight commutation error + transport timing.
  {
    torch::manual_seed(7);
    auto nn = torch::randn({V, 3}, opts);
    nn = nn / nn.norm(2, -1, true);
    const auto a = torch::rand({V, 3}, opts) * 0.6F + 0.3F;
    const float th = 0.7F;
    const auto R = torch::tensor({{std::cos(th), 0.0F, std::sin(th)},
                                  {0.0F, 1.0F, 0.0F},
                                  {-std::sin(th), 0.0F, std::cos(th)}},
                                 opts);
    const auto d = torch::tensor({0.3F, -0.6F, 0.7F}, opts);
    const auto t0 = clk::now();
    const auto nposed = ncg::recon::transport_normals(nn, torch::ones({V, 1}, opts), R.unsqueeze(0));
    const double tms = ms_since(t0) + 0.0 * nposed.sum().item<double>();
    const auto cA = ncg::recon::shade_sh(a, ncg::recon::sh_directional_light(d, torch::ones({3}, opts), 0.2F), nposed);
    const auto cB = ncg::recon::shade_sh(
        a, ncg::recon::sh_directional_light(torch::matmul(R.t(), d), torch::ones({3}, opts), 0.2F), nn);
    NCG_LOG_INFO("--- C3: animate∘relight vs relight∘animate ---");
    NCG_LOG_INFO("  max commutation error = {:.2e}  (transport {:.2f}ms / {} verts)",
                 (cA - cB).abs().max().item<double>(), tms, V);
  }

  // Render throughput on the real body (forward splat).
  {
    auto cloud = ncg::recon::gaussians_on_body(verts, 0.012F, torch::full({V, 3}, 0.7F, opts),
                                               ncg::recon::per_vertex_scale(verts, 0.75F));
    cloud.to_(device);
    const auto cam = ncg::runtime::Camera::orbit(verts.mean(0), 2.5F, 20.0F, 10.0F, 50.0F, 512, 512,
                                                 device);
    (void)ncg::runtime::render_gaussians(cloud, cam).image.sum().item<double>();  // warm up
    std::vector<double> fr;
    for (int i = 0; i < 30; ++i) {
      const auto t0 = clk::now();
      const auto img = ncg::runtime::render_gaussians(cloud, cam).image;
      (void)img.sum().item<double>();
      fr.push_back(ms_since(t0));
    }
    const auto [fm, fs] = stat(fr);
    NCG_LOG_INFO("--- render throughput (512x512, {} gaussians) ---", V);
    NCG_LOG_INFO("  forward splat = {:.2f}±{:.2f}ms  ({:.0f} FPS)", fm, fs, 1000.0 / fm);
  }

  NCG_LOG_INFO("benchmark complete -> {}", rec.dir().string());
  return 0;
}

// C4 — recover a person's motion STYLE from several extracted clips (different actions) and apply
// it to a target action (projecting the target onto the recovered style subspace = "this action,
// in their style", and regularizing the noisy casual motion). Pure motion math (CPU); writes the
// styled motion .npy for `export --motion`.
//   ncg_cli style --motions a.npy,b.npy,c.npy [--target a.npy] --rank 8 --out_motion styled.npy
int cmd_style(const ncg::app::Args& args) {
  const auto mpaths = split_csv(args.require("motions"));
  NCG_CHECK(!mpaths.empty(), "style: --motions is empty");
  std::vector<torch::Tensor> clips;
  int64_t J = 0;
  for (const auto& p : mpaths) {
    const auto m = ncg::io::load_npy(p).to(at::kFloat);  // [T,J,3]
    NCG_CHECK(m.dim() == 3 && m.size(2) == 3, "style: each motion .npy must be [T,J,3]");
    J = m.size(1);
    clips.push_back(m.reshape({m.size(0), -1}).contiguous());  // [T, J*3]
  }
  ncg::recon::MotionStyleConfig cfg;
  cfg.rank = args.get_int("rank", 8);
  cfg.iterations = args.get_int("iters", 60);
  const auto res = ncg::recon::solve_motion_style(clips, cfg);
  NCG_LOG_INFO("style: recovered rank-{} style from {} clips", cfg.rank, clips.size());

  const auto tgt = ncg::io::load_npy(args.has("target") ? args.require("target") : mpaths[0])
                       .to(at::kFloat);  // [T,J,3]
  const auto styled = ncg::recon::apply_motion_style(tgt.reshape({tgt.size(0), -1}), res.style)
                          .reshape({tgt.size(0), J, 3})
                          .contiguous();
  const auto out = args.get("out_motion", "styled_motion.npy");
  ncg::io::save_npy(out, styled);
  NCG_LOG_INFO("style: styled motion ({} frames) -> {} (feed to export --motion)", styled.size(0),
               out);
  return 0;
}

// Trains an animatable Gaussian avatar from a directory of video frames of one person. Each frame
// is run through NLF to get its SMPL-X pose + a solved camera; fit_avatar then optimizes a single
// canonical cloud (anisotropic splats, per-frame exposure, D-SSIM) so it reproduces every posed
// frame. Multi-pose casual video thus becomes multi-view evidence for one avatar — the path from a
// projected-color mannequin to a real likeness. Renders fit-check + novel-view turntable frames.
//   ncg_cli avatar --frames dir/ --weights nlf.torchscript --smplx model.safetensors \
//                  [--max-frames 60 --res 288 --iters 3000 --out-prefix rock_avatar]
int cmd_avatar(const ncg::app::Args& args) {
  const auto device = ncg::cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "avatar"));

  // Enumerate frame images (sorted), then evenly subsample to --max-frames.
  namespace fs = std::filesystem;
  std::vector<std::string> all;
  for (const auto& e : fs::directory_iterator(args.require("frames"))) {
    const auto p = e.path().string();
    const auto ext = e.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".JPG") all.push_back(p);
  }
  std::sort(all.begin(), all.end());
  NCG_CHECK(!all.empty(), "avatar: no images in --frames dir");
  const int max_frames = args.get_int("max-frames", 60);
  std::vector<std::string> paths;
  if (static_cast<int>(all.size()) <= max_frames) {
    paths = all;
  } else {
    const double step = static_cast<double>(all.size()) / max_frames;
    for (int i = 0; i < max_frames; ++i) paths.push_back(all[static_cast<size_t>(i * step)]);
  }
  NCG_LOG_INFO("avatar: {} frames selected from {} in {}", paths.size(), all.size(),
               args.require("frames"));

  ncg::body::NlfConfig nc;
  nc.detection = args.get_int("detection", 0);
  auto nlf = ncg::body::Nlf::load(args.require("weights"), device, nc);
  const auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);

  const int res = args.get_int("res", 288);
  namespace F = torch::nn::functional;
  auto downscale = [&](const torch::Tensor& img) {
    const double s = std::min(1.0, static_cast<double>(res) /
                                       static_cast<double>(std::max(img.size(1), img.size(2))));
    auto small = F::interpolate(img.unsqueeze(0), F::InterpolateFuncOptions()
                                                      .scale_factor(std::vector<double>{s, s})
                                                      .mode(torch::kBilinear)
                                                      .align_corners(false))
                     .squeeze(0);
    return std::make_pair(small, s);
  };

  std::vector<ncg::fit::AvatarFrame> frames;
  torch::Tensor betas0;
  torch::Tensor init_colors;
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto img_full = ncg::io::load_image(paths[i]).to(device);
    auto [img, s] = downscale(img_full);
    ncg::body::NlfPrediction pred;
    try {
      pred = nlf.detect(img_full);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("avatar: NLF failed on {} ({}), skipping", paths[i], e.what());
      continue;
    }
    if (!betas0.defined()) betas0 = pred.params.betas.to(device);

    ncg::fit::AvatarFrame fr;
    fr.pose_aa = pred.params.pose_aa.squeeze(0).to(device);  // [J,3] camera-frame orientation kept
    fr.transl = pred.params.transl.squeeze(0).to(device);
    fr.target = img;
    const int w = static_cast<int>(img.size(2));
    const int h = static_cast<int>(img.size(1));
    fr.camera = ncg::runtime::solve_pinhole_camera(pred.vertices3d.to(device),
                                                   pred.vertices2d.to(device) * s, w, h);
    frames.push_back(std::move(fr));

    if (!init_colors.defined()) {  // seed appearance from the first good frame
      const auto v2d = pred.vertices2d.to(device) * s;
      auto cols = ncg::recon::sample_vertex_colors(img, v2d).clamp(0.0, 1.0);
      const auto vis = ncg::recon::vertex_visibility(v2d, pred.vertices3d.select(1, 2).to(device), h, w);
      init_colors = torch::where(vis.unsqueeze(1) > 0, cols, torch::full_like(cols, 0.6F));
    }
  }
  NCG_CHECK(frames.size() >= 2, "avatar: need >=2 usable frames");
  NCG_LOG_INFO("avatar: training on {} frames at {}px", frames.size(), res);

  ncg::fit::AvatarFitConfig cfg;
  cfg.iterations = args.get_int("iters", 3000);
  cfg.init_scale = args.get_float("scale", 0.015F);
  cfg.lambda_dssim = args.get_float("dssim", 0.2F);
  cfg.per_view_exposure = args.get_int("exposure", 1) != 0;
  cfg.log_every = 50;
  cfg.dump_every = args.get_int("dump-every", 500);
  const auto canonical = ncg::fit::fit_avatar(model, betas0, frames, init_colors, cfg, &rec);

  // Fit-check: render the avatar at frame 0's pose/camera next to the target.
  const auto prefix = args.get("out-prefix", "avatar");
  {
    ncg::body::SmplxParams p0;
    p0.betas = betas0;
    p0.pose_aa = frames[0].pose_aa.unsqueeze(0);
    p0.transl = frames[0].transl.unsqueeze(0);
    const auto vt0 = model.forward(p0).vertex_transforms.squeeze(0);
    const auto posed = ncg::fit::deform_avatar(canonical, vt0);
    const auto fit = ncg::runtime::render_soft_aniso(posed, frames[0].camera).image;
    ncg::io::save_png(prefix + "_fit0.png", fit.detach());
    ncg::io::save_png(prefix + "_tgt0.png", frames[0].target.detach());
    NCG_LOG_INFO("avatar: fit0 PSNR vs target = {:.2f} dB",
                 ncg::record::psnr(fit.detach(), frames[0].target));
  }
  // Novel-view turntable of the canonical (rest-pose) avatar — shows a coherent 3D likeness.
  {
    const int nv = args.get_int("turn", 8);
    const auto cams = ncg::runtime::orbit_trajectory(canonical.positions.mean(0),
                                                     args.get_float("radius", 2.4F), 0.0F, nv,
                                                     50.0F, res, res, device);
    for (int i = 0; i < nv; ++i) {
      const auto im = ncg::runtime::render_soft_aniso(canonical, cams[i]).image.detach();
      char name[64];
      std::snprintf(name, sizeof(name), "%s_turn%02d.png", prefix.c_str(), i);
      ncg::io::save_png(name, im);
    }
    NCG_LOG_INFO("avatar: wrote {} turntable views -> {}_turn*.png", nv, prefix);
  }
  NCG_LOG_INFO("avatar: done -> {}_fit0.png / {}_turn*.png ({} gaussians)", prefix, prefix,
               canonical.size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  ncg::init_logging();
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: ncg_cli "
                 "<pipeline|render|turntable|select|fitimg|fit|fuse|relight|export|benchmark|"
                 "runtime|nerf|style|avatar> [--flags]\n");
    return 2;
  }
  const std::string cmd = argv[1];
  const ncg::app::Args args(argc, argv);
  try {
    if (cmd == "pipeline") return cmd_pipeline(args);
    if (cmd == "render") return cmd_render(args);
    if (cmd == "turntable") return cmd_turntable(args);
    if (cmd == "select") return cmd_select(args);
    if (cmd == "fitimg") return cmd_fitimg(args);
    if (cmd == "fit") return cmd_fit(args);
    if (cmd == "fuse") return cmd_fuse(args);
    if (cmd == "relight") return cmd_relight(args);
    if (cmd == "export") return cmd_export(args);
    if (cmd == "benchmark") return cmd_benchmark(args);
    if (cmd == "delight") return cmd_delight(args);
    if (cmd == "runtime") return cmd_runtime(args);
    if (cmd == "style") return cmd_style(args);
    if (cmd == "avatar") return cmd_avatar(args);
    if (cmd == "nerf") return cmd_nerf(args);
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ncg_cli %s error: %s\n", cmd.c_str(), e.what());
    return 1;
  }
}
