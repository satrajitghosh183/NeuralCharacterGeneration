// ncg_cli — pipeline driver with subcommands. Phase 1 wires the parts that exist.
//
//   ncg_cli render --smplx model.safetensors --out out.png [--width .. --height ..]
//   ncg_cli select --images a.jpg,b.jpg,c.jpg
//   ncg_cli fit    --image photo.jpg --weights nlf.safetensors   (NLF port: not implemented)
#include <args.hpp>

#include <ncg/body/album_gate.hpp>
#include <ncg/body/nlf.hpp>
#include <ncg/body/smplx.hpp>
#include <ncg/core/device.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/fit/fit_avatar.hpp>
#include <ncg/fit/fit_image.hpp>
#include <ncg/fit/splat_bind.hpp>
#include <ncg/diffuse/completion.hpp>
#include <ncg/diffuse/scheduler.hpp>
#include <ncg/diffuse/sd_guidance.hpp>
#include <ncg/diffuse/sds.hpp>
#include <ncg/geom/solve_geometry.hpp>
#include <ncg/io/image.hpp>
#include <ncg/io/npy.hpp>
#include <ncg/io/safetensors.hpp>
#include <ncg/recon/face_identity.hpp>
#include <ncg/mesh/extract.hpp>
#include <ncg/nerf/nerf.hpp>
#include <ncg/recon/appearance.hpp>
#include <ncg/recon/init_from_body.hpp>
#include <ncg/recon/inverse_render.hpp>
#include <ncg/recon/uv_texture.hpp>
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
#include <limits>
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

// A-pose: lower the arms from SMPL-X's T-pose (joints 16/17 = shoulders) so the rest mesh reads as
// a relaxed standing person, not a mannequin. Cosmetic only — does not add hair/clothes/face detail.
void apply_apose(torch::Tensor& pose, float s) {
  if (pose.size(1) <= 17) return;
  pose.index_put_({0, 16, 2}, -s);  // left shoulder down
  pose.index_put_({0, 17, 2}, s);   // right shoulder down
}

// Per-texel robust albedo (C5 at high resolution): lift the C1/C2 inverse-render from per-vertex
// (~10^4) to per-texel (T^2) over the SMPL-X UV layout. For each valid texel, barycentrically map to
// the image in every frame, sample observation + normal + visibility, and run the same robust
// solver in texel space — so the consistent face/skin gains real resolution. Returns albedo [T,T,3].
torch::Tensor recover_uv_albedo(const ncg::body::SmplxModel& model,
                                const std::vector<torch::Tensor>& imgs,
                                const std::vector<torch::Tensor>& v2ds,
                                const std::vector<torch::Tensor>& nrms,
                                const std::vector<torch::Tensor>& viss, int T,
                                const torch::Tensor& rest_verts, torch::Tensor& mask_out,
                                torch::Tensor& normal_out, torch::Tensor& pos_out,
                                torch::Tensor& gnrm_out, float detail_weight = 1.5F,
                                float deshade = 0.5F, float chroma = 0.6F, float seam = 0.6F) {
  const auto device = model.uv_coords().device();
  const auto ras = ncg::recon::uv_rasterize(model.uv_coords(), model.uv_faces(), T);
  const auto face = ras.face.to(device);                              // [T^2]
  const auto bary = ras.bary.to(device);                             // [T^2,3]
  const auto valid = (face >= 0).to(at::kFloat);                     // [T^2]
  const auto faces = model.faces().to(device);                      // [F,3]
  const auto geomv = faces.index_select(0, face.clamp_min(0)).reshape(-1);  // [T^2*3]
  const int64_t TT = face.size(0);
  namespace F = torch::nn::functional;

  std::vector<torch::Tensor> obs_l, nrm_l, w_l;
  for (size_t f = 0; f < imgs.size(); ++f) {
    const auto img = imgs[f].to(device);                             // [3,H,W]
    const int H = static_cast<int>(img.size(1)), W = static_cast<int>(img.size(2));
    const auto v2d = v2ds[f].to(device);                            // [V,2]
    const auto pv = v2d.index_select(0, geomv).reshape({TT, 3, 2});  // [T^2,3,2]
    const auto texel_uv = (pv * bary.unsqueeze(2)).sum(1);          // [T^2,2] pixel coords
    const auto gx = texel_uv.select(1, 0) / (W - 1) * 2 - 1;
    const auto gy = texel_uv.select(1, 1) / (H - 1) * 2 - 1;
    const auto grid = torch::stack({gx, gy}, 1).view({1, TT, 1, 2});
    const auto samp = F::grid_sample(
        img.unsqueeze(0), grid,
        F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
    obs_l.push_back(samp.view({3, TT}).t().contiguous());          // [T^2,3]
    const auto nv = nrms[f].to(device).index_select(0, geomv).reshape({TT, 3, 3});
    const auto tn = (nv * bary.unsqueeze(2)).sum(1);               // [T^2,3]
    nrm_l.push_back(tn / tn.norm(2, 1, true).clamp_min(1e-6));
    const auto vv = viss[f].to(device).index_select(0, geomv).reshape({TT, 3});
    auto tw = std::get<0>(vv.min(1)) * valid;                      // visibility × valid
    // Front-facing: camera-space normal z<0 = facing the camera. Grazing/back texels (which sample
    // sky/background when the subject is small in frame) get ~0 weight — kills the background bleed.
    const auto tnu = tn / tn.norm(2, 1, true).clamp_min(1e-6F);
    tw = tw * torch::relu(-tnu.select(1, 2));
    // In-bounds: reject texels that project outside the image (no edge background bleed).
    const auto u = texel_uv.select(1, 0);
    const auto vc = texel_uv.select(1, 1);
    const auto inb = ((u >= 0) & (u <= W - 1) & (vc >= 0) & (vc <= H - 1)).to(at::kFloat);
    // Specular down-weight: a Lambertian albedo solve must not trust blown, DESATURATED highlights
    // (they are view-dependent reflection, not albedo — the source of the baked-in bright blotches).
    // Specular ≈ bright (high luminance) AND low saturation (R≈G≈B). Soft-gate both, drop those obs.
    const auto of = obs_l.back();                                  // [T^2,3] this frame's samples
    const auto lum = of.mean(1);                                  // [T^2]
    const auto mx = std::get<0>(of.max(1)).clamp_min(1e-3F);
    const auto sat = (mx - std::get<0>(of.min(1))) / mx;          // [T^2]
    const auto spec = torch::sigmoid((lum - 0.72F) * 16.0F) * torch::sigmoid((0.24F - sat) * 16.0F);
    w_l.push_back(tw * inb * (1.0F - 0.96F * spec));              // [T^2] harder specular reject
  }
  // ============ STEP 1: PER-PHOTO RADIOMETRIC EQUALIZATION (before any fusion) =====================
  // Each photo baked in its own exposure + white balance; AVERAGING them is the root cause of the
  // mottled chroma (diag row 1) AND the faceted seams (row 2). Solve a per-photo 3-channel gain that
  // minimizes cross-photo disagreement on SHARED texels (alternating: weighted ref → per-channel LSQ
  // gain → gauge-normalize), so every photo lives in ONE albedo space. Gauge: per-channel mean gain=1
  // (absolute tone is set later by white-balance) → no global drift.
  auto O = torch::stack(obs_l, 0);                                  // [N,T^2,3]
  const auto W = torch::stack(w_l, 0);                             // [N,T^2]
  const int64_t Nph = O.size(0);
  {
    const auto Wt = W.unsqueeze(2);                                // [N,T^2,1]
    const auto cnt2 = (((W > 0.2F).to(at::kFloat).sum(0)) >= 2.0F).to(at::kFloat).unsqueeze(1);  // shared
    auto shared_var = [&](const torch::Tensor& Oe) {
      const auto wm = (Wt * Oe).sum(0) / Wt.sum(0).clamp_min(1e-6F);
      const auto v = (Wt * (Oe - wm.unsqueeze(0)).pow(2)).sum(0) / Wt.sum(0).clamp_min(1e-6F);
      return ((v * cnt2).sum() / cnt2.sum().clamp_min(1.0F)).item<double>();
    };
    const double vb = shared_var(O);
    auto eqg = torch::ones({Nph, 1, 3}, O.options());
    for (int it = 0; it < 6; ++it) {
      const auto ref = (Wt * (O * eqg)).sum(0) / Wt.sum(0).clamp_min(1e-6F);      // [T^2,3]
      auto g = ((Wt * ref.unsqueeze(0) * O).sum(1) /
                (Wt * O.pow(2)).sum(1).clamp_min(1e-6F)).clamp(0.3F, 3.0F);       // [N,3] LSQ scale
      g = g / g.mean(0, true).clamp_min(1e-6F);                                   // gauge
      eqg = g.view({Nph, 1, 3});
    }
    O = (O * eqg).clamp(0.0F, 2.0F);
    NCG_LOG_INFO("equalize-gain: shared-texel var {:.6f} -> {:.6f} ({:.2f}x global exposure/WB)", vb,
                 shared_var(O), vb / std::max(shared_var(O), 1e-12));
    for (int64_t i = 0; i < Nph; ++i) obs_l[static_cast<size_t>(i)] = O[i].contiguous();
  }
  // ============ STEP 2: robust per-texel MEDIAN merge. The residual cross-photo disagreement after
  // global equalization is SPATIALLY-LOCAL (cast shadows / specular / occlusion edges) — a per-photo
  // gain OR a per-photo SH delight can't model it (the SH solve is ill-conditioned on casual photos
  // and dividing by it AMPLIFIES disagreement, measured). The robust per-texel MEDIAN is what rejects
  // that — it discards the one shadowed/specular view a mean would smear in. solve_inverse_render is
  // kept only for the per-view lights → photometric normal map.
  ncg::recon::InverseRenderConfig ic;
  ic.iterations = 80;
  ic.robust = true;
  const auto ir = ncg::recon::solve_inverse_render(O, torch::stack(nrm_l, 0), W, ic);  // lights→normals
  // GATE1 (reported honestly): cross-photo shared-skin variance, raw vs robust per-texel-median residual.
  torch::Tensor albedo;
  {
    const auto good = (W > 0.2F).unsqueeze(2);                                    // [N,T^2,1]
    const auto wmean = (W.unsqueeze(2) * O).sum(0) / W.unsqueeze(2).sum(0).clamp_min(1e-6F);
    const auto Om = torch::where(good, O, torch::full_like(O, std::numeric_limits<float>::quiet_NaN()));
    const auto med = std::get<0>(torch::nanmedian(Om, 0));                        // [T^2,3]
    albedo = torch::where(torch::isnan(med), wmean, med).clamp(0.0F, 1.0F);
    // GATE1 metric: residual of each view vs the consensus MEDIAN on shared skin texels (how much the
    // merge had to reject) — raw mean-residual vs median-residual.
    const auto Wt = W.unsqueeze(2);
    const auto cnt2 = (((W > 0.2F).to(at::kFloat).sum(0)) >= 2.0F).to(at::kFloat).unsqueeze(1);
    auto resid = [&](const torch::Tensor& ref) {
      const auto v = (Wt * (O - ref.unsqueeze(0)).pow(2)).sum(0) / Wt.sum(0).clamp_min(1e-6F);
      return ((v * cnt2).sum() / cnt2.sum().clamp_min(1.0F)).item<double>();
    };
    const double vr = resid(wmean), vd = resid(albedo);
    NCG_LOG_INFO("merge(GATE1): shared var vs mean {:.6f} vs median-consensus {:.6f} ({:.2f}x rejected)",
                 vr, vd, vd / std::max(vr, 1e-12));
  }

  // ---- algorithmic UV cleanup: confidence-weighted push-pull inpaint + edge-aware smoothing -----
  // Per-texel coverage Σ_f w is the confidence. Low-coverage texels — UV seams, rarely-seen cheek/
  // jaw, specular-rejected spots — are filled from confident neighbours by normalized convolution
  // (blur(a·c)/blur(c)), iterated so confidence diffuses inward; confident texels are preserved.
  // Purely algorithmic (no manual touch-up): removes seam discontinuities and isolated speckles.
  {
    namespace Fc = torch::nn::functional;
    const auto conf = torch::stack(w_l, 0).sum(0).clamp_min(0.0F).view({1, 1, T, T});  // [1,1,T,T]
    auto k1 = torch::tensor({1.F, 4.F, 6.F, 4.F, 1.F}, albedo.options());
    auto k2 = torch::outer(k1, k1);
    k2 = k2 / k2.sum();
    const auto kc = k2.view({1, 1, 5, 5});
    const auto ka = kc.expand({3, 1, 5, 5}).contiguous();
    auto blur = [&](const torch::Tensor& x, const torch::Tensor& ker, int64_t g) {
      return Fc::conv2d(x, ker, Fc::Conv2dFuncOptions().padding(2).groups(g));
    };
    auto a = albedo.t().reshape({1, 3, T, T}).contiguous();  // [1,3,T,T]
    const auto hi = (conf > 0.5F * conf.mean()).to(albedo.dtype());  // originally-confident mask
    auto c = conf.clone();
    for (int it = 0; it < 16; ++it) {  // push-pull: fill holes from confident neighbours
      const auto filled = blur(a * c, ka, 3) / blur(c, kc, 1).clamp_min(1e-6F);
      a = a * hi + filled * (1.0F - hi);
      c = torch::maximum(conf, blur(c, kc, 1) * (1.0F - hi) + conf * hi);
    }
    a = a * 0.65F + blur(a, ka, 3) * 0.35F;  // mild edge-aware smooth (de-speckle, keep pores)
    albedo = a.reshape({3, T * T}).t().contiguous().clamp(0.0F, 1.0F);
  }

  // ---- HOMOMORPHIC DESHADE: remove baked shading/AO blotches the in-the-wild delighting missed ----
  // The Lambertian+SH delight can't model cast shadows / occlusion under harsh casual lighting, so
  // residual LOW-FREQUENCY luminance (dark eye sockets, bright forehead) survives in the albedo.
  // Texture-resolution rendering makes those blotches obvious. Divide out the low-frequency luminance
  // (normalized convolution over the valid UV region) with a BOUNDED gain — flattens the shading while
  // leaving pores/edges (high-freq) and skin colour (chroma) untouched.
  if (deshade > 0.0F) {
    namespace Fc = torch::nn::functional;
    auto bk1 = torch::tensor({1.F, 4.F, 6.F, 4.F, 1.F}, albedo.options());
    auto bk2 = torch::outer(bk1, bk1);
    bk2 = bk2 / bk2.sum();
    const auto bk = bk2.view({1, 1, 5, 5});
    auto blur1 = [&](const torch::Tensor& x) {
      return Fc::conv2d(x, bk, Fc::Conv2dFuncOptions().padding(2).groups(1));
    };
    const auto vmask = valid.view({1, 1, T, T});
    auto lum = albedo.mean(1).view({1, 1, T, T});                       // [1,1,T,T]
    auto lp = lum * vmask;
    for (int it = 0; it < 24; ++it)                                     // big low-pass (normalized)
      lp = blur1(lp) / blur1(vmask).clamp_min(1e-6F) * vmask + lp * (1.0F - vmask);
    const auto tgt = (lum * vmask).sum() / vmask.sum().clamp_min(1e-6F);  // global mean luminance
    auto gain = (tgt / lp.clamp_min(0.05F)).clamp(1.0F - deshade, 1.0F + deshade);  // bounded
    albedo = (albedo * gain.view({T * T, 1})).clamp(0.0F, 1.0F);
  }

  // ============ STEP 3: SEAM-AWARE multi-band blend (kill the faceting) ============================
  // Per-texel independence leaves hard adjacent jumps at photo boundaries (diag row 2). Reconstruct as
  // base(low-freq, heavily smoothed ACROSS seams) + detail(high-freq pores), so the mid-frequency STEP
  // discontinuities vanish while fine skin texture survives. Normalized convolution stays inside the
  // valid UV region (no background bleed).
  if (seam > 0.0F) {
    namespace Fc = torch::nn::functional;
    const auto vmask = valid.view({1, 1, T, T});
    auto bk1 = torch::tensor({1.F, 4.F, 6.F, 4.F, 1.F}, albedo.options());
    auto bk2 = torch::outer(bk1, bk1);
    bk2 = bk2 / bk2.sum();
    const auto kc = bk2.view({1, 1, 5, 5});
    const auto ka = kc.expand({3, 1, 5, 5}).contiguous();
    auto nblur = [&](const torch::Tensor& x) {
      return Fc::conv2d(x * vmask, ka, Fc::Conv2dFuncOptions().padding(2).groups(3)) /
                 Fc::conv2d(vmask, kc, Fc::Conv2dFuncOptions().padding(2).groups(1)).clamp_min(1e-6F) *
                 vmask + x * (1.0F - vmask);
    };
    // Heavily smooth to dissolve the mid-frequency seam STEPS (a frequency split can't separate a
    // step-edge from a pore — both are broadband — so we smooth here and let the per-view DETAIL pass
    // below re-add pores from a SINGLE view, which carries no cross-view seam).
    auto a = albedo.t().reshape({1, 3, T, T}).contiguous();
    const int nb = static_cast<int>(4 + 26 * seam);
    for (int it = 0; it < nb; ++it) a = nblur(a);
    albedo = a.reshape({3, T * T}).t().contiguous().clamp(0.0F, 1.0F);
    // CHROMA cleanup: skin colour is spatially smooth. Heavily smooth the CHROMA (keeps luminance
    // detail) to remove residual colour outliers (seam-edge tints, specular residue) + mildly
    // desaturate toward neutral skin → drops saturation variance (GATE2).
    const auto lum = albedo.mean(1).clamp_min(0.02F).view({T * T, 1});  // [T^2,1]
    auto ch = (albedo / lum).t().reshape({1, 3, T, T}).contiguous();    // chroma ratio image
    for (int it = 0; it < 14; ++it) ch = nblur(ch);
    auto chf = ch.reshape({3, T * T}).t();                             // [T^2,3]
    chf = (1.0F + (chf - 1.0F) * 0.62F);                               // desaturate toward neutral skin
    albedo = (lum * chf).clamp(0.0F, 1.0F);
  }

  // ---- DETAIL TRANSFER: real high-frequency skin detail from the single SHARPEST view per texel ---
  // The averaged+cleaned base above is clean, delit and relightable — but SMOOTH (averaging across
  // views destroys pores/edges/stubble). Recover crispness by adding the HIGH-PASS of the best view
  // per texel (the most frontal/visible/unspecular observation). Subtracting that view's own low-pass
  // removes its lighting, so only fine DETAIL transfers — relightability of the base is preserved.
  if (detail_weight > 0.0F) {
    namespace Fc = torch::nn::functional;
    const auto Wst = torch::stack(w_l, 0);                                   // [N,T^2]
    const auto Ost = torch::stack(obs_l, 0);                                 // [N,T^2,3]
    const auto bestv = std::get<1>(Wst.max(0));                              // [T^2] sharpest view
    const auto bestw = std::get<0>(Wst.max(0)).clamp(0.0F, 1.0F).unsqueeze(1);  // its confidence
    const auto sharp = Ost.gather(0, bestv.view({1, TT, 1}).expand({1, TT, 3})).squeeze(0);  // [T^2,3]
    // LUMINANCE-ONLY detail: pores/edges live in luminance, not colour. Adding per-channel detail
    // would re-inject the very white-balance chroma noise we just smoothed away — so high-pass the
    // best view's LUMINANCE and add it equally to all channels.
    const auto slum = sharp.mean(1, /*keepdim=*/true);                       // [T^2,1]
    auto dk1 = torch::tensor({1.F, 4.F, 6.F, 4.F, 1.F}, albedo.options());
    auto dk2 = torch::outer(dk1, dk1);
    dk2 = dk2 / dk2.sum();
    const auto dka = dk2.view({1, 1, 5, 5});
    const auto simg = slum.t().reshape({1, 1, T, T}).contiguous();
    const auto sblur = Fc::conv2d(simg, dka, Fc::Conv2dFuncOptions().padding(2).groups(1));
    const auto detail = (simg - sblur).reshape({1, T * T}).t();              // [T^2,1] lum high-pass
    albedo = (albedo + detail_weight * detail * bestw).clamp(0.0F, 1.0F);    // broadcast to 3 ch
  }

  // WHITE BALANCE: casual indoor/shade photos leave a cool (blue) cast that survives delighting → the
  // skin reads purple-grey. Gray-world the texture toward a natural warm skin reference so the tone is
  // right (bounded gain; only shifts global colour — identity/detail/luminance untouched).
  {
    const auto skin = albedo.index({valid > 0.5F});                         // [P,3] valid texels
    const auto cur = skin.mean(0).clamp_min(0.05F);
    const auto target = torch::tensor({0.72F, 0.645F, 0.605F}, albedo.options());  // natural skin (chroma~0.16)
    const auto gain = (target / cur).clamp(0.6F, 1.7F);
    albedo = (albedo * gain.view({1, 3})).clamp(0.0F, 1.0F);
  }

  // Per-texel 3D surface position (barycentric on the rest mesh) — lets the caller render a
  // TEXTURE-RESOLUTION point cloud (one splat per texel) instead of a vertex-count-limited one.
  pos_out = (rest_verts.to(device).index_select(0, geomv).reshape({TT, 3, 3}) * bary.unsqueeze(2))
                .sum(1);  // [T^2,3]
  mask_out = valid.view({T, T});

  // ---- per-texel photometric normals (photometric stereo on the UV map) ----
  // Under the linear (order-1) part of each frame's recovered SH light, O/albedo = a_f + g_f·n.
  // Stack over frames & channels and solve a weighted 3×3 normal equation per texel for n — the
  // multi-illumination diversity (the thesis) is exactly what makes the normal observable.
  {
    const auto opts = albedo.options();
    const auto B = ncg::recon::sh_basis(torch::eye(3, opts));           // [3,9] basis at the 3 axes
    const auto b0 = B.index({0, 0});                                    // DC term (axis-invariant)
    const auto M = B.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});  // [3,3]
    const auto obs = torch::stack(obs_l, 0);                            // [N,T^2,3]
    const auto w = torch::stack(w_l, 0);                                // [N,T^2]
    const auto L1 = ir.lights.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                     torch::indexing::Slice(1, 4)});    // [N,3ch,3k] order-1 coeffs
    const auto g = torch::einsum("ak,fck->fca", {M, L1});              // [N,3ch,3axis]
    const auto a_dc = ir.lights.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) * b0;  // [N,3ch]
    const auto ratio = obs / albedo.unsqueeze(0).clamp_min(0.05F);     // [N,T^2,3]
    const auto y = (ratio - a_dc.unsqueeze(1)).permute({0, 2, 1});     // [N,3ch,T^2]
    const auto GG = torch::einsum("fca,fcb->fab", {g, g});             // [N,3,3]
    const auto ATA = torch::einsum("ft,fab->tab", {w, GG});           // [T^2,3,3]
    const auto gy = torch::einsum("fca,fct->fat", {g, y});            // [N,3,T^2]
    const auto ATy = torch::einsum("ft,fat->ta", {w, gy});           // [T^2,3]
    const auto ridge = torch::eye(3, opts).unsqueeze(0) * 1e-2F;
    auto n = torch::linalg_solve(ATA + ridge, ATy.unsqueeze(2)).squeeze(2);  // [T^2,3] object-space
    n = torch::nan_to_num(n);
    n = n / n.norm(2, 1, true).clamp_min(1e-6F);

    // Convert object-space normals to TANGENT space (what glTF normalTexture expects) using the
    // canonical TBN frame per texel: geometric normal + UV tangent + bitangent.
    ncg::mesh::TriMesh rm;
    rm.vertices = rest_verts.to(at::kCPU);
    rm.faces = model.faces().to(at::kCPU);
    const auto vnorm = ncg::mesh::compute_vertex_normals(rm).to(device);            // [V,3]
    const auto vtan = ncg::recon::compute_uv_tangents(rest_verts, model.faces(), model.uv_coords(),
                                                      model.uv_faces()).to(device);  // [V,3]
    auto bw = [&](const torch::Tensor& vv) {  // barycentric gather per texel -> [T^2,3]
      return (vv.index_select(0, geomv).reshape({TT, 3, 3}) * bary.unsqueeze(2)).sum(1);
    };
    auto ng = bw(vnorm);
    ng = ng / ng.norm(2, 1, true).clamp_min(1e-6F);
    gnrm_out = ng.clone();  // per-texel object-space geometric normal (for shaded preview render)
    auto tg = bw(vtan);
    tg = tg - ng * (ng * tg).sum(1, true);  // Gram-Schmidt orthogonalize
    tg = tg / tg.norm(2, 1, true).clamp_min(1e-6F);
    const auto bg = torch::cross(ng, tg, 1);
    const auto nt = torch::stack({(n * tg).sum(1), (n * bg).sum(1), (n * ng).sum(1)}, 1);  // tangent
    const auto flat = torch::tensor({0.5F, 0.5F, 1.0F}, opts).view({1, 3});
    auto nmap = (nt * 0.5F + 0.5F) * valid.unsqueeze(1) + flat * (1.0F - valid.unsqueeze(1));
    normal_out = nmap.view({T, T, 3});
  }
  return albedo.view({T, T, 3});
}

// Trains an animatable Gaussian avatar from a directory of video frames of one person. Each frame
// is run through NLF to get its SMPL-X pose + a solved camera; fit_avatar then optimizes a single
// canonical cloud (anisotropic splats, per-frame exposure, D-SSIM) so it reproduces every posed
// frame. Multi-pose casual video thus becomes multi-view evidence for one avatar — the path from a
// projected-color mannequin to a real likeness. Renders fit-check + novel-view turntable frames.
//   ncg_cli avatar --frames dir/ --weights nlf.torchscript --smplx model.safetensors \
//                  [--max-frames 60 --res 288 --iters 3000 --out-prefix rock_avatar]
// `attribute` — BLOCKER #1 / M1: multi-modal anchor attribution. From ONE reference image build a
// signature {ArcFace face, NLF-β body shape, appearance colour histogram} at inference (no per-subject
// training), then score EVERY frame's subject with whichever cues are visible (partial-cue). The body
// + appearance cues attribute PROFILE/BACK/faceless frames (where ArcFace can't) to the subject, while
// other people are rejected — the contamination + faceless-frame fix the runway needed. Emits per-frame
// w_subject. (Temporal track-stitching is the next increment; per-frame multi-cue lands first.)
int cmd_attribute(const ncg::app::Args& args) {
  namespace fs = std::filesystem;
  const auto ndev = ncg::cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
  auto nlf = ncg::body::Nlf::load(args.require("nlf"), ndev, {});
  auto det = ncg::body::FaceDetector::load(args.require("detector"), at::kCPU);  // traced CPU weights
  auto arc = ncg::body::ArcFace::load(args.require("arcface"), at::kCPU);

  // Appearance descriptor: normalized 8-bin/channel RGB histogram over the subject's projected bbox.
  auto color_hist = [](const torch::Tensor& img_cpu, const torch::Tensor& v2d) -> torch::Tensor {
    const int64_t H = img_cpu.size(1), W = img_cpu.size(2);
    const auto vc = v2d.to(at::kCPU);
    const int x0 = std::clamp<int>((int)vc.select(1, 0).min().item<float>(), 0, (int)W - 2);
    const int x1 = std::clamp<int>((int)vc.select(1, 0).max().item<float>(), x0 + 1, (int)W);
    const int y0 = std::clamp<int>((int)vc.select(1, 1).min().item<float>(), 0, (int)H - 2);
    const int y1 = std::clamp<int>((int)vc.select(1, 1).max().item<float>(), y0 + 1, (int)H);
    const auto crop = img_cpu.slice(1, y0, y1).slice(2, x0, x1).clamp(0, 1);  // [3,h,w]
    std::vector<torch::Tensor> hs;
    for (int c = 0; c < 3; ++c) hs.push_back(torch::histc(crop[c], 8, 0.0, 1.0));
    auto h = torch::cat(hs);
    return h / h.sum().clamp_min(1.0);  // [24]
  };
  auto sig = [&](const std::string& path, torch::Tensor& beta, torch::Tensor& a512,
                 torch::Tensor& hist, float& fscore) {
    const auto img = ncg::io::load_image(path, 3);  // [3,H,W] cpu [0,1]
    const auto pred = nlf.detect(img.to(ndev));
    beta = pred.params.betas.reshape({-1}).to(at::kCPU);
    hist = color_hist(img, pred.vertices2d);
    const auto faces = det.detect(img, 0.5F);
    fscore = 0.0F;
    a512 = torch::Tensor{};
    if (!faces.empty()) { a512 = arc.embed(img, faces[0]); fscore = faces[0].score; }
  };

  // Gather frames first — needed so the reference can be auto-selected from them if necessary.
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(args.require("frames"))) {
    const auto x = e.path().extension().string();
    if (x == ".jpg" || x == ".jpeg" || x == ".png" || x == ".JPG") paths.push_back(e.path().string());
  }
  std::sort(paths.begin(), paths.end());
  NCG_CHECK(!paths.empty(), "attribute: no image frames found in --frames dir");

  // ---- reference signature ----
  // Use --ref when it yields a valid NLF detection; otherwise (no --ref, or a --ref with no
  // detectable person, e.g. a frame where the subject is occluded) auto-pick the frame with the
  // strongest face detection — the most frontal/clear view of the subject — so a bad reference can
  // never abort the run. NLF throws when it finds no person; we treat that as "skip this candidate".
  torch::Tensor rb, ra, rh; float rf = 0.0F;
  std::string ref_used;
  const std::string ref_arg = args.get("ref", "");
  if (!ref_arg.empty()) {
    try { sig(ref_arg, rb, ra, rh, rf); ref_used = ref_arg; }
    catch (const std::exception& e) {
      NCG_LOG_WARN("attribute: --ref '{}' has no detectable person ({}); auto-selecting a reference",
                   ref_arg, e.what());
    }
  }
  if (!rb.defined()) {  // scan frames for the clearest face to anchor the signature
    float best = -1.0F;
    for (const auto& p : paths) {
      torch::Tensor b, a, h; float fsc;
      try { sig(p, b, a, h, fsc); } catch (const std::exception&) { continue; }
      if (fsc > best) { best = fsc; rb = b; ra = a; rh = h; rf = fsc; ref_used = p; }
      if (best > 0.9F) break;  // a strong frontal face is a good enough anchor; stop scanning
    }
    NCG_CHECK(rb.defined(), "attribute: no frame yielded a valid detection to use as a reference");
  }
  NCG_LOG_INFO("attribute: reference = {} (face cue {}, score {:.2f}, |β|={:.2f})", ref_used,
               ra.defined() ? "present" : "absent", rf, rb.norm().item<float>());

  const float kappa = args.get_float("kappa", 10.0F), tau = args.get_float("tau", 0.45F);
  const float bscale = args.get_float("beta-scale", 2.0F);
  int kept = 0, faceless_kept = 0, rejected = 0;
  std::vector<float> ws;
  for (const auto& p : paths) {
    torch::Tensor b, a, h; float fsc;
    try { sig(p, b, a, h, fsc); } catch (const std::exception&) { ws.push_back(0); continue; }
    // partial-cue score: face (if visible) + body-shape + appearance.
    float num = 0, den = 0;
    bool has_face = a.defined() && ra.defined();
    if (has_face) { num += fsc * (a.dot(ra).item<float>()); den += fsc; }  // ArcFace cosine (L2-normed)
    const float sb = std::exp(-(b - rb).norm().item<float>() / bscale);    // body-shape similarity
    num += 1.0F * sb; den += 1.0F;
    const float sa = torch::minimum(h, rh).sum().item<float>();            // histogram intersection
    num += 1.0F * sa; den += 1.0F;
    const float ai = num / std::max(den, 1e-6F);
    const float w = 1.0F / (1.0F + std::exp(-kappa * (ai - tau)));
    ws.push_back(w);
    if (w > 0.5F) { ++kept; if (!has_face) ++faceless_kept; } else ++rejected;
  }
  // E7-style summary + GATE numbers.
  ncg::io::save_npy(args.get("out", "w_subject.npy"),
                    torch::tensor(ws, at::TensorOptions().dtype(at::kFloat)).contiguous());
  NCG_LOG_INFO("attribute(GATE M1): {} frames -> {} subject ({} of them FACELESS, kept via body+app), "
               "{} rejected (contamination). w_subject -> {}", paths.size(), kept, faceless_kept,
               rejected, args.get("out", "w_subject.npy"));
  return 0;
}

// `mvbench` — M4 CONTROLLED robustness benchmark (the paper's money figure). From a known synthetic
// avatar we render a clean multi-view set (the CEILING), then inject CONTROLLED contamination — frames
// of a DIFFERENT person (different shape + inverted appearance) and motion-blurred + camera-jittered
// frames — and fit under an ablation of the defences: naive / +robust(C2) / +M1 attribution / +M3
// confidence-blur / +ALL. Every condition's held-out-view PSNR-vs-true-subject is measured, so the
// table reads as "how much of the gap to the clean ceiling each channel recovers". Fully synthetic =
// ground truth is exact and the result is verifiable, with no dependence on scavenged real data.
int cmd_mvbench(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "mvbench requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "mvbench"));
  const auto prefix = (rec.dir() / "mvb").string();
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  const auto dopt = at::TensorOptions().device(device).dtype(at::kFloat);
  const int res = args.get_int("res", 256);
  const int Ns = args.get_int("subject-views", 24);   // clean subject views
  const int Kc = args.get_int("contam-views", 8);      // wrong-person contaminant frames
  const int Kb = args.get_int("blur-views", 8);        // motion-blurred + jittered subject frames
  const float scale = args.get_float("scale", 0.008F);
  const float radius = args.get_float("radius", 2.4F), elev = 10.0F, fov = 50.0F;

  // ---- ground-truth SUBJECT (neutral shape, smooth-gradient albedo) ----
  ncg::body::SmplxParams rest;
  rest.betas = torch::zeros({1, model.num_betas()}, dopt);
  rest.pose_aa = torch::zeros({1, model.num_joints(), 3}, dopt);
  rest.transl = torch::zeros({1, 3}, dopt);
  const auto v_subj = model.forward(rest).vertices.squeeze(0);  // [V,3]
  const int64_t V = v_subj.size(0);
  const auto mn = std::get<0>(v_subj.min(0)), mx = std::get<0>(v_subj.max(0));
  const auto A_subj = ((v_subj - mn) / (mx - mn).clamp_min(1e-4)).clamp(0.05, 0.95);  // [V,3]
  const auto gt_subj = ncg::recon::gaussians_on_body(
      v_subj, scale, A_subj, ncg::recon::per_vertex_scale(v_subj.to(at::kCPU), 0.75).to(device));
  const auto center = gt_subj.positions.mean(0);
  const auto cams = ncg::runtime::orbit_trajectory(center, radius, elev, Ns, fov, res, res, device);

  // ---- CONTAMINANT: a different person — different shape (betas) + inverted appearance ----
  ncg::body::SmplxParams r2 = rest;
  r2.betas = torch::full({1, model.num_betas()}, 1.5F, dopt);  // clearly different body
  const auto v_cont = model.forward(r2).vertices.squeeze(0);
  // A different person's colouring: a strong colour CAST (darker R/G, bluer) gives a DISTINCT colour
  // histogram. Spatial inversion (1-A) would NOT — a histogram is permutation-invariant to layout, so
  // it'd leave the distribution unchanged and the M1 appearance cue blind. The cast shifts the actual
  // per-channel distribution, which is what attribution keys on.
  const auto cast = torch::tensor({0.45F, 0.6F, 1.25F}, dopt).view({1, 3});
  const auto A_cont = (A_subj * cast + 0.04).clamp(0.05, 0.95);
  const auto gt_cont = ncg::recon::gaussians_on_body(
      v_cont, scale, A_cont, ncg::recon::per_vertex_scale(v_cont.to(at::kCPU), 0.75).to(device));
  const auto center_c = gt_cont.positions.mean(0);

  // Separable Gaussian blur of a [3,H,W] image (for the motion-blur frames).
  auto gblur = [&](const torch::Tensor& img, double s) {
    namespace Fn = torch::nn::functional;
    const auto o = img.options();
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * s)));
    const auto x = torch::arange(-r, r + 1, o);
    auto k = torch::exp(-0.5 * (x / s).pow(2));
    k = k / k.sum();
    auto im = img.unsqueeze(0);
    im = Fn::conv2d(im, k.view({1, 1, 1, -1}).expand({3, 1, 1, k.size(0)}).contiguous(),
                    Fn::Conv2dFuncOptions().padding({0, r}).groups(3));
    im = Fn::conv2d(im, k.view({1, 1, -1, 1}).expand({3, 1, k.size(0), 1}).contiguous(),
                    Fn::Conv2dFuncOptions().padding({r, 0}).groups(3));
    return im.squeeze(0);
  };
  // Foreground colour histogram (8 bins/channel over alpha>0 pixels) — the M1 appearance cue.
  auto fg_hist = [&](const torch::Tensor& img, const torch::Tensor& alpha) {
    const auto m = (alpha.reshape({-1}) > 0.05F);
    std::vector<torch::Tensor> hs;
    for (int c = 0; c < 3; ++c)
      hs.push_back(torch::histc(img[c].reshape({-1}).masked_select(m), 8, 0.0, 1.0));
    auto h = torch::cat(hs);
    return h / h.sum().clamp_min(1.0);
  };

  // ---- assemble the contaminated training set (+ provenance + M1 attribution weight) ----
  std::vector<ncg::fit::AvatarFrame> frames;  // full set
  std::vector<int> kind;                       // 0 clean, 1 contaminant, 2 blurred
  std::vector<torch::Tensor> hists;            // per-frame appearance descriptor
  auto add = [&](const ncg::runtime::Camera& cam, const ncg::recon::GaussianCloud& cloud, int knd,
                 bool blur) {
    auto out = ncg::runtime::render_soft_aniso(cloud, cam);
    auto tgt = out.image.detach();
    if (blur) tgt = gblur(tgt, args.get_float("blur-sigma", 2.5F)).detach();
    ncg::fit::AvatarFrame fr;
    fr.pose_aa = torch::zeros({model.num_joints(), 3}, dopt);
    fr.transl = torch::zeros({3}, dopt);
    fr.camera = cam;
    fr.target = tgt;
    hists.push_back(fg_hist(tgt, out.alpha.detach()));
    frames.push_back(std::move(fr));
    kind.push_back(knd);
  };
  for (int i = 0; i < Ns; ++i) add(cams[i], gt_subj, 0, false);                       // clean subject
  const auto ccams = ncg::runtime::orbit_trajectory(center_c, radius, elev, Kc, fov, res, res, device);
  for (int i = 0; i < Kc; ++i) add(ccams[i], gt_cont, 1, false);                       // wrong person
  for (int i = 0; i < Kb; ++i) {                                                       // blurred + jittered
    auto cam = cams[(i * 3) % Ns];
    cam.t = cam.t + torch::randn_like(cam.t) * args.get_float("jitter", 0.04F);        // camera misalignment
    add(cam, gt_subj, 2, true);
  }
  // POSE-MISESTIMATED subject frames (M2's target): a sharp, correct-person frame whose CAMERA POSE is
  // wrong (rendered from the true camera, but the fit is GIVEN a mis-estimated one — the analogue of
  // NLF pose noise). M1 (same person → high w_subject) and M3 (sharp → high quality) leave these
  // untouched; only M2's joint pose factorization (per-frame camera bundle-adjustment) recovers them.
  const int Kp = args.get_int("pose-views", 6);
  for (int i = 0; i < Kp; ++i) {
    const float az = 360.0F * (static_cast<float>(i) + 0.5F) / static_cast<float>(Kp);
    const auto truecam = ncg::runtime::Camera::orbit(center, radius, az, elev, fov, res, res, device);
    const float daz = ((i % 2) ? 1.0F : -1.0F) * args.get_float("pose-jitter-deg", 8.0F);
    const auto badcam =
        ncg::runtime::Camera::orbit(center, radius, az + daz, elev + 0.3F * daz, fov, res, res, device);
    const auto out = ncg::runtime::render_soft_aniso(gt_subj, truecam);  // TRUE-pose target
    const auto tgt = out.image.detach();
    ncg::fit::AvatarFrame fr;
    fr.pose_aa = torch::zeros({model.num_joints(), 3}, dopt);
    fr.transl = torch::zeros({3}, dopt);
    fr.camera = badcam;  // fit STARTS from the wrong camera; M2 must recover it
    fr.target = tgt;
    hists.push_back(fg_hist(tgt, out.alpha.detach()));
    frames.push_back(std::move(fr));
    kind.push_back(3);
  }
  const int64_t Ftot = static_cast<int64_t>(frames.size());

  // M1 attribution weight: appearance-histogram similarity to the reference (clean frame 0).
  const float kappa = args.get_float("attrib-kappa", 12.0F), tau = args.get_float("attrib-tau", 0.55F);
  std::vector<float> wsub(static_cast<size_t>(Ftot), 1.0F);
  {
    const auto ref = hists[0];
    int sup = 0;
    for (int64_t i = 0; i < Ftot; ++i) {
      const float sim = torch::minimum(hists[static_cast<size_t>(i)], ref).sum().item<float>();
      wsub[static_cast<size_t>(i)] = 1.0F / (1.0F + std::exp(-kappa * (sim - tau)));
      if (wsub[static_cast<size_t>(i)] < 0.5F) ++sup;
    }
    NCG_LOG_INFO("mvbench: {} frames ({} clean + {} contaminant + {} blurred + {} pose-bad); M1 "
                 "suppresses {} (w_subject<0.5)", Ftot, Ns, Kc, Kb, Kp, sup);
  }

  // M3 quality weight: per-frame sharpness (variance-of-Laplacian), normalized — the blurred frames
  // score low and so (a) are sampled less and (b) drive a wider confidence-blur. This is the per-frame
  // confidence M3 consumes; it is ORTHOGONAL to M1's who-channel (a sharp wrong-person frame still has
  // high quality but low w_subject; a blurred subject frame the reverse).
  std::vector<float> wqual(static_cast<size_t>(Ftot), 1.0F);
  {
    const auto lapk =
        torch::tensor({0.F, 1.F, 0.F, 1.F, -4.F, 1.F, 0.F, 1.F, 0.F}, dopt).view({1, 1, 3, 3});
    std::vector<float> sv(static_cast<size_t>(Ftot));
    float lo = 1e30F, hi = -1e30F;
    for (int64_t i = 0; i < Ftot; ++i) {
      const auto g = frames[static_cast<size_t>(i)].target.mean(0, true).unsqueeze(0);  // [1,1,H,W]
      const auto lp = torch::nn::functional::conv2d(
          g, lapk, torch::nn::functional::Conv2dFuncOptions().padding(1));
      sv[static_cast<size_t>(i)] = lp.var().item<float>();
      lo = std::min(lo, sv[static_cast<size_t>(i)]);
      hi = std::max(hi, sv[static_cast<size_t>(i)]);
    }
    for (int64_t i = 0; i < Ftot; ++i)
      wqual[static_cast<size_t>(i)] =
          (hi > lo) ? 0.2F + 0.8F * (sv[static_cast<size_t>(i)] - lo) / (hi - lo) : 1.0F;
  }

  const auto cov = torch::ones({V}, dopt);
  const auto init_gray = torch::full({V, 3}, 0.5F, dopt);  // appearance must be LEARNED from frames
  const int iters = args.get_int("iters", 1200);
  // Held-out clean-subject view BETWEEN training azimuths — the generalization metric.
  const auto hc = ncg::runtime::Camera::orbit(center, radius, 360.0F / Ns / 2.0F, elev, fov, res, res, device);
  const auto gh = ncg::runtime::render_soft_aniso(gt_subj, hc);
  const auto hmask = (gh.alpha.detach() > 0.05F).to(at::kFloat);
  ncg::io::save_png(prefix + "_gt_held.png", gh.image.detach());

  // One ablation condition: fit a subset under given defences, return held-out PSNR vs true subject.
  auto run = [&](const std::vector<ncg::fit::AvatarFrame>& fr, bool attrib, bool quality, bool robust,
                 bool blur, bool pose, bool frobust, const char* name) {
    std::vector<ncg::fit::AvatarFrame> f = fr;
    for (size_t i = 0; i < f.size(); ++i)
      f[i].weight = (attrib ? wsub[i] : 1.0F) * (quality ? wqual[i] : 1.0F);
    ncg::fit::AvatarFitConfig cfg;
    cfg.iterations = iters;
    cfg.per_view_exposure = false;
    cfg.densify = false;
    cfg.robust = robust;
    cfg.conf_blur = blur;
    cfg.refine_pose = pose;  // joint camera bundle-adjustment baseline (shown unstable)
    cfg.pose_reg = args.get_float("pose-reg", 2.0F);
    cfg.lr_pose = args.get_float("lr-pose", 3e-3F);
    cfg.pose_refine_from = static_cast<int>(args.get_float("pose-warmup", 0.5F) * iters);
    cfg.frame_robust = frobust;  // M2: frame-level residual gating (the stable pose channel)
    cfg.frame_robust_from = static_cast<int>(args.get_float("frobust-warmup", 0.4F) * iters);
    cfg.frame_robust_k = args.get_float("frobust-k", 2.0F);
    auto fit = ncg::fit::fit_avatar(model, rest.betas.squeeze(0), f, init_gray, cfg, nullptr, cov);
    const auto fh = ncg::runtime::render_soft_aniso(fit.canonical, hc);
    const double p = ncg::record::psnr(fh.image.detach() * hmask, gh.image.detach() * hmask);
    ncg::io::save_png(prefix + "_" + name + "_held.png", fh.image.detach());
    NCG_LOG_INFO("mvbench[{:>10}]: held-out PSNR = {:.2f} dB", name, p);
    return p;
  };

  // CEILING = clean subject frames only (no contamination); then the contaminated-set ablation.
  //                     frames   attrib quality robust blur   name
  std::vector<ncg::fit::AvatarFrame> clean(frames.begin(), frames.begin() + Ns);
  //                      frames  attr  qual  robust blur  pose  frob   name
  const double p_ceil = run(clean, false, false, false, false, false, false, "ceiling");
  const double p_naive = run(frames, false, false, false, false, false, false, "naive");
  const double p_rob = run(frames, false, false, true, false, false, false, "robust");   // prior C2
  const double p_ba = run(frames, false, false, false, false, true, false, "jointBA");    // unstable
  const double p_m1 = run(frames, true, false, false, false, false, false, "m1");          // who
  const double p_m3 = run(frames, false, true, false, true, false, false, "m3");           // quality
  const double p_m2 = run(frames, false, false, false, false, false, true, "m2");          // frame-robust pose
  // OURS = M1 (who) + M3 (quality+blur) + M2 (frame-level residual gating). The prior per-pixel
  // robust/C2 (over-rejects clean signal) and joint camera BA (free DOF → degenerate over-fit) both
  // HURT — kept as baseline columns. M2-as-frame-gating is the stable pose channel: it only removes
  // the influence of unreconcilable (pose-bad) views, never adds DOF.
  const double p_ours = run(frames, true, true, false, true, false, true, "ours");
  NCG_LOG_INFO("mvbench SUMMARY (ceiling {:.2f} dB, contamination gap -{:.2f}): naive {:.2f} | "
               "robust/C2 {:.2f} | jointBA {:.2f} | M1 {:.2f} | M2 {:.2f} | M3 {:.2f} | "
               "OURS(M1+M2+M3) {:.2f} (recovers {:.0f}% of the gap; robust & jointBA HURT) -> {}",
               p_ceil, p_ceil - p_naive, p_naive, p_rob, p_ba, p_m1, p_m2, p_m3, p_ours,
               (p_naive < p_ceil) ? 100.0 * (p_ours - p_naive) / (p_ceil - p_naive) : 100.0, prefix);
  return 0;
}

// `mvtest` — SELF-CONTAINED data-wall-vs-fit-bug isolator. Render a known-clean avatar (gaussians on
// the rest SMPL-X body, clean per-vertex albedo) from N known orbit cameras → PERFECT multi-view
// input (exact poses, full 360° yaw, single subject, zero blur). Feed those back to fit_avatar
// --densify (init == GT) and measure held-out-view reconstruction. CLEAN → the fit/densify is sound
// and every prior corruption was DATA quality. CORRUPT on perfect input → the bug is in the fit.
int cmd_mvtest(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "mvtest requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "mvtest"));
  const auto prefix = (rec.dir() / "mv").string();
  auto model = ncg::body::SmplxModel::load(args.require("smplx"), device);
  const auto dopt = at::TensorOptions().device(device).dtype(at::kFloat);
  ncg::body::SmplxParams rest;
  rest.betas = torch::zeros({1, model.num_betas()}, dopt);
  rest.pose_aa = torch::zeros({1, model.num_joints(), 3}, dopt);
  rest.transl = torch::zeros({1, 3}, dopt);
  const auto gt_verts = model.forward(rest).vertices.squeeze(0);  // [V,3] rest body
  const int64_t V = gt_verts.size(0);
  torch::Tensor albedo;
  if (args.has("albedo")) albedo = ncg::io::load_npy(args.require("albedo")).to(device, at::kFloat);
  if (!albedo.defined() || albedo.size(0) != V) {  // fallback: smooth spatial gradient (still tests geometry)
    const auto mn = std::get<0>(gt_verts.min(0)), mx = std::get<0>(gt_verts.max(0));
    albedo = ((gt_verts - mn) / (mx - mn).clamp_min(1e-4)).clamp(0.05, 0.95);
  }
  const auto pvs = ncg::recon::per_vertex_scale(gt_verts.to(at::kCPU), 0.75).to(device);
  auto gt = ncg::recon::gaussians_on_body(gt_verts, args.get_float("scale", 0.008F), albedo, pvs);
  const auto center = gt.positions.mean(0);
  const int res = args.get_int("res", 400);
  const int N = args.get_int("views", 36);
  const float radius = args.get_float("radius", 2.4F), elev = 10.0F;
  const auto cams = ncg::runtime::orbit_trajectory(center, radius, elev, N, 50.0F, res, res, device);
  std::vector<ncg::fit::AvatarFrame> frames;
  for (int i = 0; i < N; ++i) {
    ncg::fit::AvatarFrame fr;
    fr.pose_aa = torch::zeros({model.num_joints(), 3}, dopt);
    fr.transl = torch::zeros({3}, dopt);
    fr.camera = cams[i];
    fr.target = ncg::runtime::render_soft_aniso(gt, cams[i]).image.detach();  // PERFECT GT view
    frames.push_back(std::move(fr));
  }
  ncg::fit::AvatarFitConfig cfg;
  cfg.iterations = args.get_int("iters", 1500);
  cfg.densify = args.get_int("densify", 1) != 0;
  cfg.lr_color = 0.0;  // init == GT albedo; test geometry/densify, not colour
  cfg.max_dev = args.get_float("max-dev", 0.012F);
  cfg.min_scale = args.get_float("min-scale", 0.0035F);
  cfg.opacity_floor = args.get_float("opacity-floor", 0.6F);
  cfg.densify_grad = args.get_float("densify-grad", 6e-5F);
  cfg.per_view_exposure = false;  // perfect data — no exposure variance
  cfg.robust = false;
  const auto cov = torch::ones({V}, dopt);
  auto fit = ncg::fit::fit_avatar(model, rest.betas.squeeze(0), frames, albedo, cfg, &rec, cov);
  // Held-out view BETWEEN training azimuths — the real generalization test.
  const auto hc = ncg::runtime::Camera::orbit(center, radius, 360.0F / N / 2.0F, elev, 50.0F, res, res, device);
  const auto gh = ncg::runtime::render_soft_aniso(gt, hc);
  const auto fh = ncg::runtime::render_soft_aniso(fit.canonical, hc);
  const auto mask = (gh.alpha.detach() > 0.05F).to(at::kFloat);
  const double psnr = ncg::record::psnr(fh.image.detach() * mask, gh.image.detach() * mask);
  ncg::io::save_png(prefix + "_gt_held.png", gh.image.detach());
  ncg::io::save_png(prefix + "_fit_held.png", fh.image.detach());
  NCG_LOG_INFO("mvtest: PERFECT multi-view, held-out recon PSNR fit-vs-GT = {:.2f} dB "
               "(N={} views, {} splats, densify={}) -> {}_fit_held.png", psnr, N,
               fit.canonical.size(), cfg.densify, prefix);
  return 0;
}

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

  // M1 attribution (the WHO channel) — optional, enabled when --arcface + --detector are given. Per
  // frame we compute ArcFace face similarity (when a face is visible), NLF-β body-shape similarity,
  // and an appearance-histogram similarity to the album's strongest-face frame, fuse them into a
  // partial-cue w_subject, and FOLD it into the fit weight so contaminating frames (other people on
  // a runway/red-carpet clip) contribute ~0 while faceless-but-matching frames still count. Reuses
  // the per-frame NLF pass below — no extra inference. Disabled => behaviour unchanged.
  const bool attrib = !args.get("arcface", "").empty() && !args.get("detector", "").empty();
  std::optional<ncg::body::FaceDetector> a_det;
  std::optional<ncg::body::ArcFace> a_arc;
  if (attrib) {
    a_det.emplace(ncg::body::FaceDetector::load(args.require("detector"), at::kCPU));
    a_arc.emplace(ncg::body::ArcFace::load(args.require("arcface"), at::kCPU));
    NCG_LOG_INFO("avatar: M1 attribution ON (ArcFace + NLF-β + appearance)");
  }
  auto color_hist = [](const torch::Tensor& img_cpu, const torch::Tensor& v2d) -> torch::Tensor {
    const int64_t H = img_cpu.size(1), W = img_cpu.size(2);
    const auto vc = v2d.to(at::kCPU);
    const int x0 = std::clamp<int>((int)vc.select(1, 0).min().item<float>(), 0, (int)W - 2);
    const int x1 = std::clamp<int>((int)vc.select(1, 0).max().item<float>(), x0 + 1, (int)W);
    const int y0 = std::clamp<int>((int)vc.select(1, 1).min().item<float>(), 0, (int)H - 2);
    const int y1 = std::clamp<int>((int)vc.select(1, 1).max().item<float>(), y0 + 1, (int)H);
    const auto crop = img_cpu.slice(1, y0, y1).slice(2, x0, x1).clamp(0, 1);
    std::vector<torch::Tensor> hs;
    for (int c = 0; c < 3; ++c) hs.push_back(torch::histc(crop[c], 8, 0.0, 1.0));
    auto h = torch::cat(hs);
    return h / h.sum().clamp_min(1.0);  // [24]
  };
  std::vector<torch::Tensor> at_beta, at_arc, at_hist;  // per-kept-frame attribution cues (aligned)
  std::vector<float> at_fsc;                            // per-kept-frame face-detection score

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

  // --identity: recover a clean canonical albedo from inconsistent data via the robust C1/C2 inverse
  // renderer (per-frame SH lighting solved away, clothing-swap/occlusion/wrong-person observations
  // rejected per vertex), instead of naively averaging color across frames (which blurs). This is
  // the route to "dump incoherent footage, get a coherent relightable identity."
  const bool identity = args.get_int("identity", 0) != 0;
  const auto faces_cpu = model.has_faces() ? model.faces().to(at::kCPU) : torch::Tensor();
  std::vector<torch::Tensor> id_obs, id_nrm, id_w;  // per-frame [V,3],[V,3],[V] for the solver
  std::vector<torch::Tensor> id_img, id_v2d;        // per-frame image + scaled v2d (per-texel solve)
  std::vector<torch::Tensor> id_betas;              // per-frame SMPL-X shape (robust personalization)

  std::vector<ncg::fit::AvatarFrame> frames;
  std::vector<float> sharp_raw, yaws;  // E1 sharpness + E7 viewing-azimuth per kept frame
  torch::Tensor betas0;
  torch::Tensor init_colors;
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto img_cpu = ncg::io::load_image(paths[i]);  // CPU copy for the (CPU) face nets
    const auto img_full = img_cpu.to(device);
    auto [img, s] = downscale(img_full);
    ncg::body::NlfPrediction pred;
    try {
      pred = nlf.detect(img_full);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("avatar: NLF failed on {} ({}), skipping", paths[i], e.what());
      continue;
    }
    // In identity mode keep only full-vertex detections so the frame list stays aligned 1:1 with the
    // solver's observations (needed to gate refinement by per-frame consistency).
    if (identity && pred.vertices2d.size(0) != model.num_verts()) continue;
    if (!betas0.defined()) betas0 = pred.params.betas.to(device);

    // Identity mode: collect full-res per-vertex observation, posed normal and visibility for the
    // robust inverse-render solver (same inputs as `ncg_cli delight`).
    if (identity && pred.vertices2d.size(0) == model.num_verts() && faces_cpu.defined()) {
      const auto v2df = pred.vertices2d.to(device);
      ncg::mesh::TriMesh tm;
      tm.vertices = pred.vertices3d.to(at::kCPU);
      tm.faces = faces_cpu;
      id_nrm.push_back(ncg::mesh::compute_vertex_normals(tm).to(device));
      id_obs.push_back(ncg::recon::sample_vertex_colors(img_full, v2df).clamp(0.0, 1.0));
      id_w.push_back(ncg::recon::vertex_visibility(
          v2df, pred.vertices3d.select(1, 2).to(device), static_cast<int64_t>(img_full.size(1)),
          static_cast<int64_t>(img_full.size(2))));
      id_img.push_back(img);                          // downscaled image for per-texel sampling
      id_v2d.push_back(pred.vertices2d.to(device) * s);  // v2d in the downscaled image's pixels
      id_betas.push_back(pred.params.betas.to(device));  // per-frame shape estimate
    }

    ncg::fit::AvatarFrame fr;
    fr.pose_aa = pred.params.pose_aa.squeeze(0).to(device);  // [J,3] camera-frame orientation kept
    fr.transl = pred.params.transl.squeeze(0).to(device);
    fr.target = img;
    const int w = static_cast<int>(img.size(2));
    const int h = static_cast<int>(img.size(1));
    fr.camera = ncg::runtime::solve_pinhole_camera(pred.vertices3d.to(device),
                                                   pred.vertices2d.to(device) * s, w, h);
    // E1 — per-frame sharpness (variance of Laplacian on the downscaled subject image).
    {
      const auto gray = img.mean(0, true).unsqueeze(0);  // [1,1,H,W]
      const auto lapk = torch::tensor({0.F, 1.F, 0.F, 1.F, -4.F, 1.F, 0.F, 1.F, 0.F}, img.options())
                            .view({1, 1, 3, 3});
      const auto lap = torch::nn::functional::conv2d(
          gray, lapk, torch::nn::functional::Conv2dFuncOptions().padding(1));
      sharp_raw.push_back(lap.var().item<float>());
    }
    // E7 — viewing azimuth from the global orientation (Rodrigues on the forward axis).
    {
      const auto aa = pred.params.pose_aa.squeeze(0).select(0, 0).to(at::kCPU);  // [3] global orient
      const float ang = aa.norm().item<float>();
      float yaw = 0.0F;
      if (ang > 1e-6F) {
        const float ax0 = aa[0].item<float>() / ang, ax1 = aa[1].item<float>() / ang,
                    ax2 = aa[2].item<float>() / ang;
        const float c = std::cos(ang), sn = std::sin(ang);
        const float fx = ax1 * sn + ax0 * ax2 * (1 - c);   // (R·[0,0,1]).x
        const float fz = c + ax2 * ax2 * (1 - c);          // (R·[0,0,1]).z
        yaw = std::atan2(fx, fz);
      }
      yaws.push_back(yaw);
    }
    // M1 — attribution cues for this kept frame (reuses the NLF pred; face nets run on the CPU image).
    if (attrib) {
      torch::Tensor a;
      float fsc = 0.0F;
      const auto faces = a_det->detect(img_cpu, 0.5F);
      if (!faces.empty()) { a = a_arc->embed(img_cpu, faces[0]); fsc = faces[0].score; }
      at_arc.push_back(a);  // undefined when no face — partial cue
      at_fsc.push_back(fsc);
      at_beta.push_back(pred.params.betas.reshape({-1}).to(at::kCPU));
      at_hist.push_back(color_hist(img_cpu, pred.vertices2d));
    }
    frames.push_back(std::move(fr));

    if (!init_colors.defined()) {  // seed appearance from the first good frame
      const auto v2d = pred.vertices2d.to(device) * s;
      auto cols = ncg::recon::sample_vertex_colors(img, v2d).clamp(0.0, 1.0);
      const auto vis = ncg::recon::vertex_visibility(v2d, pred.vertices3d.select(1, 2).to(device), h, w);
      init_colors = torch::where(vis.unsqueeze(1) > 0, cols, torch::full_like(cols, 0.6F));
    }
  }
  NCG_CHECK(frames.size() >= 2, "avatar: need >=2 usable frames");

  // ===== EXTRACTION WEIGHTING (E5 pose-consistency + E1 sharpness) + E7 coverage histogram =====
  // Combine per-frame confidence into a sampling weight: sharp, pose-consistent frames contribute
  // more; blurry / pose-jump frames contribute LESS (not dropped). Emit the yaw-coverage histogram.
  if (args.get_int("extract-weight", 1) != 0 && frames.size() == sharp_raw.size()) {
    const int64_t F = static_cast<int64_t>(frames.size());
    // E1: normalize sharpness to [0,1] (relative within this source).
    float smin = 1e30F, smax = -1e30F;
    for (float s : sharp_raw) { smin = std::min(smin, s); smax = std::max(smax, s); }
    // E5: pose-consistency — deviation of each frame's full pose from the per-source MEDIAN pose.
    std::vector<torch::Tensor> poses;
    for (const auto& fr : frames) poses.push_back(fr.pose_aa.reshape({-1}).to(at::kCPU));
    const auto P = torch::stack(poses, 0);                          // [F, J*3]
    const auto medp = std::get<0>(P.median(0));                    // [J*3]
    const auto dev = (P - medp.unsqueeze(0)).norm(2, 1);           // [F] pose deviation
    const float dscale = std::max(1e-3F, dev.median().item<float>());
    int dropped = 0;
    float wsum = 0;
    for (int64_t i = 0; i < F; ++i) {
      const float sn = (smax > smin) ? (sharp_raw[i] - smin) / (smax - smin) : 1.0F;  // E1
      const float pc = std::exp(-0.5F * std::pow(dev[i].item<float>() / dscale, 2.0F));  // E5
      float wgt = (0.3F + 0.7F * sn) * pc;  // keep a floor on sharpness so no frame is fully zeroed
      if (wgt < 0.05F) { ++dropped; }       // effectively-dropped (rare true-garbage) — logged, not gated
      frames[static_cast<size_t>(i)].weight = wgt;
      wsum += wgt;
    }
    // E7: yaw-coverage histogram (12 bins over [-180,180]) — the GO/NO-GO multi-view artifact.
    int hist[12] = {0};
    for (float y : yaws) {
      int b = static_cast<int>((y + static_cast<float>(M_PI)) / (2 * static_cast<float>(M_PI)) * 12);
      hist[std::clamp(b, 0, 11)]++;
    }
    int occupied = 0;
    for (int b = 0; b < 12; ++b) if (hist[b] > 0) ++occupied;
    std::string hs;
    for (int b = 0; b < 12; ++b) hs += std::to_string(hist[b]) + (b < 11 ? "," : "");
    NCG_LOG_INFO("extract: {} frames, mean weight {:.2f}, {} low-weight; yaw-coverage {}/12 bins "
                 "occupied [{}] (E7 GO/NO-GO)", F, wsum / F, dropped, occupied, hs);
  }

  // ===== M1 ATTRIBUTION FOLD: multiply the WHO channel into each frame's fit weight =====
  // Reference = the strongest-face frame (clearest view of the album's dominant subject). Off-subject
  // frames (different β + appearance, and a non-matching face when one is visible) get w_subject -> 0
  // and so stop poisoning the shared canonical appearance. Applied AFTER E1/E5 so it composes.
  if (attrib && at_fsc.size() == frames.size() && !frames.empty()) {
    size_t ri = 0;
    for (size_t i = 1; i < at_fsc.size(); ++i)
      if (at_fsc[i] > at_fsc[ri]) ri = i;
    const auto rb = at_beta[ri];
    const auto ra = at_arc[ri];
    const auto rh = at_hist[ri];
    const float kappa = args.get_float("attrib-kappa", 10.0F), tau = args.get_float("attrib-tau", 0.45F);
    const float bscale = args.get_float("attrib-beta-scale", 2.0F);
    int suppressed = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
      float num = 0, den = 0;
      if (at_arc[i].defined() && ra.defined()) {
        num += at_fsc[i] * at_arc[i].dot(ra).item<float>();
        den += at_fsc[i];
      }
      const float sb = std::exp(-(at_beta[i] - rb).norm().item<float>() / bscale);
      num += sb; den += 1.0F;
      const float sa = torch::minimum(at_hist[i], rh).sum().item<float>();
      num += sa; den += 1.0F;
      const float ai = num / std::max(den, 1e-6F);
      const float ws = 1.0F / (1.0F + std::exp(-kappa * (ai - tau)));
      frames[i].weight *= ws;
      if (ws < 0.5F) ++suppressed;
    }
    NCG_LOG_INFO("avatar(M1 fold): ref frame #{} (face {:.2f}); {} of {} frames suppressed as "
                 "off-subject (w_subject<0.5)", ri, at_fsc[ri], suppressed, frames.size());
  }
  NCG_LOG_INFO("avatar: training on {} frames at {}px", frames.size(), res);

  ncg::recon::GaussianCloud canonical;
  torch::Tensor binding;
  if (identity) {
    // The award-worthy path on incoherent data: jointly solve a single canonical albedo + per-frame
    // SH lighting with robust per-observation consistency (C1/C2). The face/skin (consistent across
    // all footage) anchor a clean identity; outfits/occlusion/wrong-person frames are down-weighted.
    NCG_CHECK(id_obs.size() >= 2, "avatar --identity: need >=2 frames with full-res vertices");
    ncg::recon::InverseRenderConfig ic;
    ic.iterations = args.get_int("iters", 80);
    ic.robust = true;
    const auto ir = ncg::recon::solve_inverse_render(torch::stack(id_obs, 0), torch::stack(id_nrm, 0),
                                                     torch::stack(id_w, 0), ic);
    // Gauge-fix the albedo (identifiable up to a per-channel scale): match its mean to the robust
    // mean observed color so it displays at a sensible brightness.
    auto albedo = torch::nan_to_num(ir.albedo).clamp_min(0.0F);
    const auto obs_mean = torch::stack(id_obs, 0).mean(0).mean(0).clamp_min(1e-3F);  // [3]
    const auto alb_mean = albedo.mean(0).clamp_min(1e-3F);                          // [3]
    albedo = (albedo * (obs_mean / alb_mean).view({1, 3})).clamp(0.0F, 1.0F);
    // Use the solver's per-vertex uncertainty: poorly-constrained vertices (rarely/never seen
    // consistently in incoherent data) have garbage albedo and produce bright spikes. Blend their
    // albedo toward the neutral mean and fade their opacity by a confidence ∝ precision.
    auto conf = torch::nan_to_num(ir.precision).mean(1, true).clamp_min(0.0F);       // [V,1]
    conf = conf / (conf + conf.median().clamp_min(1e-8F));                           // [V,1] in [0,1)
    albedo = albedo * conf + obs_mean.view({1, 3}) * (1.0F - conf);
    NCG_LOG_INFO("avatar --identity: recovered canonical albedo from {} frames (mean consistency {:.2f})",
                 id_obs.size(), ir.consistency.mean().item<double>());

    // Personalize geometry first (robust median of per-frame SMPL-X shape) so the UV solve and the
    // body share the same rest mesh — and so the per-texel normals get the right tangent frame.
    if (id_betas.size() >= 3) betas0 = std::get<0>(torch::stack(id_betas, 0).median(0));
    ncg::body::SmplxParams rp;
    rp.betas = betas0;
    rp.pose_aa = torch::zeros({1, model.num_joints(), 3}, betas0.options());
    apply_apose(rp.pose_aa, args.get_float("apose", 1.0F));  // arms down (not a T-pose mannequin)
    rp.transl = torch::zeros({1, 3}, betas0.options());
    const auto rest_v = model.forward(rp).vertices.squeeze(0);

    // UV albedo texture. Default = bake the robust PER-VERTEX albedo (the clean turntable result)
    // into the texture: coarse but artifact-free. The per-texel solver (--uv-pertexel) is sharper
    // only when the face is well-aligned across frames; on generic SMPL-X faces it smears, so it is
    // off by default until landmark-based face alignment lands.
    if (args.has("uv-texture") && model.has_uv() && !id_img.empty()) {
      const int T = args.get_int("uv-texture", 512);
      const auto pfx = args.get("out-prefix", "avatar");
      if (args.get_int("uv-pertexel", 0) != 0) {
        torch::Tensor uvmask, uvnrm, uvpos, uvgn;
        const auto uvtex = recover_uv_albedo(model, id_img, id_v2d, id_nrm, id_w, T, rest_v, uvmask,
                                             uvnrm, uvpos, uvgn);
        ncg::io::save_png(pfx + "_albedo_uv.png", uvtex.permute({2, 0, 1}).contiguous().detach());
        ncg::io::save_png(pfx + "_normal_uv.png", uvnrm.permute({2, 0, 1}).contiguous().detach());
      } else {
        auto ras = ncg::recon::uv_rasterize(model.uv_coords(), model.uv_faces(), T);
        torch::Tensor uvmask;
        const auto uvtex = ncg::recon::bake_to_uv(ras, albedo, model.faces(), uvmask);  // [T,T,3]
        ncg::io::save_png(pfx + "_albedo_uv.png", uvtex.permute({2, 0, 1}).contiguous().detach());
      }
      NCG_LOG_INFO("avatar --identity: wrote {}x{} UV albedo texture -> {}_albedo_uv.png", T, T, pfx);
    }

    // Build the rigged avatar: SMPL-X body geometry + the robust identity albedo.
    auto pvs = ncg::recon::per_vertex_scale(rest_v, args.get_float("scale_mult", 0.75F));
    pvs = torch::nan_to_num(pvs).clamp(0.004F, 0.02F);  // bound scale: no giant/degenerate splats
    canonical = ncg::recon::gaussians_on_body(rest_v, args.get_float("scale", 0.01F), albedo, pvs);
    canonical.to_(device);
    canonical.opacities = (0.2F + 0.8F * conf.to(device)).clamp(0.0F, 1.0F);  // fade uncertain verts
    binding = torch::arange(model.num_verts(), at::TensorOptions().dtype(at::kLong).device(device));

    // Consistency-gated photometric refinement: the solver tells us which frames are trustworthy;
    // run the photoreal anisotropic fit (+ densification) on just that coherent subset, starting
    // from the clean identity albedo, to sharpen the face/detail without re-muddying on outliers.
    if (args.get_int("refine", 0) != 0) {
      const auto fscore = ir.consistency.mean(1);  // [N] per-frame mean consistency
      const double thr = fscore.median().item<double>();
      std::vector<ncg::fit::AvatarFrame> coherent;
      for (size_t k = 0; k < frames.size() && k < static_cast<size_t>(fscore.size(0)); ++k)
        if (fscore[static_cast<int64_t>(k)].item<double>() >= thr) coherent.push_back(frames[k]);
      NCG_LOG_INFO("avatar --identity --refine: photometric refine on {}/{} coherent frames",
                   coherent.size(), frames.size());
      if (coherent.size() >= 2) {
        ncg::fit::AvatarFitConfig rc;
        rc.iterations = args.get_int("refine_iters", 2500);
        rc.init_scale = args.get_float("scale", 0.012F);
        rc.per_view_exposure = true;
        rc.robust = true;
        rc.densify = args.get_int("densify", 1) != 0;
        rc.log_every = 100;
        auto rr = ncg::fit::fit_avatar(model, betas0, coherent, albedo, rc, &rec);
        canonical = rr.canonical;
        binding = rr.binding;
      }
    }
  } else {
    ncg::fit::AvatarFitConfig cfg;
    cfg.iterations = args.get_int("iters", 3000);
    cfg.init_scale = args.get_float("scale", 0.015F);
    cfg.lambda_dssim = args.get_float("dssim", 0.2F);
    cfg.per_view_exposure = args.get_int("exposure", 1) != 0;
    cfg.robust = args.get_int("robust", 1) != 0;  // C2 robust consistency on by default (mixed data)
    cfg.robust_k = args.get_float("robust_k", 3.0F);
    cfg.conf_blur = args.get_int("conf-blur", 0) != 0;  // M3 confidence-weighted anisotropic-blur loss
    cfg.conf_blur_max = args.get_float("conf-blur-max", 2.5F);
    cfg.log_every = 50;
    cfg.dump_every = args.get_int("dump-every", 500);
    cfg.densify = args.get_int("densify", 0) != 0;
    cfg.refine_pose = args.get_int("refine-pose", 0) != 0;  // bundle-adjust per-frame cameras (video)
    cfg.lr_pose = args.get_float("lr-pose", 2e-3F);
    cfg.pose_reg = args.get_float("pose-reg", 50.0F);
    // Supervision-fix knobs (Part 2) for the multi-angle densify test: anti-floater clamp + min-scale
    // (no speckle) + opacity floor (no dark holes) + optional colour freeze.
    cfg.max_dev = args.get_float("max-dev", 0.012F);
    cfg.min_scale = args.get_float("min-scale", 0.0035F);
    cfg.opacity_floor = args.get_float("opacity-floor", 0.6F);
    cfg.densify_grad = args.get_float("densify-grad", 6e-5F);
    if (args.get_int("freeze-color", 0) != 0) cfg.lr_color = 0.0;  // keep the seeded clean albedo
    const auto cov = torch::ones({init_colors.size(0)}, init_colors.options());  // video sees whole body
    auto result = ncg::fit::fit_avatar(model, betas0, frames, init_colors, cfg, &rec, cov);
    canonical = result.canonical;
    binding = result.binding;
  }

  // Fit-check: render the avatar at frame 0's pose/camera next to the target.
  const auto prefix = args.get("out-prefix", "avatar");
  {
    ncg::body::SmplxParams p0;
    p0.betas = betas0;
    p0.pose_aa = frames[0].pose_aa.unsqueeze(0);
    p0.transl = frames[0].transl.unsqueeze(0);
    const auto vt0 = model.forward(p0).vertex_transforms.squeeze(0);
    const auto posed = ncg::fit::deform_avatar(canonical, vt0, binding);
    const auto out0 = ncg::runtime::render_soft_aniso(posed, frames[0].camera);
    const auto fit = out0.image.detach();
    ncg::io::save_png(prefix + "_fit0.png", fit);
    ncg::io::save_png(prefix + "_tgt0.png", frames[0].target.detach());
    // Body-masked PSNR: the avatar renders the body on a black background while the target has a
    // full scene, so whole-image PSNR is meaningless — measure only inside the rendered silhouette.
    const auto mask = (out0.alpha.detach() > 0.05F).to(at::kFloat);
    NCG_LOG_INFO("avatar: fit0 body-masked PSNR vs target = {:.2f} dB ({:.0f}% body coverage)",
                 ncg::record::psnr(fit * mask, frames[0].target.detach() * mask),
                 100.0 * mask.mean().item<double>());
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

  // ---- engine export: dual representation driven by one SMPL-X skeleton ----
  // (1) Skinned Gaussian .ply: high-fidelity render asset (any GS plugin/viewer) + a .skin sidecar
  //     binding each splat to the skeleton, so the splats follow the physics rig.
  // (2) Rigged (optionally animated) mesh .glb: the universal, physics-ready body that imports into
  //     any engine and drives ragdoll/colliders — the same 55-joint skeleton both share.
  {
    const int64_t Jn = model.num_joints();
    ncg::body::SmplxParams rp;
    rp.betas = betas0;
    rp.pose_aa = torch::zeros({1, Jn, 3}, betas0.options());
    if (identity) apply_apose(rp.pose_aa, args.get_float("apose", 1.0F));  // arms-down rest mesh
    rp.transl = torch::zeros({1, 3}, betas0.options());
    const auto ro = model.forward(rp);
    const auto rest_verts = ro.vertices.squeeze(0);  // [V,3]
    const auto joints = ro.joints.squeeze(0);        // [J,3]
    const auto lbs = model.lbs_weights().to(device);  // [V,J]

    // Per-splat skinning: each Gaussian inherits its bound vertex's top-4 bone influences.
    const auto lbs_g = lbs.index_select(0, binding.defined() && binding.numel() > 0
                                                ? binding
                                                : torch::arange(model.num_verts(),
                                                                binding.options()));
    const auto tk = lbs_g.topk(4, /*dim=*/1);
    auto sw_g = std::get<0>(tk);
    const auto idx_g = std::get<1>(tk);
    sw_g = sw_g / sw_g.sum(1, true).clamp_min(1e-8);
    ncg::mesh::write_gaussian_ply(canonical, prefix + ".ply", idx_g, sw_g);
    NCG_LOG_INFO("avatar: wrote skinned Gaussian splat -> {}.ply (+ .skin, {} splats)", prefix,
                 canonical.size());

    // Rigged mesh .glb with the trained per-vertex appearance (when 1:1) and the SMPL-X rig.
    if (model.has_faces()) {
      ncg::mesh::TriMesh tm;
      tm.vertices = rest_verts.to(at::kCPU);
      tm.faces = model.faces().to(at::kCPU);
      const auto normals = ncg::mesh::compute_vertex_normals(tm);
      const auto vcol =
          (canonical.size() == model.num_verts() ? canonical.colors : init_colors).to(at::kCPU);
      const auto parents = model.parents();
      const auto skin = model.lbs_weights();
      if (args.has("motion")) {
        auto motion = ncg::io::load_npy(args.require("motion")).to(betas0.options());  // [T,J,3]
        const auto quats = aa_to_quat(motion);                                         // [T,J,4]
        const float fps = args.get_float("fps", 24.0F);
        const auto times = torch::arange(motion.size(0), at::kFloat) / fps;
        ncg::mesh::write_glb_animated(rest_verts, model.faces(), normals, vcol, joints, parents,
                                      skin, quats, times, prefix + ".glb");
        NCG_LOG_INFO("avatar: wrote rigged+animated mesh -> {}.glb ({} frames)", prefix,
                     motion.size(0));
      } else {
        ncg::mesh::write_glb_skinned(rest_verts, model.faces(), normals, vcol, joints, parents,
                                     skin, prefix + ".glb");
        NCG_LOG_INFO("avatar: wrote rigged mesh -> {}.glb", prefix);
      }
      // Textured variant: the high-res per-texel albedo + normal map on the UV-mapped rigged mesh,
      // and the baked locomotion clip when --motion is given (photoreal AND animated).
      if (identity && args.has("uv-texture") && model.has_uv()) {
        torch::Tensor tquats, ttimes;
        if (args.has("motion")) {
          const auto m = ncg::io::load_npy(args.require("motion")).to(betas0.options());  // [T,J,3]
          tquats = aa_to_quat(m);
          ttimes = torch::arange(m.size(0), at::kFloat) / args.get_float("fps", 24.0F);
        }
        ncg::mesh::write_glb_textured(rest_verts, model.faces(), normals, model.uv_coords(),
                                      model.uv_faces(), joints, parents, skin,
                                      prefix + "_albedo_uv.png", prefix + "_textured.glb",
                                      prefix + "_normal_uv.png", tquats, ttimes);
        NCG_LOG_INFO("avatar: wrote UV-textured{} rigged mesh -> {}_textured.glb",
                     args.has("motion") ? "+animated" : "", prefix);
      }
    }
  }
  NCG_LOG_INFO("avatar: done -> {}.ply (splats) + {}.glb (rigged mesh) + {}_fit0.png ({} gaussians)",
               prefix, prefix, prefix, canonical.size());
  return 0;
}

// ============================================================================================
// `gate` — Phase A on a real album: detect→embed→robust subject gate→dense landmarks. Reports the
// C2 contamination result (usable subject faces vs rejected other-person faces). CPU by default so
// it never competes with a training GPU. Produces the PhotoBundle stream Phase B will consume.
//   ncg_cli gate --frames dir/ --detector det.ts --facemesh fm.ts --arcface arc.ts
int cmd_gate(const ncg::app::Args& args) {
  const auto device = args.get_int("cuda", 0) != 0 && ncg::cuda_available() ? at::Device(at::kCUDA, 0)
                                                                            : at::Device(at::kCPU);
  namespace fs = std::filesystem;
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(args.require("frames"))) {
    const auto ext = e.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".JPG" || ext == ".JPEG")
      paths.push_back(e.path().string());
  }
  std::sort(paths.begin(), paths.end());
  NCG_CHECK(!paths.empty(), "gate: no images in --frames");

  auto det = ncg::body::FaceDetector::load(args.require("detector"), device);
  auto mesh = ncg::body::FaceMeshNet::load(args.require("facemesh"), device);
  auto arc = ncg::body::ArcFace::load(args.require("arcface"), device);
  ncg::body::AlbumGateModels models{&det, &mesh, &arc};
  ncg::body::AlbumGateConfig cfg;
  cfg.min_det_score = args.get_float("det-score", 0.5F);
  cfg.subject.same_id_cos = args.get_float("same-cos", 0.5F);
  cfg.subject.keep_cos = args.get_float("keep-cos", 0.45F);

  const auto bundles = ncg::body::gate_album(paths, models, cfg);

  int usable = 0, rejected = 0;
  for (const auto& b : bundles) {
    if (b.usable) ++usable;
    rejected += static_cast<int>(b.rejected.size());
  }
  NCG_LOG_INFO("gate: {} photos → {} usable subject faces, {} other/contamination faces rejected",
               paths.size(), usable, rejected);
  // Optional visual audit: dump every kept-subject and rejected face crop so the decisions can be
  // eyeballed (kept should all be the same person; dropped should be others).
  const bool dump = args.has("dump");
  std::string ddir;
  if (dump) { ddir = args.get("dump", "gate_dump"); fs::create_directories(ddir); }
  auto save_crop = [&](const torch::Tensor& img, const torch::Tensor& bbox, const std::string& fn) {
    namespace Fn = torch::nn::functional;
    const int64_t H = img.size(1), W = img.size(2);
    const auto b = bbox.to(at::kCPU);
    const int64_t x0 = std::clamp<int64_t>((int64_t)b[0].item<float>(), 0, W - 1);
    const int64_t y0 = std::clamp<int64_t>((int64_t)b[1].item<float>(), 0, H - 1);
    const int64_t x1 = std::clamp<int64_t>((int64_t)b[2].item<float>(), x0 + 1, W);
    const int64_t y1 = std::clamp<int64_t>((int64_t)b[3].item<float>(), y0 + 1, H);
    auto c = Fn::interpolate(img.slice(1, y0, y1).slice(2, x0, x1).unsqueeze(0),
                             Fn::InterpolateFuncOptions().size(std::vector<int64_t>{128, 128})
                                 .mode(torch::kBilinear).align_corners(false)).squeeze(0);
    ncg::io::save_png(fn, c.clamp(0.0, 1.0));
  };
  int ki = 0, di = 0;
  for (const auto& b : bundles) {
    if (b.usable)
      NCG_LOG_INFO("  [keep] trust {:.3f}  {} ({} pts)", b.w_prior,
                   fs::path(b.path).filename().string(), b.dense_uv.numel() / 2);
    for (const auto& r : b.rejected)
      NCG_LOG_INFO("  [drop:{}] dist {:.3f}  {}", r.reason, r.embed_dist,
                   fs::path(r.path).filename().string());
    if (!dump || (!b.usable && b.rejected.empty())) continue;
    torch::Tensor img;
    try { img = ncg::io::load_image(b.path, 3); } catch (...) { continue; }
    if (b.usable) {
      char nm[64]; std::snprintf(nm, sizeof(nm), "%s/keep_%03d_t%02d.png", ddir.c_str(), ki++,
                                 (int)(b.w_prior * 99));
      save_crop(img, b.subject_bbox, nm);
    }
    for (const auto& r : b.rejected) {
      char nm[64]; std::snprintf(nm, sizeof(nm), "%s/drop_%03d.png", ddir.c_str(), di++);
      save_crop(img, r.bbox, nm);
    }
  }
  if (dump) NCG_LOG_INFO("gate: wrote {} kept + {} dropped face crops to {}/", ki, di, ddir);
  return 0;
}

// ============================================================================================
// `geom` — Phase B on a real album: gate → dense FaceMesh correspondences → solve_geometry
// (identity β + out-of-subspace Δv + observability). Exports the personalized neutral mesh
// (v̄ + shapedirs·β + Δv), the SMPL-X mean for comparison, and the observability map.
//   ncg_cli geom --frames dir/ --detector d.ts --facemesh f.ts --arcface a.ts \
//                --embed facemesh_smplx_embed.safetensors --smplx smplx.safetensors
int cmd_geom(const ncg::app::Args& args) {
  const auto device = at::Device(at::kCPU);  // CPU: gate nets are tiny, solve is CPU; training-safe
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "geom"));
  namespace fs = std::filesystem;
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(args.require("frames"))) {
    const auto ext = e.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".JPG") paths.push_back(e.path().string());
  }
  std::sort(paths.begin(), paths.end());
  NCG_CHECK(!paths.empty(), "geom: no images in --frames");

  // ---- Phase A: gate the album ----
  auto det = ncg::body::FaceDetector::load(args.require("detector"), device);
  auto mesh = ncg::body::FaceMeshNet::load(args.require("facemesh"), device);
  auto arc = ncg::body::ArcFace::load(args.require("arcface"), device);
  ncg::body::AlbumGateModels gm{&det, &mesh, &arc};
  const auto bundles = ncg::body::gate_album(paths, gm, {});

  std::vector<torch::Tensor> lm_l, cf_l;
  std::vector<float> wp;
  for (const auto& b : bundles)
    if (b.usable) { lm_l.push_back(b.dense_uv); cf_l.push_back(b.dense_conf); wp.push_back(b.w_prior); }
  NCG_CHECK(lm_l.size() >= 3, "geom: need >=3 usable subject photos; got {}", lm_l.size());
  const auto landmarks2d = torch::stack(lm_l, 0);                       // [N,K,2]
  const auto conf = torch::stack(cf_l, 0);                             // [N,K]
  const auto w_prior = torch::tensor(wp);                             // [N]

  // ---- assets: FaceMesh→SMPL-X embedding + SMPL-X bases ----
  auto emb = ncg::io::SafeTensors::open(args.require("embed"));
  const auto assoc = emb.view("assoc").clone().to(at::kLong);          // [K]
  const auto bary = emb.view("bary").clone().to(at::kFloat);           // [K,3]
  auto st = ncg::io::SafeTensors::open(args.require("smplx"));
  const auto v_template = st.view("v_template").clone().to(at::kFloat);
  const auto faces = st.view("faces").clone().to(at::kLong);
  const auto id_dirs = st.view("face_id_dirs").clone().to(at::kFloat);
  const auto ex_dirs = st.view("face_expr_dirs").clone().to(at::kFloat);
  const auto lbs_w = st.view("lbs_weights").clone().to(at::kFloat);   // [V,J] rig skin weights
  const auto parents = st.view("parents").clone().to(at::kLong);      // [J]
  const auto Jreg = st.view("J_regressor").clone().to(at::kFloat);    // [J,V]

  NCG_LOG_INFO("geom: {} usable photos, {} dense points; building cotangent Laplacian ({} verts)…",
               lm_l.size(), landmarks2d.size(1), v_template.size(0));
  const auto lap = ncg::geom::cotangent_laplacian(v_template, faces);

  // β from NLF's well-constrained whole-body fit (averaged over the gated photos) — the stable
  // identity. SMPL-X's β is global; face landmarks alone under-constrain it, so we FIX β here and
  // let Phase B solve only the off-subspace Δv. (--nlf optional; without it β is solved from the
  // face landmarks, kept sane by --id-ridge.)
  torch::Tensor beta_fixed, albedo, app_obs, body_betas;
  std::vector<torch::Tensor> nrm_l, vis_l, uv_img, uv_v2d;  // hoisted for the per-texel UV texture
  std::vector<ncg::fit::AvatarFrame> aframes;               // posed frames for the free-splat fit
  const bool densify = args.get_int("densify", 0) != 0;     // Phase-C free-splat / adaptive layer
  const auto ndev = ncg::cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
  if (args.has("nlf")) {
    auto nlf = ncg::body::Nlf::load(args.require("nlf"), ndev, {});
    const auto faces_cpu = faces.to(at::kCPU);
    const int sr = args.get_int("sample-res", 2560);
    std::vector<torch::Tensor> betas, obs_l;
    for (const auto& b : bundles) {
      if (!b.usable) continue;
      torch::Tensor img;
      try { img = ncg::io::load_image(b.path, 3).to(ndev); } catch (const std::exception&) { continue; }
      ncg::body::NlfPrediction pred;
      try { pred = nlf.detect(img); } catch (const std::exception&) { continue; }
      if (pred.vertices2d.size(0) != v_template.size(0)) continue;  // need the full mesh
      betas.push_back(pred.params.betas.reshape({-1}).to(at::kCPU));
      const auto v2d = pred.vertices2d.to(ndev);
      nrm_l.push_back(
          ncg::mesh::compute_vertex_normals({pred.vertices3d.to(at::kCPU), faces_cpu}).to(ndev));
      obs_l.push_back(ncg::recon::sample_vertex_colors(img, v2d).clamp(0.0, 1.0));
      vis_l.push_back(ncg::recon::vertex_visibility(v2d, pred.vertices3d.select(1, 2).to(ndev),
                                                    static_cast<int64_t>(img.size(1)),
                                                    static_cast<int64_t>(img.size(2))));
      // downscaled image + scaled v2d for the high-res per-texel UV texture (the sharp face path).
      const double s = std::min(1.0, static_cast<double>(sr) / std::max(img.size(1), img.size(2)));
      uv_img.push_back(torch::nn::functional::interpolate(
          img.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions()
                                .scale_factor(std::vector<double>{s, s})
                                .mode(torch::kBilinear).align_corners(false)).squeeze(0));
      uv_v2d.push_back(v2d * static_cast<float>(s));
      // Free-splat layer: collect this gated photo as a posed supervision frame (pose+camera+image).
      if (densify) {
        ncg::fit::AvatarFrame fr;
        fr.pose_aa = pred.params.pose_aa.reshape({-1, 3}).to(ndev);  // [J,3]
        fr.transl = pred.params.transl.reshape({-1}).to(ndev);       // [3]
        const int dr = args.get_int("densify-res", 256);
        const double sd = std::min(1.0, static_cast<double>(dr) / std::max(img.size(1), img.size(2)));
        fr.target = torch::nn::functional::interpolate(
            img.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions()
                                  .scale_factor(std::vector<double>{sd, sd})
                                  .mode(torch::kBilinear).align_corners(false)).squeeze(0).clamp(0, 1);
        fr.camera = ncg::runtime::solve_pinhole_camera(
            pred.vertices3d.to(ndev), (v2d * static_cast<float>(sd)),
            static_cast<int>(fr.target.size(2)), static_cast<int>(fr.target.size(1)));
        aframes.push_back(fr);
      }
    }
    if (!betas.empty()) {
      const auto bavg = std::get<0>(torch::stack(betas, 0).median(0)).to(at::kFloat);  // robust avg
      body_betas = bavg.clone();  // whole-body SMPL-X shape for the free-splat fit (fit_avatar)
      const int64_t nid = id_dirs.size(2), m = std::min<int64_t>(bavg.size(0), nid);
      beta_fixed = torch::zeros({nid}, at::kFloat);
      beta_fixed.slice(0, 0, m).copy_(bavg.slice(0, 0, m));
      NCG_LOG_INFO("geom: β fixed from NLF over {} photos (|β|={:.3f})", betas.size(),
                   beta_fixed.norm().item<float>());
    }
    if (!obs_l.empty()) {  // recover relightable per-vertex albedo (C1/C2) for the textured export
      ncg::recon::InverseRenderConfig ic;
      ic.iterations = 60;
      ic.robust = obs_l.size() >= 2;
      albedo = ncg::recon::solve_inverse_render(torch::stack(obs_l, 0), torch::stack(nrm_l, 0),
                                                torch::stack(vis_l, 0), ic)
                   .albedo.clamp(0.0, 1.0)
                   .to(at::kCPU);
      // APPEARANCE observability: how confidently each vertex was actually colored by the photos
      // (sum of per-view visibility). This is the firewall the Phase-E completion gate consumes —
      // the appearance analog of the Phase-B geometry o(v). Saturating sum → [0,1]-ish coverage.
      // Fraction of views that saw each vertex front-facing/unoccluded — discriminates the
      // poorly-seen back/sides (→ completion) from the well-photographed front (→ frozen).
      app_obs = torch::stack(vis_l, 0).to(at::kFloat).mean(0).clamp(0.0, 1.0).to(at::kCPU);  // [V]
      NCG_LOG_INFO("geom: albedo from {} views; appearance coverage min/med/max={:.2f}/{:.2f}/{:.2f}"
                   ", {} verts <0.2 (need completion)", obs_l.size(),
                   app_obs.min().item<float>(),
                   app_obs.median().item<float>(), app_obs.max().item<float>(),
                   (app_obs < 0.2F).sum().item<int64_t>());
    }
  }

  ncg::geom::GeomConfig cfg;
  cfg.iterations = args.get_int("iters", 20);
  cfg.lap_weight = args.get_float("lap", 5.0F);
  // Strong β prior: independent FaceMesh landmarks + an approximate embedding would otherwise drive
  // β to absurd values (which then absorb the off-subspace signal that should go to Δv).
  cfg.id_ridge = args.get_float("id-ridge", 150.0F);  // keep β in the plausible SMPL-X range —
  cfg.expr_ridge = args.get_float("expr-ridge", 2.0F);  // face-only landmarks under-constrain global β
  cfg.mag_weight = args.get_float("mag", 2.0F);
  cfg.max_dv = args.get_float("max-dv", 0.03F);  // ≤3 cm displacement on the (~0.2 m) SMPL-X face
  const auto R = ncg::geom::solve_geometry(v_template, id_dirs, ex_dirs, faces, lap, assoc, bary,
                                           landmarks2d, conf, w_prior, cfg, beta_fixed);

  const auto beta = R.beta;
  const auto id_verts = v_template + torch::einsum("vck,k->vc", {id_dirs, beta});  // identity (β only)
  const auto pers = id_verts + R.delta_v;                                          // + Δv
  const auto prefix = (rec.dir() / args.get("out-prefix", "geom")).string();
  ncg::mesh::write_obj({pers, faces}, prefix + "_personalized.obj");
  ncg::mesh::write_obj({id_verts, faces}, prefix + "_identity.obj");
  ncg::mesh::write_obj({v_template, faces}, prefix + "_mean.obj");
  ncg::io::save_npy(prefix + "_obs.npy", R.obs.contiguous());
  ncg::io::save_npy(prefix + "_delta_v.npy", R.delta_v.contiguous());

  // ---- Phase D: export the two-layer deployable character ----
  if (albedo.defined()) {
    const auto pcpu = pers.to(at::kCPU);
    const auto normals = ncg::mesh::compute_vertex_normals({pcpu, faces});
    const auto joints = torch::matmul(Jreg, pcpu);  // rest-pose joints of the displaced shaped body
    // Layer 1: rigged + textured DISPLACED mesh (drives animation/physics/collision).
    ncg::mesh::write_glb_skinned(pcpu, faces, normals, albedo, joints, parents, lbs_w,
                                 prefix + "_character.glb");
    // Layer 1b: SHARP + white-balanced UV-textured version (the improved face pipeline on the FULL
    // body) — pore-level per-texel albedo + photometric normal map instead of per-vertex colour.
    if (ncg::cuda_available() && !uv_img.empty()) {
      try {
        auto smodel = ncg::body::SmplxModel::load(args.require("smplx"), ndev);
        if (smodel.has_uv()) {
          const int T = args.get_int("tex-res", 1024);
          torch::Tensor uvmask, uvnrm, uvpos, uvgn;
          const auto uvtex = recover_uv_albedo(
              smodel, uv_img, uv_v2d, nrm_l, vis_l, T, pers.to(ndev), uvmask, uvnrm, uvpos, uvgn,
              args.get_float("detail", 0.7F), args.get_float("deshade", 0.5F),
              args.get_float("chroma", 0.6F));
          ncg::io::save_png(prefix + "_albedo_uv.png", uvtex.permute({2, 0, 1}).contiguous().detach());
          ncg::io::save_png(prefix + "_normal_uv.png", uvnrm.permute({2, 0, 1}).contiguous().detach());
          ncg::io::save_npy(prefix + "_uvcoords.npy", smodel.uv_coords().to(at::kCPU).contiguous());
          ncg::io::save_npy(prefix + "_uvfaces.npy", smodel.uv_faces().to(at::kInt).contiguous());
          ncg::mesh::write_glb_textured(pcpu, faces, normals, smodel.uv_coords(), smodel.uv_faces(),
                                        joints, parents, lbs_w, prefix + "_albedo_uv.png",
                                        prefix + "_char_textured.glb", prefix + "_normal_uv.png");
          NCG_LOG_INFO("geom: SHARP UV-textured character -> {}_char_textured.glb (+ _albedo_uv.png)",
                       prefix);

          // ===== FREE-SPLAT LAYER (Phase C): the resolution fix. Seed Layer-2 from the CLEAN albedo
          // (sampled per-vertex from the fusion-fixed UV), FREEZE colour (lr_color=0 → densify adds
          // RESOLUTION, not colour, so it can't undo the fusion fix), run fit_avatar with adaptive
          // densification against the gated photos, then k-NN-bind the free splats to the displaced
          // mesh so they skin with the rig (anti-swim). Layer-1 mesh/rig is unchanged. =====
          if (densify && aframes.size() >= 3 && body_betas.defined()) {
            namespace Fn2 = torch::nn::functional;
            const auto fl = faces.reshape(-1).to(ndev).to(at::kLong);
            const auto uvl = smodel.uv_faces().reshape(-1).to(at::kLong);
            auto vuv = torch::zeros({pcpu.size(0), 2}, uvtex.options());     // [V,2] per-vertex UV
            vuv.index_put_({fl}, smodel.uv_coords().index_select(0, uvl));
            const auto grid = torch::stack({vuv.select(1, 0) * 2 - 1, (1 - vuv.select(1, 1)) * 2 - 1}, 1)
                                  .view({1, -1, 1, 2});
            const auto vcol = Fn2::grid_sample(uvtex.permute({2, 0, 1}).unsqueeze(0), grid,
                                  Fn2::GridSampleFuncOptions().mode(torch::kBilinear)
                                      .padding_mode(torch::kBorder).align_corners(true))
                                  .squeeze(3).squeeze(0).t().contiguous().clamp(0.0, 1.0);  // [V,3]
            ncg::fit::AvatarFitConfig fc;
            fc.iterations = args.get_int("densify-iters", 1800);
            fc.lr_color = 0.0;            // FREEZE colour — preserve the clean albedo
            fc.lr_position = 2e-4;        // small position freedom (densify needs a position gradient)
            fc.densify = true;
            fc.position_reg = args.get_float("dev-reg", 20.0F);   // soft anti-floater
            fc.max_dev = args.get_float("max-dev", 0.012F);       // HARD cap: splats stay ≤1.2cm off-mesh
            fc.min_scale = args.get_float("min-scale", 0.0035F);  // splats overlap into a surface
            fc.opacity_floor = args.get_float("opacity-floor", 0.6F);  // FIX2: no dark holes in covered regions
            fc.densify_grad = args.get_float("densify-grad", 6e-5F);   // FIX3: gentle, residual-driven (don't chase count)
            fc.densify_until = args.get_int("densify-iters", 1800) - 300;
            fc.per_view_exposure = true;  // FIX1: per-frame exposure/WB equalization in the fit
            fc.robust = true;
            fc.max_gaussians = args.get_int("max-splats", 90000);
            fc.init_scale = args.get_float("scale", 0.008F);
            const auto bb = body_betas.slice(0, 0, std::min<int64_t>(body_betas.size(0),
                                                                     smodel.num_betas())).to(ndev);
            const auto cov = app_obs.defined() ? app_obs.to(ndev) : torch::Tensor{};  // FIX2 coverage
            auto fit = ncg::fit::fit_avatar(smodel, bb, aframes, vcol.to(ndev), fc, nullptr, cov);
            auto fcloud = fit.canonical;
            fcloud.to_(at::kCPU);
            const int64_t Nf = fcloud.size();
            const auto bind = ncg::fit::bind_splats_knn(fcloud.positions, pcpu, 4);  // k-NN to mesh
            const auto vw = lbs_w.index_select(0, bind.idx.reshape({-1}))
                                .reshape({Nf, bind.idx.size(1), lbs_w.size(1)});
            const auto sb = (vw * bind.weight.unsqueeze(2)).sum(1);                 // [N,J] blended lbs
            const auto t4 = sb.topk(4, 1);
            auto sw = std::get<0>(t4);
            sw = sw / sw.sum(1, true).clamp_min(1e-9);
            ncg::mesh::write_gaussian_ply(fcloud, prefix + "_free.ply", std::get<1>(t4).to(at::kLong), sw);
            NCG_LOG_INFO("geom: FREE-SPLAT layer -> {}_free.ply  N={} (was {} verts) opacity_std={:.4f}",
                         prefix, Nf, pcpu.size(0), fcloud.opacities.std().item<float>());
            // Before/after render: baseline 1:1 vertex splats vs the densified free splats, same view.
            {
              auto base = ncg::recon::gaussians_on_body(pers.to(ndev), 0.008F, vcol.to(ndev),
                                                        ncg::recon::per_vertex_scale(pcpu, 0.75F).to(ndev));
              auto fr = fcloud; fr.to_(ndev);
              const auto hthr = torch::quantile(pcpu.select(1, 1), 0.86F).item<float>();
              const auto hm = (pers.to(ndev).select(1, 1) > hthr).unsqueeze(1);
              const auto hc = pers.to(ndev).masked_select(hm).reshape({-1, 3}).mean(0);
              for (int k = 0; k < 5; ++k) {
                const float az = -40.0F + 20.0F * k;
                const auto cam = ncg::runtime::Camera::orbit(hc, 0.42F, az, 5.0F, 28.0F, 768, 768, ndev);
                char nm[40];
                std::snprintf(nm, sizeof(nm), "_free_turn%+03d.png", (int)az);
                ncg::io::save_png(prefix + nm, ncg::runtime::render_soft_aniso(fr, cam).image.detach().to(at::kCPU));
                std::snprintf(nm, sizeof(nm), "_base_turn%+03d.png", (int)az);
                ncg::io::save_png(prefix + nm, ncg::runtime::render_soft_aniso(base, cam).image.detach().to(at::kCPU));
              }
              NCG_LOG_INFO("geom: free-splat before/after renders -> {}_base_turn* / {}_free_turn*", prefix, prefix);
            }
          }
        }
      } catch (const std::exception& e) {
        NCG_LOG_WARN("geom: textured/free-splat export skipped ({})", e.what());
      }
    }
    // Save the raw identity tensors so `complete` can rebuild the cloud without re-running NLF.
    ncg::io::save_npy(prefix + "_verts.npy", pcpu.contiguous());
    ncg::io::save_npy(prefix + "_albedo.npy", albedo.contiguous());
    if (app_obs.defined()) ncg::io::save_npy(prefix + "_appobs.npy", app_obs.contiguous());
    // Layer 2: free splats on the displaced mesh, k-NN bound, exported with per-splat bone skinning
    // (blend the bound verts' lbs_weights → top-4) so the engine GS component deforms them with the rig.
    const auto pvs = ncg::recon::per_vertex_scale(pcpu, 0.75F);
    auto cloud = ncg::recon::gaussians_on_body(pcpu, args.get_float("scale", 0.008F), albedo, pvs);
    const auto cpos = cloud.positions.to(at::kCPU);
    const auto bind = ncg::fit::bind_splats_knn(cpos, pcpu, 4);
    const int64_t Ns = bind.idx.size(0), kk = bind.idx.size(1);
    const auto vw = lbs_w.index_select(0, bind.idx.reshape({-1})).reshape({Ns, kk, lbs_w.size(1)});
    const auto sb = (vw * bind.weight.unsqueeze(2)).sum(1);  // [N,J] per-splat bone weights
    const auto t4 = sb.topk(4, 1);
    auto sw = std::get<0>(t4);
    sw = sw / sw.sum(1, true).clamp_min(1e-9);
    ncg::mesh::write_gaussian_ply(cloud, prefix + "_character.ply", std::get<1>(t4).to(at::kLong), sw);
    NCG_LOG_INFO("geom: two-layer character -> {}_character.glb (rigged textured displaced mesh) + "
                 "{}_character.ply(+.skin) ({} k-NN-bound free splats)", prefix, prefix, cpos.size(0));

    // STOP-AND-LOOK: novel-view turntable of the personalized character (needs a GPU; the solve is
    // CPU). Render the same albedo-colored splat cloud on-device so we can eyeball the likeness.
    if (ncg::cuda_available()) {
      const auto dv = at::Device(at::kCUDA, 0);
      const int res = args.get_int("res", 640);
      auto rcloud = ncg::recon::gaussians_on_body(pcpu.to(dv), args.get_float("scale", 0.008F),
                                                  albedo.to(dv), pvs.to(dv));
      const int nv = args.get_int("turn", 8);
      const auto cams = ncg::runtime::orbit_trajectory(rcloud.positions.mean(0),
                                                       args.get_float("radius", 2.4F), 0.0F, nv,
                                                       50.0F, res, res, dv);
      for (int i = 0; i < nv; ++i) {
        const auto im = ncg::runtime::render_soft_aniso(rcloud, cams[i]).image.detach().to(at::kCPU);
        char name[96];
        std::snprintf(name, sizeof(name), "%s_turn%02d.png", prefix.c_str(), i);
        ncg::io::save_png(name, im);
      }
      NCG_LOG_INFO("geom: wrote {} turntable views -> {}_turn*.png", nv, prefix);
    }
  }

  const auto dvn = R.delta_v.norm(2, 1);
  NCG_LOG_INFO("geom: |β|={:.3f}  residual={:.3f}px  |Δv| mean={:.4f} max={:.4f}  "
               "observed verts(o>0.5)={}  → {}_personalized.obj",
               beta.norm().item<float>(), R.residual, dvn.mean().item<float>(),
               dvn.max().item<float>(), (R.obs > 0.5F).sum().item<int64_t>(), prefix);
  rec.log_scalar("geom", "residual_px", R.residual);
  rec.log_scalar("geom", "dv_max", dvn.max().item<double>());
  rec.log_scalar("geom", "n_observed", (R.obs > 0.5F).sum().item<double>());
  return 0;
}

// ============================================================================================
// `complete` — PHASE E CAPSTONE. Render-consistent, observability-gated SDS completion of the
// personalized character (docs/method.md §M10/§M11). Takes the geom outputs (verts/albedo +
// APPEARANCE observability) + the ported SD prior, then for random novel views: renders the avatar,
// VAE-encodes to a latent, computes the SDS gradient (the diffusion prior pulling the render toward
// the data manifold), GATES it by the per-pixel observability (BIT-EXACT zero on well-photographed
// surface, full on the unseen back/sides), and backprops through the differentiable VAE+renderer to
// the splat appearance. Net effect: the unseen regions get plausible, view-consistent detail while
// the photographed identity is never diffusion-rewritten. Exports the completed splat .ply + a
// before/after turntable.
// ============================================================================================
int cmd_complete(const ncg::app::Args& args) {
  NCG_CHECK(ncg::cuda_available(), "complete requires a CUDA device");
  const auto device = at::Device(at::kCUDA, 0);
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "complete"));
  const auto prefix = (rec.dir() / args.get("out-prefix", "completed")).string();

  // Identity tensors from the geom run.
  const auto verts = ncg::io::load_npy(args.require("verts")).to(device, at::kFloat);    // [V,3]
  const auto albedo0 = ncg::io::load_npy(args.require("albedo")).to(device, at::kFloat);  // [V,3]
  auto app_obs = ncg::io::load_npy(args.require("appobs")).to(device, at::kFloat);        // [V]
  if (app_obs.dim() == 2) app_obs = app_obs.select(1, 0);
  const int64_t V = verts.size(0);

  // One splat per vertex; appearance (colors) is the optimized parameter, geometry stays fixed.
  const auto pvs = ncg::recon::per_vertex_scale(verts.to(at::kCPU), 0.75).to(device);
  auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.008F), albedo0, pvs);
  cloud.colors = albedo0.clone().detach().requires_grad_(true);  // leaf parameter
  // Per-splat observability cloud (colors = app_obs), rendered no-grad to get the per-pixel gate.
  auto obs_cloud = cloud;
  obs_cloud.colors = app_obs.unsqueeze(1).expand({V, 3}).contiguous();

  // Spatial coherence prior: a cotangent Laplacian on the mesh so neighbouring splats keep similar
  // colour — without it, SDS drives each splat independently into rainbow high-frequency noise.
  const auto faces = ncg::io::SafeTensors::open(args.require("smplx"))
                         .view("faces").clone().to(at::kLong);
  const auto Lap = ncg::geom::cotangent_laplacian(verts.to(at::kCPU), faces).to(device);
  const float w_anchor = args.get_float("anchor", 6.0F);   // stay near the photographed albedo
  const float w_lap = args.get_float("color-lap", 8.0F);   // spatial smoothness on splat colours
  const int diffuse_steps = args.get_int("diffuse", 8);    // heat-smoothing steps on the SDS gradient
  const float diffuse_mu = args.get_float("diffuse-mu", 0.2F);

  // The ported SD prior + schedule.
  const std::string sd = args.require("sd-dir");
  ncg::diffuse::SdGuidanceConfig gcfg;
  gcfg.guidance = args.get_float("guidance", 15.0F);
  auto guide = ncg::diffuse::SdGuidance::load(sd + "/sd_unet.ts", sd + "/sd_vae.ts",
                                              sd + "/sd_cond.safetensors", device, gcfg);
  const ncg::diffuse::DdpmSchedule sch({}, device);
  const auto pred = guide.predictor();
  ncg::diffuse::SdsConfig scfg;
  ncg::diffuse::CompletionConfig ccfg;
  ccfg.obs_lo = args.get_float("obs-lo", 0.15F);
  ccfg.obs_hi = args.get_float("obs-hi", 0.35F);

  const int res = args.get_int("res", 512);
  const float radius = args.get_float("radius", 2.4F);
  const auto center = cloud.positions.mean(0);
  torch::optim::Adam opt({cloud.colors}, torch::optim::AdamOptions(args.get_float("lr", 0.02F)));

  // Front view (azimuth 0) as the identity-protection witness: it must NOT drift.
  const auto front_cam = ncg::runtime::Camera::orbit(center, radius, 0.0F, 0.0F, 50.0F, res, res, device);
  const auto front_before =
      ncg::runtime::render_soft_aniso(cloud, front_cam).image.detach().clone();

  const int iters = args.get_int("iters", 300);
  for (int it = 0; it < iters; ++it) {
    // Bias views toward the UNDER-observed back/sides (azimuth away from frontal), some elevation.
    const float az = static_cast<float>((it * 47) % 360);
    const float el = static_cast<float>(((it * 13) % 41) - 20);  // [-20,20] deg
    const auto cam = ncg::runtime::Camera::orbit(center, radius, az, el, 50.0F, res, res, device);

    const auto out = ncg::runtime::render_soft_aniso(cloud, cam);
    const auto rgb = out.image.unsqueeze(0);  // [1,3,H,W], graph-connected to cloud.colors

    torch::Tensor obs_latent;
    {
      torch::NoGradGuard ng;
      const auto obs_img = ncg::runtime::render_soft_aniso(obs_cloud, cam).image.narrow(0, 0, 1);
      obs_latent = torch::adaptive_avg_pool2d(obs_img.unsqueeze(0), {res / 8, res / 8});  // [1,1,h,w]
    }

    const auto latent = guide.encode_image(rgb);          // [1,4,h,w] differentiable
    const auto r = ncg::diffuse::sds_loss(latent, sch, pred, scfg);
    const auto gated = ncg::diffuse::apply_completion_gate(r.grad, obs_latent, ccfg);  // zero on observed
    const auto target = (latent - gated).detach();
    const auto sds_surrogate = 0.5 * torch::nn::functional::mse_loss(
                                         latent, target,
                                         torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));
    // Regularizers keep SDS honest: anchor to the recovered albedo + Laplacian color smoothness.
    const auto anchor = w_anchor * (cloud.colors - albedo0).pow(2).mean();
    const auto Lc = Lap.is_sparse() ? torch::mm(Lap, cloud.colors) : torch::matmul(Lap, cloud.colors);
    const auto smooth = w_lap * Lc.pow(2).mean();
    const auto loss = sds_surrogate + anchor + smooth;
    opt.zero_grad();
    loss.backward();
    {
      // SMOOTH THE UPDATE, NOT THE COLORS. SDS produces a spatially-incoherent per-splat gradient
      // (the speckle source). Heat-diffuse the gradient on the mesh before the optimizer step so
      // only LOW-FREQUENCY, coherent colour changes are applied — the original photographed albedo
      // detail is untouched (we never smooth the colours themselves), but speckle can't accumulate.
      torch::NoGradGuard ng;
      auto g = cloud.colors.mutable_grad();
      if (g.defined()) {
        g = torch::nan_to_num(g, 0.0, 0.0, 0.0);
        for (int s = 0; s < diffuse_steps; ++s) {
          const auto Lg = Lap.is_sparse() ? torch::mm(Lap, g) : torch::matmul(Lap, g);
          g.add_(Lg, -diffuse_mu);  // heat diffusion on the gradient field
        }
        cloud.colors.mutable_grad().copy_(g);
      }
    }
    opt.step();
    {
      torch::NoGradGuard ng;
      cloud.colors.clamp_(0.0, 1.0);
      cloud.colors.copy_(torch::nan_to_num(cloud.colors, 0.5, 1.0, 0.0));
    }
    if (it % 25 == 0) {
      NCG_LOG_INFO("complete: iter {}/{} sds_grad_norm={:.4f}", it, iters, r.grad_norm);
      rec.log_scalar("complete", "sds_grad_norm", r.grad_norm);
    }
  }

  // Identity-protection witness: the frontal (well-observed) render must be ~unchanged.
  const auto front_after = ncg::runtime::render_soft_aniso(cloud, front_cam).image.detach();
  NCG_LOG_INFO("complete: front-view drift (observed identity) = {:.4f} (should be small)",
               (front_after - front_before).abs().mean().item<double>());

  // Export the completed splat cloud + a turntable to SEE the filled-in back/sides.
  auto out_cloud = cloud;
  out_cloud.colors = cloud.colors.detach();
  ncg::mesh::write_gaussian_ply(out_cloud, prefix + "_completed.ply");
  const int nv = args.get_int("turn", 8);
  const auto cams =
      ncg::runtime::orbit_trajectory(center, radius, 0.0F, nv, 50.0F, res, res, device);
  for (int i = 0; i < nv; ++i) {
    const auto im = ncg::runtime::render_soft_aniso(out_cloud, cams[i]).image.detach().to(at::kCPU);
    char name[96];
    std::snprintf(name, sizeof(name), "%s_turn%02d.png", prefix.c_str(), i);
    ncg::io::save_png(name, im);
  }
  NCG_LOG_INFO("complete: wrote completed character -> {}_completed.ply + {} turntable views",
               prefix, nv);
  return 0;
}

// ============================================================================================
// `face` — the novel identity estimator on a real album (ncg::recon::solve_face_identity).
// Runs NLF over every photo to get each mesh's projected vertices, samples the 51 SMPL-X face
// landmarks per photo (barycentric on lmk_faces_idx / lmk_bary_coords), then FUSES the album into
// a single shared NEUTRAL identity face — the face no one photo shows — while factoring out each
// photo's expression & head pose and robustly rejecting wrong-person / bad-detection frames. This
// is the multi-photo consistency NLF (single-image) cannot do: NLF gives an inconsistent identity
// per photo; we recover the one identity that explains the whole album. (docs/method.md Theorem 2.)
// ============================================================================================
int cmd_face(const ncg::app::Args& args) {
  const auto device = ncg::cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
  auto rec = ncg::record::Recorder::create("runs", args.get("run", "face"));
  namespace fs = std::filesystem;

  // ---- SMPL-X face front-end tensors (exported by tools/convert_smplx.py --num-face-id) --------
  auto st = ncg::io::SafeTensors::open(args.require("smplx"));
  auto own = [&](const std::string& k) {
    NCG_CHECK(st.has(k),
              "face: smplx asset missing '{}' — re-export with tools/convert_smplx.py "
              "(carries face_id_dirs / face_expr_dirs / lmk_faces_idx / lmk_bary_coords)",
              k);
    return st.view(k).clone();
  };
  const auto v_template = own("v_template").to(at::kFloat);            // [V,3]
  const auto faces = own("faces").to(at::kLong);                       // [F,3]
  const auto id_dirs = own("face_id_dirs").to(at::kFloat);            // [V,3,n_id]
  const auto expr_dirs = own("face_expr_dirs").to(at::kFloat);        // [V,3,n_ex]
  const auto lmk_faces = own("lmk_faces_idx").to(at::kLong);          // [L]
  const auto lmk_bary = own("lmk_bary_coords").to(at::kFloat);        // [L,3]
  const int64_t V = v_template.size(0), L = lmk_faces.size(0);
  const int64_t n_id = id_dirs.size(2), n_ex = expr_dirs.size(2);

  // Landmark = barycentric blend of its triangle's 3 corner vertices. `corner` [L,3] vertex ids.
  const auto corner = faces.index_select(0, lmk_faces);  // [L,3]
  // base [L,3], id_basis [L,3,n_id], expr_basis [L,3,n_ex] sampled at the 51 landmarks.
  const auto Vt_c = v_template.index_select(0, corner.reshape({-1})).reshape({L, 3, 3});      // [L,c,3]
  const auto base_lm = torch::einsum("lc,lcd->ld", {lmk_bary, Vt_c});                          // [L,3]
  const auto Id_c = id_dirs.index_select(0, corner.reshape({-1})).reshape({L, 3, 3, n_id});    // [L,c,3,K]
  const auto id_basis = torch::einsum("lc,lcdk->ldk", {lmk_bary, Id_c});                        // [L,3,K]
  const auto Ex_c = expr_dirs.index_select(0, corner.reshape({-1})).reshape({L, 3, 3, n_ex});
  const auto expr_basis = torch::einsum("lc,lcdk->ldk", {lmk_bary, Ex_c});                      // [L,3,Kx]

  // ---- album -> per-photo 2D landmarks via NLF -------------------------------------------------
  std::vector<std::string> all;
  for (const auto& e : fs::directory_iterator(args.require("frames"))) {
    const auto ext = e.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".JPG" || ext == ".JPEG")
      all.push_back(e.path().string());
  }
  std::sort(all.begin(), all.end());
  NCG_CHECK(!all.empty(), "face: no images in --frames dir");
  const int max_frames = args.get_int("max-frames", 80);
  std::vector<std::string> paths;
  if (static_cast<int>(all.size()) <= max_frames) {
    paths = all;
  } else {
    const double step = static_cast<double>(all.size()) / max_frames;
    for (int i = 0; i < max_frames; ++i) paths.push_back(all[static_cast<size_t>(i * step)]);
  }

  ncg::body::NlfConfig nc;
  nc.detection = args.get_int("detection", 0);
  auto nlf = ncg::body::Nlf::load(args.require("weights"), device, nc);

  const bool texture = args.get_int("texture", 1) != 0 && ncg::cuda_available();
  // SmplxModel gives the UV layout for the high-res per-texel solve (recover_uv_albedo).
  const auto model = texture ? ncg::body::SmplxModel::load(args.require("smplx"), device)
                             : ncg::body::SmplxModel::load(args.require("smplx"), at::kCPU);
  namespace Fn = torch::nn::functional;
  const int samp_res = args.get_int("sample-res", 2560);  // keep face detail (was 1024 → ~300px face)
  auto downscale = [&](const torch::Tensor& img) {  // -> (scaled CHW, scale factor) for per-texel
    const double s = std::min(1.0, static_cast<double>(samp_res) /
                                       static_cast<double>(std::max(img.size(1), img.size(2))));
    auto sm = Fn::interpolate(img.unsqueeze(0), Fn::InterpolateFuncOptions()
                                                    .scale_factor(std::vector<double>{s, s})
                                                    .mode(torch::kBilinear)
                                                    .align_corners(false))
                  .squeeze(0);
    return std::make_pair(sm, s);
  };
  std::vector<torch::Tensor> lm_list;
  std::vector<std::string> used;
  std::vector<torch::Tensor> obs_l, nrm_l, vis_l;  // per-photo full-vertex appearance (texture)
  std::vector<torch::Tensor> uv_img, uv_v2d;       // downscaled image + scaled v2d (per-texel solve)
  for (const auto& p : paths) {
    torch::Tensor img;
    try {
      img = ncg::io::load_image(p, 3).to(device);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("face: cannot read {} ({}), skipping", p, e.what());
      continue;
    }
    ncg::body::NlfPrediction pred;
    try {
      pred = nlf.detect(img);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("face: NLF failed on {} ({}), skipping", p, e.what());
      continue;
    }
    if (pred.vertices2d.size(0) != V) continue;  // need full mesh to sample the landmarks
    const auto v2d = pred.vertices2d.to(at::kCPU).to(at::kFloat);                  // [V,2]
    const auto v2d_c = v2d.index_select(0, corner.reshape({-1})).reshape({L, 3, 2});
    lm_list.push_back(torch::einsum("lc,lcd->ld", {lmk_bary, v2d_c}));             // [L,2]
    used.push_back(p);
    if (texture) {  // per-photo appearance for the robust multi-illumination albedo solve (C1/C2)
      const auto v2dd = pred.vertices2d.to(device);
      ncg::mesh::TriMesh m{pred.vertices3d.to(at::kCPU), faces};                   // posed normals
      nrm_l.push_back(ncg::mesh::compute_vertex_normals(m).to(device));
      obs_l.push_back(ncg::recon::sample_vertex_colors(img, v2dd).clamp(0.0, 1.0));
      vis_l.push_back(ncg::recon::vertex_visibility(v2dd, pred.vertices3d.select(1, 2).to(device),
                                                    static_cast<int64_t>(img.size(1)),
                                                    static_cast<int64_t>(img.size(2))));
      auto [sm, s] = downscale(img);                   // per-texel: sample the downscaled photo
      uv_img.push_back(sm);
      uv_v2d.push_back(v2dd * static_cast<float>(s));  // v2d in the downscaled image's pixels
    }
  }
  NCG_CHECK(lm_list.size() >= 2, "face: need >=2 usable detections; got {}", lm_list.size());
  const auto landmarks2d = torch::stack(lm_list, 0);  // [N,L,2]
  NCG_LOG_INFO("face: {} photos -> {} usable detections; {} landmarks, {} id dims, {} expr dims",
               paths.size(), lm_list.size(), L, n_id, n_ex);

  // ---- the contribution: robust joint identity / expression / pose factorization ---------------
  ncg::recon::FaceIdentityConfig cfg;
  cfg.iterations = args.get_int("iters", 40);
  cfg.id_ridge = std::stof(args.get("id-ridge", "3.0"));  // keep |β|~2 (0.01 over-fit → |β|=27, warped)
  cfg.expr_ridge = std::stof(args.get("expr-ridge", "0.1"));
  cfg.robust = args.get_int("robust", 1) != 0;
  const auto R = ncg::recon::solve_face_identity(base_lm, id_basis, expr_basis, landmarks2d, cfg);

  // Personalized NEUTRAL identity mesh: v_template + id_dirs · β  (expression set to 0 = neutral).
  const auto beta = R.id_shape.to(at::kCPU).to(at::kFloat);                         // [n_id]
  auto id_verts = v_template + torch::einsum("vck,k->vc", {id_dirs, beta});         // [V,3]
  // GEOMETRY UPGRADE: optionally render the texture on the Phase-B personalized OUT-OF-SUBSPACE
  // geometry (a geom run's `_verts.npy`, which carries Δv beyond the identity subspace) so the
  // texture sits on the person's real bone structure, not the SMPL-X average. Texture SAMPLING is
  // unchanged (it uses each photo's NLF projection); only the rendered/exported rest shape changes.
  if (args.has("geom-verts")) {
    const auto gv = ncg::io::load_npy(args.require("geom-verts")).to(at::kCPU).to(at::kFloat);
    if (gv.sizes() == id_verts.sizes()) {
      id_verts = gv.contiguous();
      NCG_LOG_INFO("face: rendering on personalized Phase-B geometry ({} verts) from {}",
                   id_verts.size(0), args.require("geom-verts"));
    } else {
      NCG_LOG_WARN("face: --geom-verts shape {}x{} != mesh {}x{}; ignoring", gv.size(0),
                   gv.size(1), id_verts.size(0), id_verts.size(1));
    }
  }

  const auto prefix = (rec.dir() / args.get("out-prefix", "face")).string();
  ncg::mesh::TriMesh neutral{id_verts, faces};
  ncg::mesh::TriMesh mean{v_template, faces};
  ncg::mesh::write_obj(neutral, prefix + "_identity.obj");
  ncg::mesh::write_obj(mean, prefix + "_mean.obj");

  // Report the recovery + per-photo trust (the C2 robustness signal).
  NCG_LOG_INFO("face: identity recovered |β|={:.3f}  reproj-residual={:.3f}px", beta.norm().item<float>(),
               R.residual);
  const auto w = R.weight.to(at::kCPU);
  for (size_t i = 0; i < used.size(); ++i)
    NCG_LOG_INFO("  trust {:.3f}  {}", w[static_cast<int64_t>(i)].item<float>(),
                 fs::path(used[i]).filename().string());
  rec.log_scalar("face", "residual_px", R.residual);
  rec.log_scalar("face", "beta_norm", beta.norm().item<double>());
  rec.log_scalar("face", "n_used", static_cast<double>(lm_list.size()));

  // ---- photoreal skin: robust multi-illumination albedo on the PERSONALIZED face --------------
  // The same album, now textured. Each photo's per-vertex color is delit by the C1/C2 inverse
  // renderer (per-photo SH light solved away, outliers rejected) — and we fold in the identity
  // estimator's per-photo trust w_i so the SAME frames it flagged as wrong-person/bad also lose
  // their vote on appearance (coherent robustness). Albedo is per-vertex, so it drops straight onto
  // the recovered identity geometry; we then render the face and relight it under novel lights.
  if (texture && !obs_l.empty()) {
    const int64_t Nt = static_cast<int64_t>(obs_l.size());
    auto W = torch::stack(vis_l, 0);                                          // [N,V]
    W = W * R.weight.to(device).slice(0, 0, Nt).unsqueeze(1).clamp_min(0.05); // × identity trust
    ncg::recon::InverseRenderConfig ic;
    ic.iterations = args.get_int("albedo-iters", 80);
    ic.robust = Nt >= 2;
    const auto ir = ncg::recon::solve_inverse_render(torch::stack(obs_l, 0), torch::stack(nrm_l, 0),
                                                     W, ic);
    const auto albedo = torch::nan_to_num(ir.albedo).clamp(0.0, 1.0);          // [V,3] relightable
    NCG_LOG_INFO("face: recovered relightable skin albedo from {} view(s)", Nt);
    // Export albedo + geometry so the face can be rendered as a sharp textured MESH (no splat blur).
    ncg::io::save_npy(prefix + "_albedo.npy", albedo.to(at::kCPU).contiguous());
    ncg::io::save_npy(prefix + "_verts.npy", id_verts.contiguous());
    ncg::io::save_npy(prefix + "_faces.npy", faces.to(at::kInt).contiguous());

    // ---- high-res per-texel UV albedo + photometric normals (the sharpness win) -----------------
    // Lift the albedo from per-vertex (~10⁴) to per-texel (T²) over the SMPL-X UV layout — pore-level
    // resolution — and recover a tangent-space normal map by photometric stereo across the album's
    // illumination diversity. Outputs a sharp, relightable face texture + normal map for a real engine.
    if (model.has_uv() && !uv_img.empty()) {
      const int T = args.get_int("tex-res", 1024);
      const auto verts_dev = id_verts.to(device);
      std::vector<torch::Tensor> viss_t;  // visibility × identity trust (coherent robustness)
      for (int64_t i = 0; i < Nt; ++i)
        viss_t.push_back(vis_l[static_cast<size_t>(i)] * R.weight[i].to(device).clamp_min(0.05));
      torch::Tensor uvmask, uvnrm, uvpos, uvgn;
      const auto uvtex =
          recover_uv_albedo(model, uv_img, uv_v2d, nrm_l, viss_t, T, verts_dev, uvmask, uvnrm, uvpos,
                            uvgn, args.get_float("detail", 0.7F), args.get_float("deshade", 0.5F),
                            args.get_float("chroma", 0.6F));
      // TEXTURE-RESOLUTION render: turn each valid UV texel into a 3D surface splat coloured by the
      // sharp per-texel albedo. Render detail = texture resolution (T²), not the ~10⁴ vertex count —
      // this is what actually makes the rendered face sharp (the splat portraits below were
      // vertex-limited). Saved + rendered as the primary face view.
      {
        const auto m = (uvmask.reshape({T * T}) > 0.5F);
        const auto fp = uvpos.index({m});                                // [P,3] surface points
        const auto fc = uvtex.reshape({T * T, 3}).index({m}).clamp(0.0F, 1.0F);
        const auto fn = uvgn.index({m});                                 // [P,3] geometric normals
        const int64_t P = fp.size(0);
        ncg::recon::GaussianCloud tc;
        tc.positions = fp;
        tc.colors = fc;
        tc.scales = torch::full({P, 3}, args.get_float("texel-scale", 0.0010F), fp.options());
        tc.opacities = torch::ones({P, 1}, fp.options());
        tc.rotations = torch::zeros({P, 4}, fp.options());
        tc.rotations.select(1, 0).fill_(1.0F);  // identity quaternion (w,x,y,z)
        const auto hthr = torch::quantile(id_verts.select(1, 1), 0.88).item<float>();
        const auto hmask = (id_verts.select(1, 1).to(device) > hthr).unsqueeze(1);  // [V,1]
        const auto hc = verts_dev.masked_select(hmask).reshape({-1, 3}).mean(0);    // head centroid
        const int pres = args.get_int("portrait-res", 768);
        // SHADED preview: flat albedo reads as wax; add gentle FORM shading from the per-texel
        // geometric normal (soft frontal key + strong ambient) so it reads as a lit 3D face. The
        // exported asset stays pure albedo + normal map; this only affects the preview render.
        const float shade = args.get_float("shade", 0.45F);
        auto ld = torch::tensor({0.25F, 0.25F, 1.0F}, fp.options());
        ld = ld / ld.norm();
        const auto ndl = (fn * ld).sum(1, true).clamp(0.0F, 1.0F);                  // [P,1]
        const auto shaded = (fc * ((1.0F - shade) + shade * 2.0F * ndl)).clamp(0.0F, 1.0F);
        for (int k = 0; k < 5; ++k) {
          const float az = -40.0F + 20.0F * static_cast<float>(k);
          const auto cam = ncg::runtime::Camera::orbit(hc, 0.42F, az, 5.0F, 28.0F, pres, pres, device);
          char nm[32];
          std::snprintf(nm, sizeof(nm), "_face_sharp_%+03d.png", static_cast<int>(az));
          tc.colors = fc;
          ncg::io::save_png(prefix + nm, ncg::runtime::render_gaussians(tc, cam).image);
          std::snprintf(nm, sizeof(nm), "_face_lit_%+03d.png", static_cast<int>(az));
          tc.colors = shaded;
          ncg::io::save_png(prefix + nm, ncg::runtime::render_gaussians(tc, cam).image);
        }
        NCG_LOG_INFO("face: texture-resolution face render ({} texel splats) -> {}_face_sharp_*.png "
                     "+ shaded {}_face_lit_*.png", P, prefix, prefix);

        // ---- LEARNED-PRIOR REFINEMENT: UV-texture-field SDS (task 16) ----------------------------
        // Photoreal skin/eyes need a learned prior the analytic pipeline can't synthesize. Optimize
        // the UV ALBEDO IMAGE (a contiguous 2D texture — SD's native, spatially-coherent domain, NOT
        // independent per-splat colours that speckle) with the diffusion prior: render the dense
        // texel cloud (differentiable) to near-frontal views, push the render toward the SD manifold
        // via SDS (LOW noise = refine, not regenerate), and constrain with TV smoothness + a strong
        // anchor to the analytic albedo so identity is preserved. Re-bake → _face_refined_*.png.
        torch::Tensor refined_uvtex;
        if (args.get_int("refine-sds", 0) != 0 && args.has("sd-dir")) {
          const std::string sdd = args.require("sd-dir");
          ncg::diffuse::SdGuidanceConfig gc;
          gc.guidance = args.get_float("guidance", 12.0F);
          auto guide = ncg::diffuse::SdGuidance::load(sdd + "/sd_unet.ts", sdd + "/sd_vae.ts",
                                                      sdd + "/sd_cond.safetensors", device, gc);
          const ncg::diffuse::DdpmSchedule sch({}, device);
          const auto pred = guide.predictor();
          ncg::diffuse::SdsConfig scfg;
          scfg.t_min = 20;
          scfg.t_max = args.get_int("refine-tmax", 550);  // cap noise: REFINE, don't regenerate
          auto Tuv = uvtex.permute({2, 0, 1}).unsqueeze(0).contiguous().clone().detach()
                         .requires_grad_(true);                                  // [1,3,T,T]
          const auto Tuv0 = Tuv.detach().clone();
          const auto vimg = uvmask.view({1, 1, T, T});
          const auto midx = m.nonzero().squeeze(1);                              // [P] valid texels
          torch::optim::Adam opt({Tuv}, torch::optim::AdamOptions(args.get_float("refine-lr", 0.008F)));
          const float wA = args.get_float("refine-anchor", 5.0F);
          const float wTV = args.get_float("refine-tv", 0.4F);
          const int rit = args.get_int("refine-iters", 80);
          const int64_t sub = args.get_int("refine-sub", 160000);               // texels/iter (speed)
          for (int it = 0; it < rit; ++it) {
            const auto colflat = Tuv.reshape({3, T * T}).t();                    // [T^2,3] diff'able
            const auto fc_cur = colflat.index_select(0, midx);                  // [P,3]
            torch::Tensor sel;
            if (P > sub) sel = torch::randperm(P, midx.options()).slice(0, 0, sub);
            else sel = torch::arange(P, midx.options());
            ncg::recon::GaussianCloud rc;
            rc.positions = fp.index_select(0, sel);
            rc.colors = fc_cur.index_select(0, sel);
            rc.scales = torch::full({sel.size(0), 3},
                                    args.get_float("texel-scale", 0.0010F) * 1.7F, fp.options());
            rc.opacities = torch::ones({sel.size(0), 1}, fp.options());
            rc.rotations = torch::zeros({sel.size(0), 4}, fp.options());
            rc.rotations.select(1, 0).fill_(1.0F);
            const float az = static_cast<float>(((it * 37) % 61) - 30);         // [-30,30] near-front
            const auto cam =
                ncg::runtime::Camera::orbit(hc, 0.42F, az, 5.0F, 28.0F, 512, 512, device);
            const auto rgb = ncg::runtime::render_soft_aniso(rc, cam).image.unsqueeze(0);
            const auto latent = guide.encode_image(rgb);
            const auto r = ncg::diffuse::sds_loss(latent, sch, pred, scfg);
            const auto target = (latent - r.grad).detach();
            const auto sds_surr = 0.5 * torch::nn::functional::mse_loss(
                latent, target, torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));
            const auto anchor = wA * ((Tuv - Tuv0) * vimg).pow(2).mean();
            const auto dx = (Tuv.slice(3, 1) - Tuv.slice(3, 0, T - 1)).abs().mean();
            const auto dy = (Tuv.slice(2, 1) - Tuv.slice(2, 0, T - 1)).abs().mean();
            const auto loss = sds_surr + anchor + wTV * (dx + dy);
            opt.zero_grad();
            loss.backward();
            opt.step();
            {
              torch::NoGradGuard ng;
              Tuv.clamp_(0.0, 1.0);
              Tuv.copy_(torch::nan_to_num(Tuv, 0.5, 1.0, 0.0));
            }
            if (it % 20 == 0)
              NCG_LOG_INFO("face: refine-sds iter {}/{} sds_grad_norm={:.4f}", it, rit, r.grad_norm);
          }
          refined_uvtex = Tuv.squeeze(0).permute({1, 2, 0}).detach().contiguous();  // [T,T,3]
          const auto rfc = refined_uvtex.reshape({T * T, 3}).index({m}).clamp(0.0F, 1.0F);
          ncg::recon::GaussianCloud rcl = tc;
          rcl.colors = rfc;
          for (int k = 0; k < 5; ++k) {
            const float az = -40.0F + 20.0F * static_cast<float>(k);
            const auto cam =
                ncg::runtime::Camera::orbit(hc, 0.42F, az, 5.0F, 28.0F, pres, pres, device);
            char nm[36];
            std::snprintf(nm, sizeof(nm), "_face_refined_%+03d.png", static_cast<int>(az));
            ncg::io::save_png(prefix + nm, ncg::runtime::render_gaussians(rcl, cam).image);
          }
          NCG_LOG_INFO("face: SDS-refined face -> {}_face_refined_*.png", prefix);
        }

        // ---- IMG2IMG MULTI-VIEW BAKE (task 16+): the stronger learned signal ----------------------
        // SDS-as-loss is a weak refiner. Instead, for each of several views: render the face from the
        // CURRENT UV texture, run SDEdit/img2img (a real DDIM denoise toward the photoreal manifold,
        // structure-preserving at moderate strength), then BACK-PROJECT the refined pixels into the UV
        // texture, keeping each texel's MOST-FRONTAL view (sharp, seam-free). Sequential: each view
        // renders from the texture the previous views already refined → 3D-consistent. Bakes real
        // skin/eye detail the analytic pipeline can't synthesize, without per-splat speckle.
        if (args.get_int("refine-bake", 0) != 0 && args.has("sd-dir")) {
          namespace Fn = torch::nn::functional;
          const std::string sdd = args.require("sd-dir");
          ncg::diffuse::SdGuidanceConfig gc;
          gc.guidance = args.get_float("bake-guidance", 7.5F);
          const bool ctrl = args.get_int("bake-control", 0) != 0;  // geometry-conditioned (ControlNet)
          auto guide = ncg::diffuse::SdGuidance::load(
              sdd + (ctrl ? "/control_unet.ts" : "/sd_unet.ts"), sdd + "/sd_vae.ts",
              sdd + "/sd_cond.safetensors", device, gc);
          const ncg::diffuse::DdpmSchedule sch({}, device);
          const float strength = args.get_float("bake-strength", 0.35F);
          const int steps = args.get_int("bake-steps", 30);
          const int br = 512;  // SD native render size for img2img

          auto uvcur = uvtex.reshape({T * T, 3}).clone();          // [T^2,3] mutable texture
          auto conf = torch::zeros({P}, fp.options());             // per-texel best frontality so far
          const auto pidx = m.nonzero().squeeze(1);                // [P] texel flat indices
          // Frame TIGHT on the face centroid (not the head top) so the face fills the 512 SD frame —
          // SD then resolves fine skin detail instead of smoothing a small face in a big frame.
          const auto face_c = fp.mean(0);
          const float brad = args.get_float("bake-radius", 0.30F);
          const float bfov = args.get_float("bake-fov", 24.0F);
          const std::vector<std::pair<float, float>> views = {
              {0, 0}, {-22, 0}, {22, 0}, {0, -15}, {0, 12}, {-40, 5}, {40, 5}};
          for (size_t vi = 0; vi < views.size(); ++vi) {
            const float az = views[vi].first, el = views[vi].second;
            auto rc = tc;
            rc.colors = uvcur.index({m});                          // [P,3] current colours
            const auto cam = ncg::runtime::Camera::orbit(face_c, brad, az, el, bfov, br, br, device);
            const auto rendered = ncg::runtime::render_gaussians(rc, cam).image;       // [3,br,br]
            // Project texels into this camera + frontality from the geometric normal.
            torch::Tensor uvp, depth;
            cam.project(fp, uvp, depth);                           // uvp [P,2], depth [P]
            const auto ncam = torch::matmul(fn, cam.R.t());        // normals in camera space [P,3]
            torch::Tensor refined;
            if (ctrl) {
              // Render a NORMAL MAP of this view to condition the ControlNet (locks diffusion to the
              // face surface → photoreal detail without drift/seams). Normal-as-colour, flip to the
              // normal-map convention (camera +z toward scene → outward normal has -z).
              auto nrc = rc;
              nrc.colors = (torch::stack({ncam.select(1, 0), -ncam.select(1, 1), -ncam.select(1, 2)}, 1)
                                * 0.5F + 0.5F).clamp(0.0F, 1.0F);
              const auto nmap = ncg::runtime::render_gaussians(nrc, cam).image;        // [3,br,br]
              refined = guide.img2img_control(rendered.unsqueeze(0), nmap.unsqueeze(0), strength,
                                              steps, sch).squeeze(0);
            } else {
              refined = guide.img2img(rendered.unsqueeze(0), strength, steps, sch).squeeze(0);
            }
            const auto front = torch::relu(-ncam.select(1, 2)) *
                               (depth > 0).to(fp.dtype());         // [P] frontality, in front of cam
            const auto gx = uvp.select(1, 0) / (br - 1) * 2 - 1;
            const auto gy = uvp.select(1, 1) / (br - 1) * 2 - 1;
            const auto grid = torch::stack({gx, gy}, 1).view({1, P, 1, 2});
            const auto samp = Fn::grid_sample(refined.unsqueeze(0), grid,
                                              Fn::GridSampleFuncOptions().mode(torch::kBilinear)
                                                  .padding_mode(torch::kZeros).align_corners(true))
                                  .view({3, P}).t();                // [P,3] refined colour per texel
            const auto inb = ((uvp.select(1, 0) >= 0) & (uvp.select(1, 0) <= br - 1) &
                              (uvp.select(1, 1) >= 0) & (uvp.select(1, 1) <= br - 1))
                                 .to(fp.dtype());
            const auto w = front * inb;                            // [P] this view's quality
            const auto better = (w > conf).to(fp.dtype()).unsqueeze(1);  // keep most-frontal view
            const auto old = uvcur.index_select(0, pidx);
            uvcur.index_copy_(0, pidx, samp * better + old * (1.0F - better));
            conf = torch::maximum(conf, w);
            NCG_LOG_INFO("face: bake view {}/{} (az={:.0f}) baked", vi + 1, views.size(), az);
          }
          // BEST OF BOTH: keep the ControlNet's clean, even, photoreal skin as the base, and re-inject
          // the ANALYTIC texture's identity-specific high-frequency LUMINANCE detail (the real pores
          // /edges from the photos that diffusion smoothed away). Clean skin tone + your micro-detail.
          const float bdw = args.get_float("bake-detail", 0.9F);
          if (bdw > 0.0F) {
            namespace Fb = torch::nn::functional;
            auto bk1 = torch::tensor({1.F, 4.F, 6.F, 4.F, 1.F}, uvcur.options());
            auto bk2 = torch::outer(bk1, bk1);
            bk2 = bk2 / bk2.sum();
            const auto bk = bk2.view({1, 1, 5, 5});
            const auto olum = uvtex.reshape({T * T, 3}).mean(1).reshape({1, 1, T, T});  // analytic lum
            const auto oblur = Fb::conv2d(olum, bk, Fb::Conv2dFuncOptions().padding(2));
            const auto detail = (olum - oblur).reshape({T * T, 1});                     // identity hi-freq
            uvcur = (uvcur + bdw * detail).clamp(0.0F, 1.0F);                            // onto clean base
          }
          const auto rfc = uvcur.index({m}).clamp(0.0F, 1.0F);
          auto rcl = tc;
          rcl.colors = rfc;
          for (int k = 0; k < 5; ++k) {
            const float az = -40.0F + 20.0F * static_cast<float>(k);
            const auto cam =
                ncg::runtime::Camera::orbit(hc, 0.42F, az, 5.0F, 28.0F, pres, pres, device);
            char nm[36];
            std::snprintf(nm, sizeof(nm), "_face_baked_%+03d.png", static_cast<int>(az));
            ncg::io::save_png(prefix + nm, ncg::runtime::render_gaussians(rcl, cam).image);
          }
          ncg::io::save_png(prefix + "_albedo_baked_uv.png",
                            uvcur.reshape({T, T, 3}).permute({2, 0, 1}).contiguous());
          NCG_LOG_INFO("face: img2img-baked photoreal face -> {}_face_baked_*.png", prefix);
        }
      }
      // EXTERNAL REPROJECTION: bake user-provided refined views (off-the-shelf IP-Adapter-FaceID
      // renders of the lit portraits, named refine_<+/-NN>.png at the 5 lit camera angles) back onto
      // the UV texture, keeping each texel's MOST-FRONTAL view. A stronger off-the-shelf refiner
      // replaces the analytic skin while the recovered geometry/rig stay untouched (appearance-only).
      auto uv_final = uvtex.reshape({T * T, 3}).clone();  // [T^2,3]
      if (args.has("reproject-dir")) {
        namespace Fn = torch::nn::functional;
        const std::string rd = args.require("reproject-dir");
        // Recompute the texel surface points/normals/head-centroid here (the inner-block copies are
        // out of scope) from the broader-scope UV solve outputs.
        const auto rm = (uvmask.reshape({T * T}) > 0.5F);
        const auto rfp = uvpos.index({rm});                          // [P,3]
        const auto rfn = uvgn.index({rm});                           // [P,3]
        const int64_t RP = rfp.size(0);
        const auto rhthr = torch::quantile(id_verts.select(1, 1), 0.88).item<float>();
        const auto rhmask = (id_verts.select(1, 1).to(device) > rhthr).unsqueeze(1);
        const auto rhc = verts_dev.masked_select(rhmask).reshape({-1, 3}).mean(0);
        auto conf = torch::zeros({RP}, rfp.options());
        const auto pidx = rm.nonzero().squeeze(1);
        for (int k = 0; k < 5; ++k) {
          const int az = -40 + 20 * k;
          char nm[40];
          std::snprintf(nm, sizeof(nm), "/refine_%+03d.png", az);
          const std::string path = rd + nm;
          if (!std::filesystem::exists(path)) { NCG_LOG_WARN("reproject: missing {}", path); continue; }
          const auto img = ncg::io::load_image(path, 3).to(device);  // [3,H,W] in [0,1]
          const int sz = static_cast<int>(img.size(2));
          const auto cam = ncg::runtime::Camera::orbit(rhc, 0.42F, static_cast<float>(az), 5.0F, 28.0F,
                                                       sz, sz, device);
          torch::Tensor uvp, depth;
          cam.project(rfp, uvp, depth);
          const auto ncam = torch::matmul(rfn, cam.R.t());
          const auto front = torch::relu(-ncam.select(1, 2)) * (depth > 0).to(rfp.dtype());
          const auto gx = uvp.select(1, 0) / (sz - 1) * 2 - 1;
          const auto gy = uvp.select(1, 1) / (sz - 1) * 2 - 1;
          const auto grid = torch::stack({gx, gy}, 1).view({1, RP, 1, 2});
          const auto samp = Fn::grid_sample(img.unsqueeze(0), grid,
              Fn::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros)
                  .align_corners(true)).view({3, RP}).t();           // [P,3]
          const auto inb = ((uvp.select(1, 0) >= 0) & (uvp.select(1, 0) <= sz - 1) &
                            (uvp.select(1, 1) >= 0) & (uvp.select(1, 1) <= sz - 1)).to(rfp.dtype());
          const auto w = front * inb;
          const auto better = (w > conf).to(rfp.dtype()).unsqueeze(1);
          const auto old = uv_final.index_select(0, pidx);
          uv_final.index_copy_(0, pidx, samp * better + old * (1.0F - better));
          conf = torch::maximum(conf, w);
          NCG_LOG_INFO("face: reprojected refined view az={:+d}", az);
        }
        // VERIFY render: the baked texture as a texel cloud (front + sides) so the bake can be
        // checked by eye without a glTF viewer. Uses the reprojected colours directly.
        ncg::recon::GaussianCloud bc;
        bc.positions = rfp;
        bc.colors = uv_final.index_select(0, pidx).clamp(0.0F, 1.0F);
        bc.scales = torch::full({RP, 3}, args.get_float("texel-scale", 0.0010F), rfp.options());
        bc.opacities = torch::ones({RP, 1}, rfp.options());
        bc.rotations = torch::zeros({RP, 4}, rfp.options());
        bc.rotations.select(1, 0).fill_(1.0F);
        for (int raz : {-20, 0, 20}) {
          const auto rcam = ncg::runtime::Camera::orbit(rhc, 0.42F, static_cast<float>(raz), 5.0F,
                                                        28.0F, 512, 512, device);
          char rn[40];
          std::snprintf(rn, sizeof(rn), "_reproj_%+03d.png", raz);
          ncg::io::save_png(prefix + rn, ncg::runtime::render_gaussians(bc, rcam).image);
        }
        NCG_LOG_INFO("face: baked external refined views -> UV texture (reproject-dir); "
                     "verify renders -> {}_reproj_*.png", prefix);
      }
      ncg::io::save_png(prefix + "_albedo_uv.png",
                        uv_final.reshape({T, T, 3}).permute({2, 0, 1}).contiguous().detach());
      ncg::io::save_png(prefix + "_normal_uv.png", uvnrm.permute({2, 0, 1}).contiguous().detach());
      ncg::io::save_npy(prefix + "_uvcoords.npy", model.uv_coords().to(at::kCPU).contiguous());
      ncg::io::save_npy(prefix + "_uvfaces.npy", model.uv_faces().to(at::kInt).contiguous());
      NCG_LOG_INFO("face: wrote {}x{} per-texel UV albedo + normal map -> {}_albedo_uv.png", T, T,
                   prefix);

      // ---- hair shell: give the dark scalp VOLUME (the bald scalp is the at-a-glance tell) -------
      // Segment hair from skin using the recovered albedo (hair = dark, on the upper head), then
      // offset those vertices outward along the normal. The skin↔hair boundary (no offset → offset)
      // forms a hairline ridge; the texture there is already the photo-sampled hair color. v1: a
      // volume shell, not strands — but it reads as hair instead of a skull.
      const auto vn_dev = ncg::mesh::compute_vertex_normals(ncg::mesh::TriMesh{id_verts, faces})
                              .to(device);                                        // [V,3]
      const auto yv2 = id_verts.select(1, 1).to(device);                         // [V]
      const auto hthr2 = torch::quantile(id_verts.select(1, 1), 0.80).item<float>();
      const auto head_w = torch::sigmoid((yv2 - hthr2) * 40.0F);                 // soft upper-head
      const auto dark = torch::sigmoid((0.30F - albedo.mean(1)) * 16.0F);        // dark = hair-like
      const auto hair_w = (head_w * dark).unsqueeze(1);                          // [V,1] in [0,1]
      const float thick = args.get_float("hair-thick", 0.03F);                   // ~3 cm shell
      const auto verts_hair = verts_dev + vn_dev * hair_w * thick;               // [V,3]
      ncg::io::save_npy(prefix + "_verts_hair.npy", verts_hair.to(at::kCPU).contiguous());

      // ---- engine-ready textured + rigged glTF (the deployable asset) ----------------------------
      ncg::body::SmplxParams zp;  // rest-pose joints for the rig (template body matches id_verts)
      zp.betas = torch::zeros({1, model.num_betas()}, verts_dev.options());
      zp.pose_aa = torch::zeros({1, model.num_joints(), 3}, verts_dev.options());
      zp.transl = torch::zeros({1, 3}, verts_dev.options());
      const auto joints = model.forward(zp).joints.squeeze(0);                   // [J,3]
      const auto hair_cpu = verts_hair.to(at::kCPU);
      const auto hnrm = ncg::mesh::compute_vertex_normals(ncg::mesh::TriMesh{hair_cpu, faces});

      // ---- EYEBALL GEOMETRY (--eyes): SMPL-X has no eyeballs, so refined eyes have nowhere to land
      // (dark sockets / smudges, proven unfixable by texture strength). Add a sphere at each SMPL-X
      // eye joint (23=left, 24=right), skinned to the head joint, coloured by a reserved brown texel,
      // so the rig has real 3D eyes. Best-effort: textured meshes can't be rendered server-side, verify
      // in a glТF viewer. --eyes 0 to disable.
      // All on CPU (the eyeball tensors are CPU; cat requires one device) + remember base dtypes.
      auto g_verts = hair_cpu.to(at::kCPU);     // [V,3]
      auto g_faces = faces.to(at::kCPU);        // [F,3]
      auto g_norm = hnrm.to(at::kCPU);          // [V,3]
      auto g_uv = model.uv_coords().to(at::kCPU);     // [n_uv,2]
      auto g_uvf = model.uv_faces().to(at::kCPU);     // [F,3]
      auto g_lbs = model.lbs_weights().to(at::kCPU);  // [V,J]
      auto tex_uv = uv_final;                   // [T^2,3] (may paint the eye texel)
      if (args.get_int("eyes", 1) != 0 && joints.size(0) > 24) {
        const auto jc = joints.to(at::kCPU);
        const auto eyes_c = torch::stack({jc[23], jc[24]}, 0);                  // [2,3] eyeball centers
        const float ir = std::max(0.008F, 0.18F * (jc[23] - jc[24]).norm().item<float>());
        const int nlat = 12, nlon = 16;
        std::vector<float> sv;
        std::vector<int64_t> sf;
        for (int i = 0; i <= nlat; ++i) {
          const float th = static_cast<float>(M_PI) * i / nlat;
          for (int j = 0; j <= nlon; ++j) {
            const float ph = 2.0F * static_cast<float>(M_PI) * j / nlon;
            sv.push_back(std::sin(th) * std::cos(ph));
            sv.push_back(std::cos(th));
            sv.push_back(std::sin(th) * std::sin(ph));
          }
        }
        const int Wp = nlon + 1;
        for (int i = 0; i < nlat; ++i)
          for (int j = 0; j < nlon; ++j) {
            const int a = i * Wp + j, b = a + 1, c = a + Wp, d = c + 1;
            sf.push_back(a); sf.push_back(c); sf.push_back(b);
            sf.push_back(b); sf.push_back(c); sf.push_back(d);
          }
        const auto sV = torch::from_blob(sv.data(), {static_cast<int64_t>(sv.size()) / 3, 3},
                                         torch::kFloat).clone();
        const auto sF = torch::from_blob(sf.data(), {static_cast<int64_t>(sf.size()) / 3, 3},
                                         torch::kLong).clone();
        const int64_t N = sV.size(0), Jn = g_lbs.size(1);
        // brown eye texel at UV (0.01,0.01) -> pixel (row=(1-v)*T, col=u*T) per the texture convention.
        auto tx = tex_uv.reshape({T, T, 3}).clone();
        const int pr = static_cast<int>(0.99F * T), pc = static_cast<int>(0.01F * T);
        for (int di = -2; di <= 2; ++di)
          for (int dj = -2; dj <= 2; ++dj) {
            const int rr = std::clamp(pr + di, 0, static_cast<int>(T) - 1);
            const int cc = std::clamp(pc + dj, 0, static_cast<int>(T) - 1);
            tx[rr][cc][0] = 0.32F; tx[rr][cc][1] = 0.22F; tx[rr][cc][2] = 0.17F;
          }
        tex_uv = tx.reshape({T * T, 3});
        const auto eye_uv = torch::tensor({0.01F, 0.01F}).reshape({1, 2}).expand({N, 2}).contiguous();
        std::vector<torch::Tensor> Vs{g_verts}, Ns{g_norm}, UVs{g_uv}, Ls{g_lbs}, Fs{g_faces}, UVFs{g_uvf};
        int64_t vbase = g_verts.size(0), uvbase = g_uv.size(0);
        for (int e = 0; e < 2; ++e) {
          Vs.push_back((sV * ir + eyes_c[e]).to(g_verts.scalar_type()));        // [N,3]
          Ns.push_back(sV.to(g_norm.scalar_type()));                            // outward normals
          UVs.push_back(eye_uv.to(g_uv.scalar_type()));                         // all -> brown texel
          auto lb = torch::zeros({N, Jn}); lb.select(1, 15).fill_(1.0F);        // skin to head joint
          Ls.push_back(lb.to(g_lbs.scalar_type()));
          Fs.push_back((sF + vbase).to(g_faces.scalar_type()));
          UVFs.push_back((sF + uvbase).to(g_uvf.scalar_type()));
          vbase += N; uvbase += N;
        }
        g_verts = torch::cat(Vs, 0); g_norm = torch::cat(Ns, 0); g_uv = torch::cat(UVs, 0);
        g_lbs = torch::cat(Ls, 0); g_faces = torch::cat(Fs, 0); g_uvf = torch::cat(UVFs, 0);
        ncg::io::save_png(prefix + "_albedo_uv.png",
                          tex_uv.reshape({T, T, 3}).permute({2, 0, 1}).contiguous().detach());
        NCG_LOG_INFO("face: added 2 eyeballs ({} verts each, r={:.3f}m) skinned to head joint", N, ir);
      }

      ncg::mesh::write_glb_textured(g_verts, g_faces, g_norm, g_uv, g_uvf,
                                    joints, model.parents(), g_lbs,
                                    prefix + "_albedo_uv.png", prefix + "_face.glb",
                                    prefix + "_normal_uv.png");
      NCG_LOG_INFO("face: wrote rigged + textured engine asset -> {}_face.glb", prefix);
    }

    // Personalized identity mesh (rest pose) + its normals, framed on the head for a portrait.
    const auto verts = id_verts.to(device);                                    // [V,3]
    ncg::mesh::TriMesh cm{id_verts, faces};
    const auto cnrm = ncg::mesh::compute_vertex_normals(cm).to(device);
    const auto pvs = ncg::recon::per_vertex_scale(verts, 0.75F);
    const auto yv = id_verts.select(1, 1);
    const auto hthr = torch::quantile(yv, 0.88).item<float>();
    const auto hmask = (yv.to(device) > hthr).unsqueeze(1);                     // [V,1]
    const auto head_c = verts.masked_select(hmask).reshape({-1, 3}).mean(0);    // head centroid

    auto portrait = [&](const torch::Tensor& colors, float az_deg, float el_deg) {
      auto cloud = ncg::recon::gaussians_on_body(verts, 0.008F, colors.clamp(0.0, 1.0), pvs);
      cloud.to_(device);
      const auto cam = ncg::runtime::Camera::orbit(head_c, 0.42F, az_deg, el_deg, 28.0F, 512, 512,
                                                   device);
      return ncg::runtime::render_gaussians(cloud, cam).image;
    };

    // (a) the recovered skin (flat albedo) — front + a few yaws to show it's a real 3D face.
    ncg::io::save_png(prefix + "_face_albedo.png", portrait(albedo, 0.0F, 5.0F));
    rec.log_image("face", "albedo_front", portrait(albedo, 0.0F, 5.0F));
    for (int k = 0; k < 5; ++k) {
      const float az = -40.0F + 20.0F * static_cast<float>(k);
      char nm[24];
      std::snprintf(nm, sizeof(nm), "view_%+03d", static_cast<int>(az));
      rec.log_image("face", nm, portrait(albedo, az, 5.0F));
    }
    // (b) relit under an orbiting novel light — the payoff of the albedo/light decomposition.
    const auto white = torch::ones({3}, verts.options());
    const float el = 20.0F * static_cast<float>(M_PI) / 180.0F;
    for (int k = 0; k < 6; ++k) {
      const float az = 2.0F * static_cast<float>(M_PI) * static_cast<float>(k) / 6.0F;
      const auto dir = torch::tensor(
          {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)}, verts.options());
      const auto L = ncg::recon::sh_directional_light(dir, white, 0.30F);
      char nm[24];
      std::snprintf(nm, sizeof(nm), "relit_%03d", k);
      rec.log_image("face", nm, portrait(ncg::recon::shade_sh(albedo, L, cnrm), 0.0F, 5.0F));
    }
    NCG_LOG_INFO("face: wrote {}_face_albedo.png + yaw/relit frames in {}", prefix,
                 rec.dir().string());
  }

  NCG_LOG_INFO("face: wrote personalized neutral identity -> {}_identity.obj (vs {}_mean.obj)",
               prefix, prefix);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  ncg::init_logging();
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: ncg_cli "
                 "<pipeline|render|turntable|select|fitimg|fit|fuse|relight|export|benchmark|"
                 "runtime|nerf|style|avatar|gate|geom|complete|face> [--flags]\n");
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
    if (cmd == "gate") return cmd_gate(args);
    if (cmd == "geom") return cmd_geom(args);
    if (cmd == "complete") return cmd_complete(args);
    if (cmd == "mvtest") return cmd_mvtest(args);
    if (cmd == "mvbench") return cmd_mvbench(args);
    if (cmd == "attribute") return cmd_attribute(args);
    if (cmd == "face") return cmd_face(args);
    if (cmd == "nerf") return cmd_nerf(args);
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ncg_cli %s error: %s\n", cmd.c_str(), e.what());
    return 1;
  }
}
