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
                                torch::Tensor& normal_out) {
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
    const auto spec = torch::sigmoid((lum - 0.78F) * 14.0F) * torch::sigmoid((0.22F - sat) * 14.0F);
    w_l.push_back(tw * inb * (1.0F - 0.92F * spec));              // [T^2]
  }
  ncg::recon::InverseRenderConfig ic;
  ic.iterations = 60;
  ic.robust = true;
  const auto ir = ncg::recon::solve_inverse_render(torch::stack(obs_l, 0), torch::stack(nrm_l, 0),
                                                   torch::stack(w_l, 0), ic);
  auto albedo = torch::nan_to_num(ir.albedo).clamp(0.0F, 1.0F);     // [T^2,3]
  const auto obs_mean = torch::stack(obs_l, 0).mean(0).mean(0).clamp_min(1e-3F);
  const auto alb_mean = albedo.mean(0).clamp_min(1e-3F);
  albedo = (albedo * (obs_mean / alb_mean).view({1, 3})).clamp(0.0F, 1.0F);

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
        torch::Tensor uvmask, uvnrm;
        const auto uvtex =
            recover_uv_albedo(model, id_img, id_v2d, id_nrm, id_w, T, rest_v, uvmask, uvnrm);
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
    cfg.log_every = 50;
    cfg.dump_every = args.get_int("dump-every", 500);
    cfg.densify = args.get_int("densify", 0) != 0;
    auto result = ncg::fit::fit_avatar(model, betas0, frames, init_colors, cfg, &rec);
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

  NCG_LOG_INFO("geom: {} usable photos, {} dense points; building cotangent Laplacian ({} verts)…",
               lm_l.size(), landmarks2d.size(1), v_template.size(0));
  const auto lap = ncg::geom::cotangent_laplacian(v_template, faces);

  ncg::geom::GeomConfig cfg;
  cfg.iterations = args.get_int("iters", 20);
  cfg.lap_weight = args.get_float("lap", 5.0F);
  cfg.mag_weight = args.get_float("mag", 0.5F);
  const auto R = ncg::geom::solve_geometry(v_template, id_dirs, ex_dirs, faces, lap, assoc, bary,
                                           landmarks2d, conf, w_prior, cfg);

  const auto beta = R.beta;
  const auto id_verts = v_template + torch::einsum("vck,k->vc", {id_dirs, beta});  // identity (β only)
  const auto pers = id_verts + R.delta_v;                                          // + Δv
  const auto prefix = (rec.dir() / args.get("out-prefix", "geom")).string();
  ncg::mesh::write_obj({pers, faces}, prefix + "_personalized.obj");
  ncg::mesh::write_obj({id_verts, faces}, prefix + "_identity.obj");
  ncg::mesh::write_obj({v_template, faces}, prefix + "_mean.obj");
  ncg::io::save_npy(prefix + "_obs.npy", R.obs.contiguous());
  ncg::io::save_npy(prefix + "_delta_v.npy", R.delta_v.contiguous());

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
  const int samp_res = args.get_int("sample-res", 1024);
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
  cfg.id_ridge = std::stof(args.get("id-ridge", "0.01"));
  cfg.expr_ridge = std::stof(args.get("expr-ridge", "0.1"));
  cfg.robust = args.get_int("robust", 1) != 0;
  const auto R = ncg::recon::solve_face_identity(base_lm, id_basis, expr_basis, landmarks2d, cfg);

  // Personalized NEUTRAL identity mesh: v_template + id_dirs · β  (expression set to 0 = neutral).
  const auto beta = R.id_shape.to(at::kCPU).to(at::kFloat);                         // [n_id]
  const auto id_verts = v_template + torch::einsum("vck,k->vc", {id_dirs, beta});   // [V,3]

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
      torch::Tensor uvmask, uvnrm;
      const auto uvtex =
          recover_uv_albedo(model, uv_img, uv_v2d, nrm_l, viss_t, T, verts_dev, uvmask, uvnrm);
      ncg::io::save_png(prefix + "_albedo_uv.png", uvtex.permute({2, 0, 1}).contiguous().detach());
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
      ncg::mesh::write_glb_textured(hair_cpu, faces, hnrm, model.uv_coords(), model.uv_faces(),
                                    joints, model.parents(), model.lbs_weights(),
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
                 "runtime|nerf|style|avatar|face> [--flags]\n");
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
    if (cmd == "face") return cmd_face(args);
    if (cmd == "nerf") return cmd_nerf(args);
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ncg_cli %s error: %s\n", cmd.c_str(), e.what());
    return 1;
  }
}
