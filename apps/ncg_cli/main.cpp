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
#include <ncg/fit/fit_image.hpp>
#include <ncg/io/image.hpp>
#include <ncg/mesh/extract.hpp>
#include <ncg/nerf/nerf.hpp>
#include <ncg/recon/init_from_body.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/rig/rig.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>
#include <ncg/select/selector.hpp>

#include <torch/torch.h>

#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

  auto nlf = ncg::body::Nlf::load(args.require("weights"), device);
  const auto image = ncg::io::load_image(args.require("image"), 3);
  const auto params = nlf.predict(image);
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
  const auto verts = model.forward(p).vertices.squeeze(0);

  auto cloud = ncg::recon::gaussians_on_body(verts, args.get_float("scale", 0.012F));
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

}  // namespace

int main(int argc, char** argv) {
  ncg::init_logging();
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: ncg_cli <pipeline|render|turntable|select|fitimg|fit|nerf> [--flags]\n");
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
    if (cmd == "nerf") return cmd_nerf(args);
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ncg_cli %s error: %s\n", cmd.c_str(), e.what());
    return 1;
  }
}
