// ncg_viewer — Phase-1 vertical slice: SMPL-X body -> Gaussians -> forward-splat render -> PNG.
// Proves the spine end to end on the H100. NLF (image -> pose) is not wired yet, so this
// renders the neutral T-pose body from an orbit camera.
//
//   ncg_viewer --smplx model.safetensors --out out.png [--width 512 --height 512]
//              [--azimuth 20 --elevation 10 --radius 2.5 --scale 0.012]
#include <args.hpp>

#include <ncg/body/smplx.hpp>
#include <ncg/core/device.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/io/image.hpp>
#include <ncg/recon/init_from_body.hpp>
#include <ncg/record/recorder.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
  ncg::init_logging();
  try {
    const ncg::app::Args args(argc, argv);
    const std::string smplx_path = args.require("smplx");
    const std::string out_path = args.get("out", "out.png");
    const int width = args.get_int("width", 512);
    const int height = args.get_int("height", 512);
    const float azimuth = args.get_float("azimuth", 20.0F);
    const float elevation = args.get_float("elevation", 10.0F);
    const float radius = args.get_float("radius", 2.5F);
    const float scale = args.get_float("scale", 0.012F);

    NCG_CHECK(ncg::cuda_available(), "ncg_viewer requires a CUDA device");
    const auto device = at::Device(at::kCUDA, 0);

    // Record every stage to runs/<name>_<timestamp>/.
    auto rec = ncg::record::Recorder::create("runs", args.get("run", "viewer"));
    rec.set_config("{\"app\":\"ncg_viewer\",\"smplx\":\"" + smplx_path + "\",\"width\":" +
                   std::to_string(width) + ",\"height\":" + std::to_string(height) +
                   ",\"azimuth\":" + std::to_string(azimuth) + ",\"elevation\":" +
                   std::to_string(elevation) + ",\"radius\":" + std::to_string(radius) +
                   ",\"scale\":" + std::to_string(scale) + "}");

    ncg::body::SmplxModel model = [&] {
      auto t = rec.time("body", "load");
      return ncg::body::SmplxModel::load(smplx_path, device);
    }();
    NCG_LOG_INFO("loaded SMPL-X: V={} J={} betas={}", model.num_verts(), model.num_joints(),
                 model.num_betas());
    rec.log_scalar("body", "num_verts", static_cast<double>(model.num_verts()));
    rec.log_scalar("body", "num_joints", static_cast<double>(model.num_joints()));

    ncg::Tensor verts;
    {
      auto t = rec.time("body", "forward");
      verts = model.forward(model.neutral_params(1)).vertices.squeeze(0);  // [V,3]
    }

    auto cloud = ncg::recon::gaussians_on_body(verts, scale);
    cloud.to_(device);
    rec.log_scalar("recon", "num_gaussians", static_cast<double>(cloud.size()));

    const auto center = verts.mean(0);
    const auto cam = ncg::runtime::Camera::orbit(center, radius, azimuth, elevation,
                                                 /*fov_y_deg=*/50.0F, width, height, device);

    ncg::runtime::RenderOutput render;
    {
      auto t = rec.time("render", "splat");
      render = ncg::runtime::render_gaussians(cloud, cam, {0.0F, 0.0F, 0.0F});
    }

    ncg::io::save_png(out_path, render.image);
    rec.log_image("render", "rgb", render.image);
    rec.log_image("render", "alpha", render.alpha);

    const double coverage = render.alpha.mean().item<double>();
    const double checksum = render.image.sum().item<double>();
    rec.log_scalar("render", "coverage", coverage);
    rec.log_scalar("render", "checksum", checksum);
    NCG_LOG_INFO("wrote {} ({}x{}) coverage={:.4f} checksum={:.3f} | run={}", out_path, width,
                 height, coverage, checksum, rec.dir().string());
    NCG_CHECK(coverage > 1e-3, "render produced an empty image (coverage {:.5f})", coverage);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ncg_viewer error: %s\n", e.what());
    return 1;
  }
}
