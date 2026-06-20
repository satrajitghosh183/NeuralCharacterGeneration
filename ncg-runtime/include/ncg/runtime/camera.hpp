#pragma once

#include <ncg/core/tensor.hpp>

#include <vector>

namespace ncg::runtime {

/// Pinhole camera. Extrinsics map world -> camera (x right, y down, z forward into scene).
struct Camera {
  float fx = 0.0F;
  float fy = 0.0F;
  float cx = 0.0F;
  float cy = 0.0F;
  Tensor R;  // [3,3] world->camera rotation
  Tensor t;  // [3]   world->camera translation
  int width = 0;
  int height = 0;

  /// Projects world points [N,3] to pixel coords uv [N,2] and camera-space depth [N].
  void project(const Tensor& points, Tensor& uv, Tensor& depth) const;

  /// Camera orbiting `center` ([3] tensor) at `radius`, looking at it. Angles in degrees;
  /// vertical field of view `fov_y_deg`. Square pixels (fx == fy).
  static Camera orbit(const Tensor& center, float radius, float azimuth_deg, float elevation_deg,
                      float fov_y_deg, int width, int height, at::Device device);
};

/// A turntable trajectory: `frames` cameras equally spaced in azimuth [0,360) at fixed radius
/// and elevation (Phase-5 eval / novel-view rendering).
std::vector<Camera> orbit_trajectory(const Tensor& center, float radius, float elevation_deg,
                                     int frames, float fov_y_deg, int width, int height,
                                     at::Device device);

}  // namespace ncg::runtime
