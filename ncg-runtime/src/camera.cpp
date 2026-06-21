#include <ncg/runtime/camera.hpp>

#include <ncg/core/error.hpp>

#include <cmath>

namespace ncg::runtime {

void Camera::project(const Tensor& points, Tensor& uv, Tensor& depth) const {
  NCG_CHECK(points.dim() == 2 && points.size(1) == 3, "Camera::project: points must be [N,3]");
  const auto pts = points.to(R.device(), at::kFloat);
  // X_cam = points @ R^T + t
  const auto xc = torch::matmul(pts, R.t()) + t.unsqueeze(0);  // [N,3]
  const auto x = xc.select(1, 0);
  const auto y = xc.select(1, 1);
  const auto z = xc.select(1, 2);
  depth = z;
  const auto u = fx * x / z + cx;
  const auto v = fy * y / z + cy;
  uv = torch::stack({u, v}, /*dim=*/1);  // [N,2]
}

Camera Camera::orbit(const Tensor& center, float radius, float azimuth_deg, float elevation_deg,
                     float fov_y_deg, int width, int height, at::Device device) {
  const double az = static_cast<double>(azimuth_deg) * M_PI / 180.0;
  const double el = static_cast<double>(elevation_deg) * M_PI / 180.0;
  const double r = static_cast<double>(radius);

  const auto c = center.to(at::kCPU, at::kFloat).reshape({3});
  const auto ca = c.accessor<float, 1>();

  // Eye on a sphere around the center.
  const float dx = static_cast<float>(r * std::cos(el) * std::sin(az));
  const float dy = static_cast<float>(r * std::sin(el));
  const float dz = static_cast<float>(r * std::cos(el) * std::cos(az));
  const auto eye = torch::tensor({ca[0] + dx, ca[1] + dy, ca[2] + dz});

  // Look-at basis. forward = into scene (camera +z); world up = +y.
  auto normalize = [](Tensor v) { return v / v.norm().clamp_min(1e-8); };
  const auto forward = normalize(c - eye);
  const auto world_up = torch::tensor({0.0F, 1.0F, 0.0F});
  const auto right = normalize(torch::cross(forward, world_up, /*dim=*/-1));
  const auto down = torch::cross(forward, right, /*dim=*/-1);  // y points down in image space

  // Rows of R are the camera axes expressed in world coords.
  const auto R = torch::stack({right, down, forward}, /*dim=*/0);  // [3,3]
  const auto t = -torch::matmul(R, eye);                           // [3]

  Camera cam;
  cam.height = height;
  cam.width = width;
  cam.fy = static_cast<float>((height / 2.0) /
                              std::tan(static_cast<double>(fov_y_deg) * M_PI / 360.0));
  cam.fx = cam.fy;
  cam.cx = static_cast<float>(width) / 2.0F;
  cam.cy = static_cast<float>(height) / 2.0F;
  cam.R = R.to(device);
  cam.t = t.to(device);
  return cam;
}

std::vector<Camera> orbit_trajectory(const Tensor& center, float radius, float elevation_deg,
                                     int frames, float fov_y_deg, int width, int height,
                                     at::Device device) {
  NCG_CHECK(frames > 0, "orbit_trajectory: frames must be positive");
  std::vector<Camera> cams;
  cams.reserve(static_cast<size_t>(frames));
  for (int i = 0; i < frames; ++i) {
    const float az = 360.0F * static_cast<float>(i) / static_cast<float>(frames);
    cams.push_back(Camera::orbit(center, radius, az, elevation_deg, fov_y_deg, width, height,
                                 device));
  }
  return cams;
}

}  // namespace ncg::runtime
