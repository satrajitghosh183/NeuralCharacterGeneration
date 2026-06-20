#include <ncg/mesh/extract.hpp>

#include <ncg/core/error.hpp>

#include <fstream>

namespace ncg::mesh {

void write_obj(const TriMesh& mesh, const std::string& path) {
  const auto v = mesh.vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto f = mesh.faces.to(at::kCPU, at::kLong).contiguous();
  std::ofstream os(path);
  NCG_CHECK(os.good(), "write_obj: cannot open '{}'", path);

  const auto* vp = v.data_ptr<float>();
  for (int64_t i = 0; i < v.size(0); ++i) {
    os << "v " << vp[i * 3 + 0] << ' ' << vp[i * 3 + 1] << ' ' << vp[i * 3 + 2] << '\n';
  }
  const auto* fp = f.data_ptr<int64_t>();
  for (int64_t i = 0; i < f.size(0); ++i) {
    // OBJ is 1-indexed.
    os << "f " << fp[i * 3 + 0] + 1 << ' ' << fp[i * 3 + 1] + 1 << ' ' << fp[i * 3 + 2] + 1 << '\n';
  }
  NCG_CHECK(os.good(), "write_obj: write error for '{}'", path);
}

void write_ply(const TriMesh& mesh, const std::string& path) {
  const auto v = mesh.vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto f = mesh.faces.to(at::kCPU, at::kLong).contiguous();
  std::ofstream os(path);
  NCG_CHECK(os.good(), "write_ply: cannot open '{}'", path);

  os << "ply\nformat ascii 1.0\n";
  os << "element vertex " << v.size(0) << "\n";
  os << "property float x\nproperty float y\nproperty float z\n";
  os << "element face " << f.size(0) << "\n";
  os << "property list uchar int vertex_indices\n";
  os << "end_header\n";

  const auto* vp = v.data_ptr<float>();
  for (int64_t i = 0; i < v.size(0); ++i) {
    os << vp[i * 3 + 0] << ' ' << vp[i * 3 + 1] << ' ' << vp[i * 3 + 2] << '\n';
  }
  const auto* fp = f.data_ptr<int64_t>();
  for (int64_t i = 0; i < f.size(0); ++i) {
    os << "3 " << fp[i * 3 + 0] << ' ' << fp[i * 3 + 1] << ' ' << fp[i * 3 + 2] << '\n';
  }
  NCG_CHECK(os.good(), "write_ply: write error for '{}'", path);
}

}  // namespace ncg::mesh
