#include <ncg/rig/rig.hpp>

#include <ncg/core/error.hpp>

#include <nlohmann/json.hpp>

#include <fstream>

namespace ncg::rig {

RiggedMesh make_rigged(const Tensor& vertices, const Tensor& faces, const Tensor& joints,
                       const Tensor& parents, const Tensor& skin_weights) {
  NCG_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "make_rigged: vertices must be [V,3]");
  NCG_CHECK(faces.dim() == 2 && faces.size(1) == 3, "make_rigged: faces must be [F,3]");
  const int64_t V = vertices.size(0);
  const int64_t J = joints.size(0);
  NCG_CHECK(joints.dim() == 2 && joints.size(1) == 3, "make_rigged: joints must be [J,3]");
  NCG_CHECK(parents.dim() == 1 && parents.size(0) == J, "make_rigged: parents must be [J]");
  NCG_CHECK(skin_weights.dim() == 2 && skin_weights.size(0) == V && skin_weights.size(1) == J,
            "make_rigged: skin_weights must be [V,J]");

  RiggedMesh m;
  m.vertices = vertices.to(at::kFloat).contiguous();
  m.faces = faces.to(at::kLong).contiguous();
  m.joints = joints.to(at::kFloat).contiguous();
  m.parents = parents.to(at::kLong).contiguous();
  m.skin_weights = skin_weights.to(at::kFloat).contiguous();
  return m;
}

Tensor transfer_skinning(const Tensor& target_verts, const Tensor& source_verts,
                         const Tensor& source_weights) {
  NCG_CHECK(target_verts.dim() == 2 && target_verts.size(1) == 3,
            "transfer_skinning: target_verts must be [V,3]");
  NCG_CHECK(source_verts.dim() == 2 && source_verts.size(1) == 3,
            "transfer_skinning: source_verts must be [M,3]");
  NCG_CHECK(source_weights.dim() == 2 && source_weights.size(0) == source_verts.size(0),
            "transfer_skinning: source_weights must be [M,J] matching source_verts");

  const auto tv = target_verts.to(at::kCPU, at::kFloat).contiguous();
  const auto sv = source_verts.to(at::kCPU, at::kFloat).contiguous();
  const auto sw = source_weights.to(at::kCPU, at::kFloat).contiguous();

  const auto dist = torch::cdist(tv, sv);                 // [V,M]
  const auto nearest = std::get<1>(dist.min(/*dim=*/1));  // [V] index of closest source vert
  return sw.index_select(0, nearest).contiguous();        // [V,J]
}

RiggedMesh autorig(const Tensor& /*vertices*/, const Tensor& /*faces*/) {
  NCG_THROW("autorig: learned auto-rigging (UniRig) is not ported yet — vendor UniRig weights, "
            "or use make_rigged() to inherit the SMPL-X rig for bodies.");
}

void export_rigged(const RiggedMesh& mesh, const std::string& path) {
  NCG_CHECK(mesh.vertices.defined() && mesh.faces.defined(), "export_rigged: empty mesh");

  // Geometry -> OBJ.
  {
    const auto v = mesh.vertices.to(at::kCPU, at::kFloat).contiguous();
    const auto f = mesh.faces.to(at::kCPU, at::kLong).contiguous();
    std::ofstream os(path + ".obj");
    NCG_CHECK(os.good(), "export_rigged: cannot write '{}.obj'", path);
    const auto* vp = v.data_ptr<float>();
    for (int64_t i = 0; i < v.size(0); ++i)
      os << "v " << vp[i * 3] << ' ' << vp[i * 3 + 1] << ' ' << vp[i * 3 + 2] << '\n';
    const auto* fp = f.data_ptr<int64_t>();
    for (int64_t i = 0; i < f.size(0); ++i)
      os << "f " << fp[i * 3] + 1 << ' ' << fp[i * 3 + 1] + 1 << ' ' << fp[i * 3 + 2] + 1 << '\n';
  }

  // Skeleton + skinning -> JSON sidecar.
  {
    const auto j = mesh.joints.to(at::kCPU, at::kFloat).contiguous();
    const auto p = mesh.parents.to(at::kCPU, at::kLong).contiguous();
    const auto w = mesh.skin_weights.to(at::kCPU, at::kFloat).contiguous();
    nlohmann::json js;
    js["num_joints"] = j.size(0);
    js["num_verts"] = w.size(0);
    std::vector<std::array<float, 3>> joints;
    const auto* jp = j.data_ptr<float>();
    for (int64_t i = 0; i < j.size(0); ++i) joints.push_back({jp[i * 3], jp[i * 3 + 1], jp[i * 3 + 2]});
    js["joints"] = joints;
    const auto* pp = p.data_ptr<int64_t>();
    js["parents"] = std::vector<int64_t>(pp, pp + p.size(0));
    // skin weights as a [V,J] nested array (sparse export is a later optimization).
    js["skin_weights_shape"] = {w.size(0), w.size(1)};
    const auto* wp = w.data_ptr<float>();
    js["skin_weights"] = std::vector<float>(wp, wp + w.numel());

    std::ofstream os(path + ".rig.json");
    NCG_CHECK(os.good(), "export_rigged: cannot write '{}.rig.json'", path);
    os << js.dump();
  }
}

}  // namespace ncg::rig
