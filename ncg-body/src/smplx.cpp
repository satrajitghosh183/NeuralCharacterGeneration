#include <ncg/body/smplx.hpp>

#include <ncg/core/error.hpp>
#include <ncg/io/safetensors.hpp>

#include <vector>

namespace ncg::body {
namespace {

// Axis-angle [N,3] -> rotation matrices [N,3,3] (Rodrigues).
Tensor batch_rodrigues(const Tensor& aa) {
  const auto angle = aa.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-8);  // [N,1]
  const auto axis = aa / angle;                                               // [N,3]
  const auto cos = torch::cos(angle).unsqueeze(-1);                           // [N,1,1]
  const auto sin = torch::sin(angle).unsqueeze(-1);                           // [N,1,1]

  const auto rx = axis.select(1, 0);
  const auto ry = axis.select(1, 1);
  const auto rz = axis.select(1, 2);
  const auto z = torch::zeros_like(rx);
  // Skew-symmetric cross-product matrix K.
  const auto K = torch::stack({z, -rz, ry, rz, z, -rx, -ry, rx, z}, 1).reshape({-1, 3, 3});
  const auto I = torch::eye(3, aa.options()).unsqueeze(0);
  return I + sin * K + (1.0 - cos) * torch::bmm(K, K);
}

// [N,3,3] + [N,3] -> homogeneous [N,4,4].
Tensor transform_mat(const Tensor& R, const Tensor& t) {
  const int64_t n = R.size(0);
  const auto rt = torch::cat({R, t.unsqueeze(-1)}, /*dim=*/2);  // [N,3,4]
  auto bottom = torch::zeros({n, 1, 4}, R.options());
  bottom.select(2, 3).fill_(1.0);
  return torch::cat({rt, bottom}, /*dim=*/1);  // [N,4,4]
}

}  // namespace

SmplxModel SmplxModel::load(const std::string& path, at::Device device) {
  auto st = io::SafeTensors::open(path);
  auto need = [&](const char* name) {
    NCG_CHECK(st.has(name), "SmplxModel: '{}' missing key '{}'", path, name);
    return st.view(name).to(device);
  };

  SmplxModel m;
  m.v_template_ = need("v_template").to(at::kFloat);
  m.shapedirs_ = need("shapedirs").to(at::kFloat);
  m.posedirs_ = need("posedirs").to(at::kFloat);
  m.J_regressor_ = need("J_regressor").to(at::kFloat);
  m.lbs_weights_ = need("lbs_weights").to(at::kFloat);
  m.parents_ = need("parents").to(at::kLong);

  NCG_CHECK(m.v_template_.dim() == 2 && m.v_template_.size(1) == 3, "SmplxModel: bad v_template");
  NCG_CHECK(m.parents_.dim() == 1, "SmplxModel: bad parents");
  return m;
}

SmplxParams SmplxModel::neutral_params(int64_t batch) const {
  SmplxParams p;
  const auto opts = v_template_.options();
  p.betas = torch::zeros({batch, num_betas()}, opts);
  p.pose_aa = torch::zeros({batch, num_joints(), 3}, opts);
  p.transl = torch::zeros({batch, 3}, opts);
  return p;
}

SmplxOutput SmplxModel::forward(const SmplxParams& p) const {
  const int64_t V = num_verts();
  const int64_t J = num_joints();
  const int64_t B = p.betas.size(0);

  auto pose = p.pose_aa;
  if (pose.dim() == 2) pose = pose.reshape({B, J, 3});
  NCG_CHECK(pose.size(1) == J, "SmplxModel: pose has {} joints, expected {}", pose.size(1), J);

  // 1. Shape blend.
  const auto v_shaped =
      v_template_.unsqueeze(0) + torch::einsum("vck,bk->bvc", {shapedirs_, p.betas});  // [B,V,3]

  // 2. Rest joints.
  const auto J_rest = torch::einsum("jv,bvc->bjc", {J_regressor_, v_shaped});  // [B,J,3]

  // 3. Pose rotation matrices + pose blend.
  const auto R = batch_rodrigues(pose.reshape({B * J, 3})).reshape({B, J, 3, 3});
  const auto eye3 = torch::eye(3, R.options()).view({1, 1, 3, 3});
  const auto pose_feature = (R.slice(1, 1, J) - eye3).reshape({B, (J - 1) * 9});
  const auto v_posed = v_shaped + torch::einsum("vck,bk->bvc", {posedirs_, pose_feature});

  // 4. Kinematic chain -> global transforms.
  auto rel_joints = J_rest.clone();
  const auto parents_cpu = parents_.to(at::kCPU);
  const auto* par = parents_cpu.data_ptr<int64_t>();
  for (int64_t j = 1; j < J; ++j) {
    rel_joints.select(1, j) -= J_rest.select(1, par[j]);
  }
  const auto tmats =
      transform_mat(R.reshape({B * J, 3, 3}), rel_joints.reshape({B * J, 3})).reshape({B, J, 4, 4});

  std::vector<Tensor> chain(static_cast<size_t>(J));
  chain[0] = tmats.select(1, 0);
  for (int64_t j = 1; j < J; ++j) {
    chain[static_cast<size_t>(j)] = torch::matmul(chain[static_cast<size_t>(par[j])], tmats.select(1, j));
  }
  const auto transforms = torch::stack(chain, /*dim=*/1);  // [B,J,4,4]
  const auto posed_joints = transforms.index({"...", torch::indexing::Slice(0, 3), 3});  // [B,J,3]

  // 5. Relative transforms for skinning (subtract rest-pose bone offset).
  auto j_homo = torch::cat({J_rest, torch::zeros({B, J, 1}, J_rest.options())}, -1).unsqueeze(-1);
  const auto bone = torch::matmul(transforms, j_homo);  // [B,J,4,1]
  auto pad = torch::zeros({B, J, 4, 4}, transforms.options());
  pad.index_put_({"...", torch::indexing::Slice(), 3}, bone.squeeze(-1));
  const auto rel_transforms = transforms - pad;  // [B,J,4,4]

  // 6. Skin vertices.
  const auto W = lbs_weights_.unsqueeze(0).expand({B, V, J});  // [B,V,J]
  const auto T = torch::matmul(W, rel_transforms.reshape({B, J, 16})).reshape({B, V, 4, 4});
  const auto v_homo = torch::cat({v_posed, torch::ones({B, V, 1}, v_posed.options())}, -1);  // [B,V,4]
  const auto v_skinned = torch::matmul(T, v_homo.unsqueeze(-1)).squeeze(-1).index(
      {"...", torch::indexing::Slice(0, 3)});  // [B,V,3]

  SmplxOutput out;
  out.vertices = v_skinned + p.transl.unsqueeze(1);
  out.joints = posed_joints + p.transl.unsqueeze(1);
  out.vertex_transforms = T;  // [B,V,4,4]
  return out;
}

}  // namespace ncg::body
